"""Offline checks for deterministic, disk-backed comment reconciliation."""

import copy
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from parquet_loaders.comments import COMMENT_SCHEMA
from parquet_loaders.staging import CommentStaging


def comment(comment_id="t1_child", post_id="t3_example", **values):
    return {
        "comment_id": comment_id,
        "post_id": post_id,
        "parent_comment_id": None,
        "original_parent_id": post_id,
        "author": None,
        "created_utc": None,
        "score": None,
        "content_available": True,
        "source_id": 2,
        **values,
    }


class CommentStagingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.stage = CommentStaging(self.root / "staging.sqlite")
        self.addCleanup(self.stage.close)

    def add(self, rows, task="all", split="train", stage=None):
        (stage or self.stage).add_comments(rows, task=task, split=split)

    def rows(self, stage=None):
        return [row for batch in (stage or self.stage).iter_batches(batch_size=2) for row in batch]

    def test_duplicate_actual_rows_produce_one_comment(self):
        row = comment(author="writer", score=0)
        self.add([row, row])
        self.add([row])
        self.assertEqual(self.rows(), [row])

    def test_full_metadata_wins_and_fallback_fills_nulls_in_either_order(self):
        full = comment(author=None, score=0, created_utc=1441131871.5)
        fallback = comment(author="Café 🌱", score=42, created_utc=None, source_id=5)
        expected = {**full, "author": "Café 🌱"}
        outcomes = []
        for index, candidates in enumerate((
            ((full, "all"), (fallback, "pair_task")),
            ((fallback, "pair_task"), (full, "all")),
        )):
            with CommentStaging(self.root / f"order-{index}.sqlite") as stage:
                for row, task in candidates:
                    self.add([row], task=task, stage=stage)
                outcomes.append(self.rows(stage))
        self.assertEqual(outcomes, [[expected], [expected]])

    def test_same_task_can_fill_unknown_attributes(self):
        self.add([comment()])
        self.add([comment(author="writer", score=0)])
        self.assertEqual(self.stage.get_comment("t1_child"), comment(author="writer", score=0))

    def test_same_task_conflicts_raise_instead_of_depending_on_order(self):
        self.add([comment(score=1)])
        with self.assertRaisesRegex(ValueError, "Conflicting all score"):
            self.add([comment(score=2)])
        self.assertEqual(self.stage.get_comment("t1_child")["score"], 1)

    def test_missing_parent_becomes_placeholder_with_unknown_attributes(self):
        child = comment(parent_comment_id="t1_missing", original_parent_id="t1_missing")
        self.add([child])
        self.assertEqual(self.stage.get_comment("t1_missing"), {
            "comment_id": "t1_missing", "post_id": "t3_example",
            "parent_comment_id": None, "original_parent_id": None,
            "author": None, "created_utc": None, "score": None,
            "content_available": False, "source_id": None,
        })
        self.assertEqual(self.stage.get_comment("t1_child"), child)

    def test_actual_parent_enriches_placeholder_and_retains_its_own_parent(self):
        self.add([comment(parent_comment_id="t1_parent", original_parent_id="t1_parent")])
        parent = comment("t1_parent", parent_comment_id="t1_grandparent",
                         original_parent_id="t1_grandparent", author="parent_author")
        self.add([parent])
        self.assertEqual(self.stage.get_comment("t1_parent"), parent)
        self.assertIs(self.stage.get_comment("t1_grandparent")["content_available"], False)

    def test_parent_before_or_after_child_produces_identical_results(self):
        parent = comment("t1_parent", author="writer")
        child = comment(parent_comment_id="t1_parent", original_parent_id="t1_parent")
        results = []
        for index, rows in enumerate(([parent, child], [child, parent])):
            with CommentStaging(self.root / f"parents-{index}.sqlite") as stage:
                self.add(rows, stage=stage)
                results.append(self.rows(stage))
        self.assertEqual(results[0], results[1])
        self.assertTrue(all(row["content_available"] for row in results[0]))

    def test_top_level_parent_is_not_fabricated_for_unknown_parent(self):
        unknown = comment(original_parent_id=None, source_id=5)
        root = comment(author="root_author")
        self.add([unknown], task="pair_task")
        self.add([root])
        self.assertEqual(self.stage.get_comment("t1_child"), root)

    def test_cross_post_comment_and_parent_ownership_are_rejected(self):
        self.add([comment()])
        for row in (
            comment(post_id="t3_other"),
            comment("t1_other", post_id="t3_other",
                    parent_comment_id="t1_child", original_parent_id="t1_child"),
        ):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "belongs to another post"):
                    self.add([row])
        self.assertIsNone(self.stage.get_comment("t1_other"))

    def test_placeholder_cannot_later_belong_to_another_post(self):
        self.add([comment(parent_comment_id="t1_parent", original_parent_id="t1_parent")])
        with self.assertRaisesRegex(ValueError, "belongs to another post"):
            self.add([comment("t1_parent", post_id="t3_other")])
        self.assertIs(self.stage.get_comment("t1_parent")["content_available"], False)

    def test_contradictory_original_splits_for_one_post_are_rejected(self):
        self.add([comment()])
        with self.assertRaisesRegex(ValueError, "Conflicting original split"):
            self.add([comment("t1_other")], split="heldout")

    def test_conflicting_parent_relationships_are_rejected(self):
        self.add([comment()])
        changed = comment(parent_comment_id="t1_parent", original_parent_id="t1_parent", source_id=5)
        with self.assertRaisesRegex(ValueError, "Conflicting parent"):
            self.add([changed], task="pair_task")
        self.assertIsNone(self.stage.get_comment("t1_parent"))

    def test_failed_batch_rolls_back_new_rows_and_placeholders(self):
        self.add([comment()])
        new = comment("t1_new", parent_comment_id="t1_missing", original_parent_id="t1_missing")
        conflict = comment(post_id="t3_other")
        with self.assertRaises(ValueError):
            self.add([new, conflict])
        self.assertEqual(self.rows(), [comment()])

    def test_generator_input_and_export_batches_are_bounded_and_split_filtered(self):
        self.add((comment(f"t1_train_{index}") for index in range(5)))
        self.add([comment("t1_heldout", post_id="t3_heldout", source_id=1)], split="heldout")
        batches = list(self.stage.iter_batches(batch_size=2, split="train"))
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual([r["comment_id"] for batch in batches for r in batch],
                         [f"t1_train_{index}" for index in range(5)])
        self.assertEqual(len(list(self.stage.iter_batches(split="heldout"))[0]), 1)

    def test_invalid_inputs_are_rejected_without_changes(self):
        for task, split in (("op_task", "train"), ("all", "test")):
            with self.subTest(task=task, split=split):
                with self.assertRaises(ValueError):
                    self.add([comment()], task=task, split=split)
        for row in (
            {"comment_id": "t1_incomplete"},
            comment(content_available=False),
            comment(parent_comment_id="t1_child", original_parent_id="t1_child"),
            comment(parent_comment_id="t1_parent", original_parent_id="t1_different"),
            comment(original_parent_id="t3_other"),
        ):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    self.add([row])
        self.assertEqual(self.rows(), [])

    def test_invalid_export_options_are_rejected(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    list(self.stage.iter_batches(batch_size=size))
        with self.assertRaises(ValueError):
            list(self.stage.iter_batches(split="test"))

    def test_rows_remain_unchanged_and_unknown_comment_returns_none(self):
        row = comment(author="writer")
        original = copy.deepcopy(row)
        self.add([row])
        self.assertEqual(row, original)
        self.assertIsNone(self.stage.get_comment("t1_absent"))

    def test_existing_file_is_not_overwritten(self):
        path = self.root / "existing.sqlite"
        path.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            CommentStaging(path)
        self.assertEqual(path.read_bytes(), b"keep me")

    def test_context_manager_closes_but_retains_staging_file(self):
        path = self.root / "retained.sqlite"
        with CommentStaging(path) as stage:
            self.add([comment()], stage=stage)
        self.assertTrue(path.exists())
        with self.assertRaises(sqlite3.ProgrammingError):
            stage.get_comment("t1_child")
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM candidates").fetchone()[0], 1)

    def test_reconciled_rows_round_trip_with_comment_schema(self):
        self.add([comment(parent_comment_id="t1_missing", original_parent_id="t1_missing",
                          source_id=2**40, author="Café 🌱", score=0)])
        expected = self.rows()
        path = self.root / "comments.parquet"
        pq.write_table(pa.Table.from_pylist(expected, schema=COMMENT_SCHEMA), path, compression="zstd")
        restored = pq.read_table(path)
        self.assertEqual(restored.schema, COMMENT_SCHEMA)
        self.assertEqual(restored.to_pylist(), expected)


if __name__ == "__main__":
    unittest.main()
