"""Pair-task comment candidates and their integration with reconciliation."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from parquet_loaders.comments import COMMENT_SCHEMA, map_comment, map_pair_comment
from parquet_loaders.pairs import THREAD_COMMENT_SCHEMA, map_pair
from parquet_loaders.staging import CommentStaging


POST_ID = "t3_example"
MEMBER = "pair_task/train_pair_data.jsonlist.bz2"
SOURCE = {"source_id": 5, "task": "pair_task", "original_split": "train"}
ALL_SOURCE = {"source_id": 2, "task": "all", "original_split": "train"}


def raw_comment():
    return {
        "name": "t1_root", "parent_id": POST_ID, "link_id": POST_ID,
        "author": "Café 🌱", "created_utc": 1441131871.5, "score": 7,
        "body": "Argument text", "body_html": "<p>Argument text</p>",
        "replies": {"data": {"children": ["child"]}},
    }


def pair_record():
    return {
        "op_name": POST_ID,
        "positive": {"comments": [
            raw_comment(),
            {"id": "child", "parent_id": "t1_missing", "link_id": POST_ID},
        ]},
        "negative": {"comments": [
            raw_comment(),
            {"name": "t1_negative", "parent_id": POST_ID},
        ]},
    }


class PairCommentMappingTests(unittest.TestCase):
    def test_maps_same_columns_as_discussion_comment_with_pair_provenance(self):
        row = map_pair_comment(raw_comment(), POST_ID, SOURCE)
        expected = {**map_comment(raw_comment(), POST_ID, ALL_SOURCE), "source_id": 5}
        self.assertEqual(row, expected)
        self.assertEqual(set(row), set(COMMENT_SCHEMA.names))
        self.assertIs(row["content_available"], True)

    def test_preserves_nested_parent_even_when_record_is_unavailable(self):
        row = map_pair_comment({**raw_comment(), "parent_id": "t1_missing"}, POST_ID, SOURCE)
        self.assertEqual(row["parent_comment_id"], "t1_missing")
        self.assertEqual(row["original_parent_id"], "t1_missing")

    def test_missing_parent_is_unknown_and_not_fabricated_as_top_level(self):
        record = raw_comment()
        del record["parent_id"]
        for candidate in (record, {**record, "parent_id": None}):
            with self.subTest(candidate=candidate):
                row = map_pair_comment(candidate, POST_ID, SOURCE)
                self.assertIsNone(row["parent_comment_id"])
                self.assertIsNone(row["original_parent_id"])
                self.assertIs(row["content_available"], True)
        # The existing full-discussion mapper keeps its stricter contract.
        with self.assertRaisesRegex(ValueError, "invalid parent"):
            map_comment(record, POST_ID, ALL_SOURCE)

    def test_wrong_task_is_rejected(self):
        for task in ("all", "op_task"):
            with self.subTest(task=task):
                with self.assertRaisesRegex(ValueError, "expects 'pair_task' source"):
                    map_pair_comment(raw_comment(), POST_ID, {**SOURCE, "task": task})

    def test_invalid_or_cross_post_parent_is_rejected(self):
        for parent in ("", 42, "t1_", "wrong_prefix", "t3_other", "t1_root"):
            with self.subTest(parent=parent):
                with self.assertRaises(ValueError):
                    map_pair_comment({**raw_comment(), "parent_id": parent}, POST_ID, SOURCE)

    def test_pair_outputs_candidates_matching_every_membership(self):
        rows = map_pair(pair_record(), SOURCE, MEMBER, 2)
        self.assertEqual(
            [row["comment_id"] for row in rows["comments"]],
            [row["comment_id"] for row in rows["thread_comments"]],
        )
        self.assertEqual(len(rows["comments"]), 4)
        self.assertEqual(rows["comments"][1]["comment_id"], "t1_child")
        self.assertIsNone(rows["comments"][1]["author"])
        self.assertIsNone(rows["comments"][1]["score"])
        for row in rows["comments"]:
            self.assertEqual(row["source_id"], 5)
            self.assertEqual(row["post_id"], POST_ID)

    def test_invalid_comment_error_has_member_line_outcome_and_position(self):
        record = pair_record()
        record["positive"]["comments"][1]["parent_id"] = "t3_other"
        with self.assertRaises(ValueError) as error:
            map_pair(record, SOURCE, MEMBER, 17)
        message = str(error.exception)
        self.assertIn(f"{MEMBER}, line 17:", message)
        self.assertIn("positive, position 1", message)
        self.assertIn("invalid parent", message)

    def test_does_not_mutate_pair_or_include_text_and_replies(self):
        record = pair_record()
        source = copy.deepcopy(SOURCE)
        before = copy.deepcopy((record, source))
        rows = map_pair(record, source, MEMBER, 2)
        self.assertEqual((record, source), before)
        for row in rows["comments"]:
            self.assertTrue({"body", "body_html", "replies"}.isdisjoint(row))


class PairCommentStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pair = map_pair(pair_record(), SOURCE, MEMBER, 2)

    def test_shared_comment_is_canonical_once_but_keeps_both_memberships(self):
        with CommentStaging(self.root / "shared.sqlite") as stage:
            stage.add_comments(self.pair["comments"], task="pair_task", split="train")
            actual = [r for batch in stage.iter_batches() for r in batch if r["content_available"]]
        self.assertEqual(len(actual), 3)
        shared_memberships = [r for r in self.pair["thread_comments"] if r["comment_id"] == "t1_root"]
        self.assertEqual(len(shared_memberships), 2)
        self.assertNotEqual(shared_memberships[0]["thread_id"], shared_memberships[1]["thread_id"])

    def test_discussion_metadata_wins_in_both_source_orders(self):
        full = map_comment({**raw_comment(), "score": 0, "author": None}, POST_ID, ALL_SOURCE)
        results = []
        for index, batches in enumerate((
            ((self.pair["comments"], "pair_task"), ([full], "all")),
            (([full], "all"), (self.pair["comments"], "pair_task")),
        )):
            with CommentStaging(self.root / f"order-{index}.sqlite") as stage:
                for rows, task in batches:
                    stage.add_comments(rows, task=task, split="train")
                root = stage.get_comment("t1_root")
                self.assertEqual(root["score"], 0)
                self.assertEqual(root["source_id"], 2)
                self.assertEqual(root["author"], "Café 🌱")
                results.append([r for batch in stage.iter_batches() for r in batch])
        self.assertEqual(results[0], results[1])

    def test_full_discussion_can_enrich_parent_placeholder_from_pair(self):
        with CommentStaging(self.root / "parent.sqlite") as stage:
            stage.add_comments(self.pair["comments"], task="pair_task", split="train")
            self.assertIs(stage.get_comment("t1_missing")["content_available"], False)
            parent = map_comment({"name": "t1_missing", "parent_id": POST_ID,
                                  "author": "parent_author"}, POST_ID, ALL_SOURCE)
            stage.add_comments([parent], task="all", split="train")
            self.assertEqual(stage.get_comment("t1_missing"), parent)
            self.assertEqual(stage.get_comment("t1_child")["parent_comment_id"], "t1_missing")

    def test_pair_only_candidates_and_memberships_round_trip_with_matching_keys(self):
        with CommentStaging(self.root / "parquet.sqlite") as stage:
            stage.add_comments(self.pair["comments"], task="pair_task", split="train")
            comments = [r for batch in stage.iter_batches(batch_size=2) for r in batch]
        path = self.root / "comments.parquet"
        memberships_path = self.root / "thread_comments.parquet"
        pq.write_table(pa.Table.from_pylist(comments, schema=COMMENT_SCHEMA), path)
        pq.write_table(pa.Table.from_pylist(self.pair["thread_comments"], schema=THREAD_COMMENT_SCHEMA),
                       memberships_path)
        restored = pq.read_table(path)
        memberships = pq.read_table(memberships_path).to_pylist()
        self.assertEqual(restored.schema, COMMENT_SCHEMA)
        self.assertEqual(restored.to_pylist(), comments)
        keys = {(row["comment_id"], row["post_id"]) for row in restored.to_pylist()}
        for row in memberships:
            self.assertIn((row["comment_id"], row["post_id"]), keys)
        for row in comments:
            if row["parent_comment_id"] is not None:
                self.assertIn((row["parent_comment_id"], row["post_id"]), keys)


if __name__ == "__main__":
    unittest.main()
