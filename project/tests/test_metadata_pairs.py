"""Offline pair-task mapping, key stability, and Parquet relationship tests."""

import bz2
import copy
import io
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from ArchiveReader import ArchiveReader
from parquet_loaders.pairs import (
    PAIR_SCHEMA, THREAD_SCHEMA, THREAD_COMMENT_SCHEMA, map_pair,
)


MEMBER = "pair_task/train_pair_data.jsonlist.bz2"
SOURCE = {"source_id": 5, "task": "pair_task", "original_split": "train"}
POST_ID = "t3_example"


def example_pair():
    return {
        "op_name": POST_ID,
        "op_title": "Original title",
        "op_text": "Original body",
        "op_author": "original_author",
        "positive": {
            "ancestor": "t1_root",
            "author": "Café 🌱",
            "comments": [
                {"name": "t1_root", "link_id": POST_ID, "body": "First argument"},
                {"name": "t1_reply", "link_id": POST_ID, "body": "Follow-up"},
            ],
        },
        "negative": {
            "ancestor": "raw_ancestor_value",
            "author": "another_author",
            "comments": [{"id": "negative", "link_id": POST_ID, "body": "Argument"}],
        },
    }


class PairMappingTests(unittest.TestCase):
    def test_maps_pair_two_outcomes_and_ordered_memberships(self):
        rows = map_pair(example_pair(), SOURCE, MEMBER, 7)
        pair_id = f"{MEMBER}#7"
        self.assertEqual(rows["pair"], {
            "pair_id": pair_id, "post_id": POST_ID, "source_id": 5, "source_line": 7,
        })
        self.assertEqual(rows["threads"], [
            {"thread_id": f"{pair_id}:positive", "pair_id": pair_id,
             "post_id": POST_ID, "outcome": "positive",
             "ancestor_id": "t1_root", "author": "Café 🌱"},
            {"thread_id": f"{pair_id}:negative", "pair_id": pair_id,
             "post_id": POST_ID, "outcome": "negative",
             "ancestor_id": "raw_ancestor_value", "author": "another_author"},
        ])
        self.assertEqual(rows["thread_comments"], [
            {"thread_id": f"{pair_id}:positive", "comment_id": "t1_root",
             "post_id": POST_ID, "position": 0},
            {"thread_id": f"{pair_id}:positive", "comment_id": "t1_reply",
             "post_id": POST_ID, "position": 1},
            {"thread_id": f"{pair_id}:negative", "comment_id": "t1_negative",
             "post_id": POST_ID, "position": 0},
        ])

    def test_stable_ids_do_not_depend_on_database_source_id(self):
        first = map_pair(example_pair(), SOURCE, MEMBER, 7)
        rebuilt = map_pair(example_pair(), {**SOURCE, "source_id": 2**40}, MEMBER, 7)
        self.assertEqual(first["pair"]["pair_id"], rebuilt["pair"]["pair_id"])
        self.assertEqual(first["threads"], rebuilt["threads"])
        self.assertEqual(first["thread_comments"], rebuilt["thread_comments"])
        self.assertEqual(rebuilt["pair"]["source_id"], 2**40)

    def test_same_post_can_have_multiple_pairs_and_split_ids_do_not_collide(self):
        first = map_pair(example_pair(), SOURCE, MEMBER, 1)
        second = map_pair(example_pair(), SOURCE, MEMBER, 2)
        heldout_member = "pair_task/heldout_pair_data.jsonlist.bz2"
        heldout_source = {"source_id": 6, "task": "pair_task", "original_split": "heldout"}
        heldout = map_pair(example_pair(), heldout_source, heldout_member, 1)
        self.assertEqual(len({r["pair"]["pair_id"] for r in (first, second, heldout)}), 3)

    def test_missing_optional_thread_metadata_stays_null(self):
        record = {"op_name": POST_ID, "positive": {"comments": []}, "negative": {"comments": []}}
        rows = map_pair(record, SOURCE, MEMBER, 1)
        self.assertEqual(len(rows["threads"]), 2)
        self.assertEqual(rows["thread_comments"], [])
        for thread in rows["threads"]:
            self.assertIsNone(thread["ancestor_id"])
            self.assertIsNone(thread["author"])

    def test_duplicate_membership_in_one_thread_is_rejected(self):
        record = example_pair()
        record["positive"]["comments"].append({"id": "root"})
        with self.assertRaisesRegex(ValueError, "duplicate comment ID t1_root"):
            map_pair(record, SOURCE, MEMBER, 1)

    def test_shared_comment_can_belong_to_different_threads(self):
        record = example_pair()
        record["negative"]["comments"] = [{"name": "t1_root"}]
        rows = map_pair(record, SOURCE, MEMBER, 1)
        shared = [r for r in rows["thread_comments"] if r["comment_id"] == "t1_root"]
        self.assertEqual(len(shared), 2)
        self.assertNotEqual(shared[0]["thread_id"], shared[1]["thread_id"])

    def test_missing_or_malformed_outcomes_and_comment_lists_are_rejected(self):
        for value in (None, [], {}, {"comments": None}, {"comments": {}}):
            with self.subTest(value=value):
                record = {**example_pair(), "positive": value}
                with self.assertRaises(ValueError):
                    map_pair(record, SOURCE, MEMBER, 1)
        record = example_pair()
        del record["negative"]
        with self.assertRaisesRegex(ValueError, "negative thread"):
            map_pair(record, SOURCE, MEMBER, 1)

    def test_invalid_post_and_comment_ids_are_rejected(self):
        for post_id in (None, 42, "t1_comment", "t3_"):
            with self.subTest(post_id=post_id):
                with self.assertRaisesRegex(ValueError, "Invalid post ID"):
                    map_pair({**example_pair(), "op_name": post_id}, SOURCE, MEMBER, 1)
        for comment in ("t1_root", None, {}, {"id": 42}, {"name": 42}, {"name": "t1_"}, {"name": "t3_other"}):
            with self.subTest(comment=comment):
                record = example_pair()
                record["positive"]["comments"] = [comment]
                with self.assertRaises(ValueError):
                    map_pair(record, SOURCE, MEMBER, 1)

    def test_cross_post_ownership_error_includes_source_and_line(self):
        record = example_pair()
        record["positive"]["comments"][0]["link_id"] = "t3_other"
        with self.assertRaises(ValueError) as error:
            map_pair(record, SOURCE, MEMBER, 17)
        self.assertIn(f"{MEMBER}, line 17:", str(error.exception))
        self.assertIn("belongs to t3_other", str(error.exception))

    def test_invalid_source_task_split_member_and_id_are_rejected(self):
        for source in (
            {**SOURCE, "task": "all"}, {**SOURCE, "original_split": "test"},
            {**SOURCE, "original_split": "heldout"},
            *({**SOURCE, "source_id": value} for value in (None, True, 0, -1, 5.0, "5", 2**63)),
        ):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    map_pair(example_pair(), source, MEMBER, 1)
        with self.assertRaisesRegex(ValueError, "archive member"):
            map_pair(example_pair(), SOURCE, "wrong.jsonlist.bz2", 1)

    def test_source_line_must_fit_positive_postgres_integer(self):
        for line in (None, True, 0, -1, 1.0, "1", 2**31):
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, "source_line"):
                    map_pair(example_pair(), SOURCE, MEMBER, line)

    def test_does_not_modify_inputs_or_copy_text_into_relational_rows(self):
        record = example_pair()
        source = copy.deepcopy(SOURCE)
        before = copy.deepcopy((record, source))
        rows = map_pair(record, source, MEMBER, 1)
        self.assertEqual((record, source), before)
        self.assertEqual(set(rows["pair"]), set(PAIR_SCHEMA.names))
        for row in rows["threads"]:
            self.assertEqual(set(row), set(THREAD_SCHEMA.names))
        for row in rows["thread_comments"]:
            self.assertEqual(set(row), set(THREAD_COMMENT_SCHEMA.names))


class PairParquetTests(unittest.TestCase):
    def test_archive_to_three_parquet_tables_preserves_keys_and_positions(self):
        # The blank line ensures IDs use physical source lines, not row counts.
        payload = bz2.compress(b"\n" + json.dumps(example_pair()).encode() + b"\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "cmv.tar.bz2"
            with tarfile.open(archive_path, "w:bz2") as archive:
                member = tarfile.TarInfo(MEMBER)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            reader = ArchiveReader(archive_path, {MEMBER: SOURCE})
            with closing(reader.iter_records(task="pair_task")) as records:
                mapped = [map_pair(record, SOURCE, member, line) for member, line, record in records]
            self.assertEqual(len(mapped), 1)
            rows = mapped[0]
            self.assertEqual(rows["pair"]["source_line"], 2)
            self.assertEqual(rows["pair"]["pair_id"], f"{MEMBER}#2")

            restored = {}
            for name, schema, data in (
                ("pairs", PAIR_SCHEMA, [rows["pair"]]),
                ("threads", THREAD_SCHEMA, rows["threads"]),
                ("thread_comments", THREAD_COMMENT_SCHEMA, rows["thread_comments"]),
            ):
                path = root / f"{name}.parquet"
                pq.write_table(pa.Table.from_pylist(data, schema=schema), path, compression="zstd")
                table = pq.read_table(path)
                self.assertEqual(table.schema, schema)
                self.assertEqual(table.to_pylist(), data)
                restored[name] = table.to_pylist()

            pair_keys = {(r["pair_id"], r["post_id"]) for r in restored["pairs"]}
            thread_keys = {(r["thread_id"], r["post_id"]) for r in restored["threads"]}
            for thread in restored["threads"]:
                self.assertIn((thread["pair_id"], thread["post_id"]), pair_keys)
            for membership in restored["thread_comments"]:
                self.assertIn((membership["thread_id"], membership["post_id"]), thread_keys)

            selected = pq.read_table(root / "thread_comments.parquet", filters=[
                ("thread_id", "=", f"{MEMBER}#2:positive"),
            ]).to_pylist()
            self.assertEqual([r["comment_id"] for r in selected], ["t1_root", "t1_reply"])
            self.assertEqual([r["position"] for r in selected], [0, 1])

    def test_empty_memberships_keep_explicit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thread_comments.parquet"
            pq.write_table(pa.Table.from_pylist([], schema=THREAD_COMMENT_SCHEMA), path)
            table = pq.read_table(path)
        self.assertEqual(table.schema, THREAD_COMMENT_SCHEMA)
        self.assertEqual(table.num_rows, 0)


if __name__ == "__main__":
    unittest.main()
