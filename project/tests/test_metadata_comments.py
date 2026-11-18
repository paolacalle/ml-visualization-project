"""Offline comment mapping and Parquet relationship checks."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from parquet_loaders.comments import COMMENT_SCHEMA, map_comment


POST_ID = "t3_example"
SOURCE = {"source_id": 2, "task": "all", "original_split": "train"}


def example_comment():
    return {
        "name": "t1_root",
        "id": "root",
        "link_id": POST_ID,
        "parent_id": POST_ID,
        "author": "Café 🌱",
        "created_utc": 1441131871.5,
        "score": -2,
        "body": "Text stays in the text release.",
        "body_html": "<p>Text stays in the text release.</p>",
        "replies": {"data": {"children": []}},
    }


class CommentMappingTests(unittest.TestCase):
    def test_top_level_comment_preserves_metadata_and_original_parent(self):
        row = map_comment(example_comment(), POST_ID, SOURCE)
        self.assertEqual(row, {
            "comment_id": "t1_root",
            "post_id": POST_ID,
            "parent_comment_id": None,
            "original_parent_id": POST_ID,
            "author": "Café 🌱",
            "created_utc": 1441131871.5,
            "score": -2,
            "content_available": True,
            "source_id": 2,
        })

    def test_nested_comment_preserves_parent_reference(self):
        comment = {**example_comment(), "name": "t1_child", "parent_id": "t1_root"}
        row = map_comment(comment, POST_ID, SOURCE)
        self.assertEqual(row["parent_comment_id"], "t1_root")
        self.assertEqual(row["original_parent_id"], "t1_root")

    def test_missing_parent_record_keeps_reference_for_later_reconciliation(self):
        comment = {**example_comment(), "parent_id": "t1_unavailable"}
        row = map_comment(comment, POST_ID, SOURCE)
        self.assertEqual(row["parent_comment_id"], "t1_unavailable")
        self.assertEqual(row["original_parent_id"], "t1_unavailable")
        # The supplied comment is available even though its parent is absent.
        self.assertIs(row["content_available"], True)

    def test_missing_optional_metadata_is_null(self):
        row = map_comment({"name": "t1_minimal", "parent_id": POST_ID}, POST_ID, SOURCE)
        for field in ("author", "created_utc", "score"):
            with self.subTest(field=field):
                self.assertIsNone(row[field])

    def test_falls_back_to_bare_id_when_name_is_absent(self):
        row = map_comment({"id": "fallback", "parent_id": POST_ID}, POST_ID, SOURCE)
        self.assertEqual(row["comment_id"], "t1_fallback")

    def test_rejects_invalid_comment_ids(self):
        for comment in (
            {"name": "t3_post"}, {"name": "t1_"}, {"name": 42},
            {}, {"id": None}, {"id": ""}, {"id": 42},
        ):
            with self.subTest(comment=comment):
                with self.assertRaises(ValueError):
                    map_comment({**comment, "parent_id": POST_ID}, POST_ID, SOURCE)

    def test_rejects_invalid_post_ids(self):
        for post_id in (None, 42, "", "t1_comment"):
            with self.subTest(post_id=post_id):
                with self.assertRaisesRegex(ValueError, "Invalid post ID"):
                    map_comment(example_comment(), post_id, SOURCE)

    def test_rejects_wrong_task(self):
        source = {**SOURCE, "task": "pair_task"}
        with self.assertRaisesRegex(ValueError, "expects 'all' source"):
            map_comment(example_comment(), POST_ID, source)

    def test_rejects_cross_post_ownership(self):
        comment = {**example_comment(), "link_id": "t3_other"}
        with self.assertRaisesRegex(ValueError, "belongs to"):
            map_comment(comment, POST_ID, SOURCE)

    def test_rejects_invalid_or_cross_post_parent(self):
        for parent in (None, 42, "", "t1_", "wrong_prefix", "t3_other"):
            with self.subTest(parent=parent):
                with self.assertRaisesRegex(ValueError, "invalid parent"):
                    map_comment({**example_comment(), "parent_id": parent}, POST_ID, SOURCE)

    def test_rejects_self_reply(self):
        comment = {**example_comment(), "parent_id": "t1_root"}
        with self.assertRaisesRegex(ValueError, "cannot reply to itself"):
            map_comment(comment, POST_ID, SOURCE)

    def test_does_not_modify_inputs_or_include_text(self):
        comment = example_comment()
        source = copy.deepcopy(SOURCE)
        original = copy.deepcopy((comment, source))
        row = map_comment(comment, POST_ID, source)
        self.assertEqual((comment, source), original)
        self.assertTrue({"body", "body_html", "replies"}.isdisjoint(row))


class CommentParquetTests(unittest.TestCase):
    def test_round_trip_and_filtered_lookup_preserve_reply_relationships(self):
        comments = [
            example_comment(),
            {"name": "t1_child", "parent_id": "t1_root"},
            {"name": "t1_orphan", "parent_id": "t1_unavailable"},
        ]
        # Exercise BIGINT source IDs rather than relying on today's small IDs.
        source = {**SOURCE, "source_id": 2**40}
        rows = [map_comment(comment, POST_ID, source) for comment in comments]
        for row in rows:
            self.assertEqual(set(row), set(COMMENT_SCHEMA.names))
        table = pa.Table.from_pylist(rows, schema=COMMENT_SCHEMA)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comments.parquet"
            pq.write_table(table, path, compression="zstd")
            restored = pq.read_table(path)
            selected = pq.read_table(path, filters=[
                ("post_id", "=", POST_ID), ("comment_id", "=", "t1_child"),
            ])
            wrong_post = pq.read_table(path, filters=[("post_id", "=", "t3_other")])

        self.assertEqual(restored.schema, COMMENT_SCHEMA)
        self.assertEqual(restored.to_pylist(), rows)
        self.assertEqual(selected.to_pylist(), [rows[1]])
        self.assertEqual(wrong_post.num_rows, 0)
        self.assertIsNone(restored["parent_comment_id"][0].as_py())
        self.assertEqual(restored["parent_comment_id"][2].as_py(), "t1_unavailable")

    def test_score_outside_postgres_integer_range_is_rejected(self):
        for score in (-(2**31) - 1, 2**31):
            with self.subTest(score=score):
                row = map_comment({**example_comment(), "score": score}, POST_ID, SOURCE)
                with self.assertRaises(pa.ArrowInvalid):
                    pa.Table.from_pylist([row], schema=COMMENT_SCHEMA)


if __name__ == "__main__":
    unittest.main()
