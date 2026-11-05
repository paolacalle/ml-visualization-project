"""Offline post mapping and compressed-archive-to-Parquet integration checks."""

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
from parquet_loaders.posts import POST_FIELDS, POST_SCHEMA, map_post


TRAIN_MEMBER = "all/train_period_data.jsonlist.bz2"
HELDOUT_MEMBER = "all/heldout_period_data.jsonlist.bz2"
OP_MEMBER = "op_task/train_op_data.jsonlist.bz2"
SOURCE_MAP = {
    TRAIN_MEMBER: {"source_id": 2, "task": "all", "original_split": "train"},
    HELDOUT_MEMBER: {"source_id": 1, "task": "all", "original_split": "heldout"},
    OP_MEMBER: {"source_id": 4, "task": "op_task", "original_split": "train"},
}


def example_post():
    return {
        "name": "t3_example",
        "title": 'Café 🌱\nA "quoted" title',
        "author": "example_author",
        "created_utc": 1441131871.0,
        "score": 21,
        "ups": 22,
        "downs": 1,
        "num_comments": 18,
        "gilded": 0,
        "permalink": "/r/changemyview/comments/example/",
        "over_18": False,
        "archived": True,
        "stickied": False,
        "distinguished": None,
        "link_flair_text": "[Deltas Awarded]",
        "author_flair_text": "1∆",
        "edited": False,
        # These belong in the text release or comment dataset, not a post row.
        "selftext": "Post body",
        "selftext_html": "<p>Post body</p>",
        "comments": [{"name": "t1_example", "body": "A reply"}],
    }


class PostMappingTests(unittest.TestCase):
    def test_preserves_source_ids_and_original_splits(self):
        for source in (SOURCE_MAP[TRAIN_MEMBER], SOURCE_MAP[HELDOUT_MEMBER]):
            with self.subTest(split=source["original_split"]):
                row = map_post(example_post(), source)
                self.assertEqual(row["post_id"], "t3_example")
                self.assertEqual(row["source_id"], source["source_id"])
                self.assertEqual(row["original_split"], source["original_split"])
                self.assertIs(row["content_available"], True)
                self.assertEqual(set(row), set(POST_SCHEMA.names))

    def test_missing_optional_metadata_is_null(self):
        row = map_post({"name": "t3_minimal"}, SOURCE_MAP[TRAIN_MEMBER])
        for field in POST_FIELDS:
            with self.subTest(field=field):
                self.assertIsNone(row[field])
        self.assertIsNone(row["is_edited"])
        self.assertIsNone(row["edited_utc"])

    def test_boolean_and_timestamp_edit_values(self):
        for edited, flag, timestamp in (
            (False, False, None),
            (True, True, None),
            (1441132000, True, 1441132000),
            (1441132000.5, True, 1441132000.5),
            (None, None, None),
        ):
            with self.subTest(edited=edited):
                row = map_post({**example_post(), "edited": edited}, SOURCE_MAP[TRAIN_MEMBER])
                self.assertIs(row["is_edited"], flag)
                self.assertEqual(row["edited_utc"], timestamp)

    def test_rejects_wrong_task(self):
        with self.assertRaisesRegex(ValueError, "expects 'all' source"):
            map_post(example_post(), SOURCE_MAP[OP_MEMBER])

    def test_rejects_invalid_post_ids(self):
        for post_id in (None, 42, True, "", "t1_comment", "wrong_prefix"):
            with self.subTest(post_id=post_id):
                with self.assertRaisesRegex(ValueError, "Invalid post ID"):
                    map_post({"name": post_id}, SOURCE_MAP[TRAIN_MEMBER])

    def test_does_not_modify_inputs(self):
        record = example_post()
        source = copy.deepcopy(SOURCE_MAP[TRAIN_MEMBER])
        originals = copy.deepcopy((record, source))
        map_post(record, source)
        self.assertEqual((record, source), originals)

    def test_omits_body_html_and_nested_comments(self):
        row = map_post(example_post(), SOURCE_MAP[TRAIN_MEMBER])
        self.assertTrue({"body", "selftext", "selftext_html", "comments"}.isdisjoint(row))
        self.assertEqual(row["title"], example_post()["title"])


class PostParquetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "cmv.tar.bz2"

        first = {**example_post(), "name": "t3_train_a"}
        second = {"name": "t3_train_b", "edited": 1441132000.5}
        heldout = {**example_post(), "name": "t3_heldout", "edited": True}
        payloads = {
            # Include a different task to verify the reader's task filter.
            OP_MEMBER: [json.dumps({"name": "t3_train_a", "delta_label": True}).encode()],
            # Blank lines count toward provenance line numbers but are not rows.
            TRAIN_MEMBER: [b"", json.dumps(first).encode(), b"", json.dumps(second).encode()],
            HELDOUT_MEMBER: [json.dumps(heldout).encode()],
        }
        with tarfile.open(self.archive, "w:bz2") as archive:
            for name, lines in payloads.items():
                data = bz2.compress(b"\n".join(lines) + b"\n")
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def mapped_rows(self):
        reader = ArchiveReader(self.archive, SOURCE_MAP)
        with closing(reader.iter_records(task="all")) as records:
            return [map_post(record, SOURCE_MAP[member]) for member, _, record in records]

    def test_reader_filters_tasks_and_preserves_real_line_numbers(self):
        reader = ArchiveReader(self.archive, SOURCE_MAP)
        with closing(reader.iter_records(task="all")) as records:
            provenance = [(member, line, record["name"]) for member, line, record in records]
        self.assertEqual(provenance, [
            (TRAIN_MEMBER, 2, "t3_train_a"),
            (TRAIN_MEMBER, 4, "t3_train_b"),
            (HELDOUT_MEMBER, 1, "t3_heldout"),
        ])

    def test_three_post_archive_to_parquet_round_trip(self):
        rows = self.mapped_rows()
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(set(row), set(POST_SCHEMA.names))

        original = pa.Table.from_pylist(rows, schema=POST_SCHEMA)
        path = self.root / "posts.parquet"
        pq.write_table(original, path, compression="zstd")
        restored = pq.read_table(path)

        self.assertEqual(restored.schema, POST_SCHEMA)
        self.assertTrue(restored.equals(original))
        self.assertEqual(restored.to_pylist(), rows)
        self.assertEqual(restored.num_rows, 3)
        self.assertEqual(restored["title"][0].as_py(), 'Café 🌱\nA "quoted" title')
        self.assertIsNone(restored["archived"][1].as_py())
        self.assertIs(restored["over_18"][0].as_py(), False)
        self.assertIsNone(restored["edited_utc"][0].as_py())
        self.assertEqual(restored["edited_utc"][1].as_py(), 1441132000.5)

    def test_filtered_lookup_by_database_post_id(self):
        path = self.root / "posts.parquet"
        pq.write_table(pa.Table.from_pylist(self.mapped_rows(), schema=POST_SCHEMA), path)
        result = pq.read_table(path, filters=[("post_id", "=", "t3_heldout")])
        self.assertEqual(result.num_rows, 1)
        self.assertEqual(result["source_id"][0].as_py(), 1)
        self.assertEqual(result["original_split"][0].as_py(), "heldout")

    def test_bigint_source_id_survives_round_trip(self):
        source_id = 2**40
        source = {**SOURCE_MAP[TRAIN_MEMBER], "source_id": source_id}
        row = map_post(example_post(), source)
        path = self.root / "large-source-id.parquet"
        pq.write_table(pa.Table.from_pylist([row], schema=POST_SCHEMA), path)
        result = pq.read_table(path)
        self.assertEqual(result.schema.field("source_id").type, pa.int64())
        self.assertEqual(result["source_id"][0].as_py(), source_id)

    def test_integer_fields_reject_overflow_instead_of_truncating(self):
        row = map_post({**example_post(), "score": 2**31}, SOURCE_MAP[TRAIN_MEMBER])
        with self.assertRaises(pa.ArrowInvalid):
            pa.Table.from_pylist([row], schema=POST_SCHEMA)


if __name__ == "__main__":
    unittest.main()
