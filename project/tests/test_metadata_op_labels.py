"""Offline OP-label mapping and Parquet round-trip checks."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from parquet_loaders.op_labels import OP_LABEL_SCHEMA, map_op_label


SOURCES = (
    {"source_id": 3, "task": "op_task", "original_split": "heldout"},
    {"source_id": 4, "task": "op_task", "original_split": "train"},
)


class OpLabelMappingTests(unittest.TestCase):
    def test_preserves_true_and_false_labels_and_source_ids(self):
        for source in SOURCES:
            for label in (True, False):
                with self.subTest(split=source["original_split"], label=label):
                    record = {"name": "t3_example", "delta_label": label}
                    row = map_op_label(record, source)
                    self.assertEqual(row, {
                        "post_id": "t3_example",
                        "delta_label": label,
                        "source_id": source["source_id"],
                    })
                    self.assertIs(row["delta_label"], label)

    def test_missing_label_is_not_treated_as_false(self):
        with self.assertRaisesRegex(ValueError, "boolean delta_label"):
            map_op_label({"name": "t3_example"}, SOURCES[0])

    def test_rejects_non_boolean_labels(self):
        for label in (None, 0, 1, 0.0, 1.0, "true", "false", "", [], {}):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "boolean delta_label"):
                    map_op_label({"name": "t3_example", "delta_label": label}, SOURCES[0])

    def test_rejects_missing_or_invalid_post_ids(self):
        for post_id in (None, 42, True, "", "t3_", "t1_comment", "wrong_prefix"):
            with self.subTest(post_id=post_id):
                with self.assertRaisesRegex(ValueError, "Invalid post ID"):
                    map_op_label({"name": post_id, "delta_label": True}, SOURCES[0])
        with self.assertRaisesRegex(ValueError, "Invalid post ID"):
            map_op_label({"delta_label": True}, SOURCES[0])

    def test_rejects_other_tasks(self):
        for task in ("all", "pair_task"):
            with self.subTest(task=task):
                with self.assertRaisesRegex(ValueError, "expects 'op_task' source"):
                    map_op_label(
                        {"name": "t3_example", "delta_label": True},
                        {**SOURCES[0], "task": task},
                    )

    def test_does_not_modify_inputs_or_include_text(self):
        record = {
            "name": "t3_example", "delta_label": False,
            "title": "Original title", "selftext": "Original body",
        }
        source = copy.deepcopy(SOURCES[0])
        original = copy.deepcopy((record, source))
        row = map_op_label(record, source)
        self.assertEqual((record, source), original)
        self.assertEqual(set(row), {"post_id", "delta_label", "source_id"})


class OpLabelParquetTests(unittest.TestCase):
    def test_round_trip_preserves_false_labels_and_bigint_source_ids(self):
        source = {**SOURCES[0], "source_id": 2**40}
        rows = [
            map_op_label({"name": "t3_positive", "delta_label": True}, source),
            map_op_label({"name": "t3_negative", "delta_label": False}, SOURCES[1]),
        ]
        table = pa.Table.from_pylist(rows, schema=OP_LABEL_SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op_labels.parquet"
            pq.write_table(table, path, compression="zstd")
            restored = pq.read_table(path)
            negative = pq.read_table(path, filters=[("delta_label", "=", False)])
            missing = pq.read_table(path, filters=[("post_id", "=", "t3_unlabelled")])

        self.assertEqual(restored.schema, OP_LABEL_SCHEMA)
        self.assertEqual(restored.to_pylist(), rows)
        self.assertEqual(negative.to_pylist(), [rows[1]])
        # An unlabelled post has no label row, rather than an invented False.
        self.assertEqual(missing.num_rows, 0)

    def test_empty_dataset_keeps_explicit_schema(self):
        table = pa.Table.from_pylist([], schema=OP_LABEL_SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op_labels.parquet"
            pq.write_table(table, path)
            restored = pq.read_table(path)
        self.assertEqual(restored.schema, OP_LABEL_SCHEMA)
        self.assertEqual(restored.num_rows, 0)


if __name__ == "__main__":
    unittest.main()
