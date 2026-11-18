"""Single-pass runner integration tests using a tiny six-source archive."""

import bz2
import copy
import io
import json
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from ArchiveReader import ArchiveReader
from MetadataRunner import MetadataRunner
from parquet_loaders.metadata_staging import MetadataStaging


SOURCE_MAP = {
    "all/heldout_period_data.jsonlist.bz2": {"source_id": 1, "task": "all", "original_split": "heldout"},
    "all/train_period_data.jsonlist.bz2": {"source_id": 2, "task": "all", "original_split": "train"},
    "op_task/heldout_op_data.jsonlist.bz2": {"source_id": 3, "task": "op_task", "original_split": "heldout"},
    "op_task/train_op_data.jsonlist.bz2": {"source_id": 4, "task": "op_task", "original_split": "train"},
    "pair_task/heldout_pair_data.jsonlist.bz2": {"source_id": 6, "task": "pair_task", "original_split": "heldout"},
    "pair_task/train_pair_data.jsonlist.bz2": {"source_id": 5, "task": "pair_task", "original_split": "train"},
}


def sample_records():
    records = {}
    for member, source in SOURCE_MAP.items():
        split = source["original_split"]
        post_id = f"t3_{split}"
        root, parent = f"t1_{split}_root", f"t1_{split}_parent"
        if source["task"] == "all":
            record = {
                "name": post_id, "title": f"Full {split} title", "score": 0,
                "num_comments": 1980, "comments": [
                    {"name": root, "parent_id": post_id, "score": 0},
                    {"name": parent, "parent_id": post_id},
                ],
            }
        elif source["task"] == "op_task":
            record = {"name": post_id, "title": f"OP {split} title", "author": "OP author",
                      "delta_label": split == "heldout"}
        else:
            record = {
                "op_name": post_id, "op_title": "Pair title", "op_author": "Pair author",
                "positive": {"comments": [
                    {"name": root, "parent_id": post_id, "score": 7, "author": "Café 🌱"},
                    {"name": f"t1_{split}_child", "parent_id": parent},
                ]},
                "negative": {"comments": [{"name": f"t1_{split}_negative", "parent_id": post_id}]},
            }
        records[member] = record
    return records


class MetadataRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "cmv.tar.bz2"
        self.staging = self.root / "metadata.sqlite"

    def write_archive(self, records=None, *, order=None, overrides=None):
        records = sample_records() if records is None else records
        payloads = {member: b"\n" + json.dumps(record).encode() + b"\n"
                    for member, record in records.items()}
        payloads.update(overrides or {})
        with tarfile.open(self.archive, "w:bz2") as archive:
            for member in (payloads if order is None else order):
                data = bz2.compress(payloads[member])
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    def runner(self, **options):
        return MetadataRunner(self.archive, SOURCE_MAP, self.staging, **options)

    def query(self, sql):
        with closing(sqlite3.connect(self.staging)) as connection:
            return connection.execute(sql).fetchall()

    def test_single_archive_open_routes_all_sources_and_returns_validated_counts(self):
        self.write_archive()
        with mock.patch("ArchiveReader.tarfile.open", wraps=tarfile.open) as opened:
            report = self.runner(batch_size=1).run()
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(report["status"], "validated_staging")
        self.assertEqual(report["staging_path"], str(self.staging.resolve()))
        self.assertEqual(report["validation"], {
            "tables": {"sources": 6, "posts": 2, "comments": 8, "op_labels": 2,
                       "pairs": 2, "threads": 4, "thread_comments": 6},
            "placeholders": {"posts": 0, "comments": 0},
        })
        for member, source in SOURCE_MAP.items():
            expected_comments = {"all": 2, "op_task": 0, "pair_task": 3}[source["task"]]
            self.assertEqual(report["input_by_source"][member], {"records": 1, "comments": expected_comments})
        self.assertEqual(self.query("SELECT source_id FROM pairs ORDER BY source_id"), [(5,), (6,)])
        self.assertEqual(self.query("SELECT source_line FROM pairs"), [(2,), (2,)])
        self.assertEqual(self.query("SELECT delta_label FROM op_labels ORDER BY post_id"), [(1,), (0,)])

    def test_reversed_archive_order_produces_identical_reconciled_rows(self):
        self.write_archive()
        expected = self.runner().run()["validation"]
        first_posts = self.query("SELECT * FROM reconciled_posts ORDER BY post_id")
        first_comments = self.query("SELECT * FROM reconciled_comments ORDER BY comment_id")
        self.write_archive(order=list(reversed(SOURCE_MAP)))
        reversed_path = self.root / "reversed.sqlite"
        report = MetadataRunner(self.archive, SOURCE_MAP, reversed_path).run()
        self.assertEqual(report["validation"], expected)
        with closing(sqlite3.connect(reversed_path)) as connection:
            self.assertEqual(connection.execute("SELECT * FROM reconciled_posts ORDER BY post_id").fetchall(), first_posts)
            self.assertEqual(connection.execute("SELECT * FROM reconciled_comments ORDER BY comment_id").fetchall(), first_comments)

    def test_mapped_discussion_comment_batches_respect_configured_size(self):
        self.write_archive()
        original = MetadataStaging.add_comments
        sizes = []

        def observed(stage, rows, *, task, split):
            if task == "all":
                sizes.append(len(rows))
            return original(stage, rows, task=task, split=split)

        with mock.patch.object(MetadataStaging, "add_comments", observed):
            self.runner(batch_size=1).run()
        self.assertEqual(sizes, [1, 1, 1, 1])

    def test_accepts_snapshot_json_path_and_copies_dictionary_input(self):
        self.write_archive()
        snapshot_path = self.root / "source_map.json"
        snapshot_path.write_text(json.dumps(SOURCE_MAP), encoding="utf-8")
        runner = MetadataRunner(self.archive, snapshot_path, self.staging)
        self.assertEqual(runner.run()["validation"]["tables"]["sources"], 6)
        snapshot = copy.deepcopy(SOURCE_MAP)
        runner = MetadataRunner(self.archive, snapshot, self.root / "copied.sqlite")
        snapshot["all/train_period_data.jsonlist.bz2"]["source_id"] = 999
        self.assertEqual(runner.run()["validation"]["tables"]["posts"], 2)

    def test_bad_comment_rolls_back_its_post_and_prior_batches_but_keeps_earlier_records(self):
        records = sample_records()
        member = "all/heldout_period_data.jsonlist.bz2"
        records[member]["comments"][1]["parent_id"] = "t3_other"
        order = ["all/train_period_data.jsonlist.bz2", member]
        self.write_archive(records, order=order)
        with self.assertRaises(ValueError) as error:
            self.runner(batch_size=1).run()
        self.assertIn(f"{member}, line 2", str(error.exception))
        self.assertIn("Comment position 1", str(error.exception))
        self.assertEqual(self.query("SELECT post_id FROM post_splits"), [("t3_train",)])
        self.assertEqual(self.query("SELECT count(*) FROM candidates"), [(2,)])

    def test_bad_op_label_does_not_leave_task_post_candidate(self):
        records = sample_records()
        member = "op_task/heldout_op_data.jsonlist.bz2"
        records[member]["delta_label"] = "false"
        self.write_archive(records)
        with self.assertRaisesRegex(ValueError, "boolean delta_label"):
            self.runner().run()
        self.assertEqual(self.query("SELECT count(*) FROM post_candidates WHERE task = 'op_task'"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM op_labels"), [(0,)])

    def test_bad_pair_comment_rolls_back_pair_post_candidate_and_relationships(self):
        records = sample_records()
        member = "pair_task/heldout_pair_data.jsonlist.bz2"
        records[member]["positive"]["comments"][1]["score"] = "invalid"
        self.write_archive(records)
        with self.assertRaises(ValueError) as error:
            self.runner().run()
        self.assertIn(f"{member}, line 2", str(error.exception))
        self.assertEqual(self.query("SELECT count(*) FROM pairs"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM threads"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM post_candidates WHERE task = 'pair_task'"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM candidates WHERE task = 'pair_task'"), [(0,)])

    def test_missing_source_refuses_success_even_when_staged_relationships_are_valid(self):
        records = sample_records()
        missing = "op_task/train_op_data.jsonlist.bz2"
        del records[missing]
        self.write_archive(records)
        with self.assertRaisesRegex(ValueError, f"Missing archive sources: {missing}"):
            self.runner().run()
        self.assertTrue(self.staging.exists())

    def test_empty_source_is_seen_and_has_zero_records(self):
        member = "op_task/train_op_data.jsonlist.bz2"
        self.write_archive(overrides={member: b""})
        report = self.runner().run()
        self.assertEqual(report["input_by_source"][member], {"records": 0, "comments": 0})
        self.assertEqual(report["validation"]["tables"]["op_labels"], 1)

    def test_duplicate_archive_member_is_rejected(self):
        self.write_archive(order=[*SOURCE_MAP, "all/train_period_data.jsonlist.bz2"])
        with self.assertRaisesRegex(ValueError, "Duplicate archive member"):
            self.runner().run()

    def test_malformed_json_has_source_and_physical_line_context(self):
        member = "all/heldout_period_data.jsonlist.bz2"
        self.write_archive(overrides={member: b"\n{broken json}\n"})
        with self.assertRaises(ValueError) as error:
            self.runner().run()
        self.assertIn(f"{member}, line 2", str(error.exception))

    def test_validation_failure_prevents_success_and_retains_staging_for_inspection(self):
        records = sample_records()
        discussion = records["all/train_period_data.jsonlist.bz2"]
        discussion["comments"][0]["parent_id"] = "t1_train_parent"
        discussion["comments"][1]["parent_id"] = "t1_train_root"
        records["pair_task/train_pair_data.jsonlist.bz2"]["positive"]["comments"][0]["parent_id"] = "t1_train_parent"
        self.write_archive(records)
        with self.assertRaisesRegex(ValueError, "parent cycle"):
            self.runner().run()
        self.assertEqual(self.query("SELECT count(*) FROM pairs"), [(2,)])

    def test_reader_iterator_closes_on_record_failure(self):
        records = sample_records()
        records["all/heldout_period_data.jsonlist.bz2"]["comments"] = None
        self.write_archive(records)
        original = ArchiveReader.iter_records
        closed = []

        def tracked(reader, task=None):
            try:
                yield from original(reader, task=task)
            finally:
                closed.append(True)

        with mock.patch.object(ArchiveReader, "iter_records", tracked):
            with self.assertRaisesRegex(ValueError, "flat comments list"):
                self.runner().run()
        self.assertEqual(closed, [True])

    def test_missing_archive_existing_staging_and_invalid_arguments_are_safe(self):
        with self.assertRaises(FileNotFoundError):
            self.runner().run()
        self.assertFalse(self.staging.exists())
        self.write_archive()
        self.staging.write_bytes(b"keep existing file")
        with self.assertRaises(FileExistsError):
            self.runner().run()
        self.assertEqual(self.staging.read_bytes(), b"keep existing file")
        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    self.runner(batch_size=batch_size)
        with self.assertRaises(ValueError):
            MetadataRunner(self.archive, {}, self.root / "invalid.sqlite")

    def test_cli_runs_the_same_pipeline_on_tiny_archive(self):
        self.write_archive()
        snapshot = self.root / "source_map.json"
        snapshot.write_text(json.dumps(SOURCE_MAP), encoding="utf-8")
        result = subprocess.run([
            sys.executable, str(PROJECT_DIR / "database" / "MetadataRunner.py"),
            "--archive", str(self.archive), "--source-map", str(snapshot),
            "--staging", str(self.staging), "--batch-size", "1",
        ], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "validated_staging")
        self.assertEqual(report["validation"]["tables"]["comments"], 8)


if __name__ == "__main__":
    unittest.main()
