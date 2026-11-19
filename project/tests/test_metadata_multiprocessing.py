"""Spawned-worker equivalence, bounded scheduling, and failure-prefix checks."""

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import closing
from unittest import mock

import test_metadata_runner as fixtures
from cmv_data.MetadataRunner import MetadataRunner
from cmv_data.parquet_loaders.metadata_staging import MetadataStaging


class MultiprocessMetadataRunnerTests(fixtures.MetadataRunnerTests):
    """Run the existing runner contract tests with two mapping processes."""

    def runner(self, **options):
        options.setdefault("workers", 2)
        return MetadataRunner(self.archive, fixtures.SOURCE_MAP, self.staging, **options)

    def test_parallel_and_sequential_tables_match_and_only_parent_writes(self):
        self.write_archive()
        sequential_path = self.root / "sequential.sqlite"
        sequential = MetadataRunner(self.archive, fixtures.SOURCE_MAP, sequential_path).run()
        writer_pids = []
        original = MetadataStaging.add_posts

        def observed(stage, *args, **kwargs):
            writer_pids.append(os.getpid())
            return original(stage, *args, **kwargs)

        with mock.patch("cmv_data.MetadataRunner.ProcessPoolExecutor", wraps=ProcessPoolExecutor) as pool:
            with mock.patch.object(MetadataStaging, "add_posts", observed):
                parallel = self.runner(batch_size=1).run()
        self.assertEqual(pool.call_args.kwargs["max_workers"], 2)
        self.assertEqual(pool.call_args.kwargs["mp_context"].get_start_method(), "spawn")
        self.assertTrue(writer_pids)
        self.assertEqual(set(writer_pids), {os.getpid()})
        self.assertEqual(parallel["validation"], sequential["validation"])
        self.assertEqual(parallel["input_by_source"], sequential["input_by_source"])
        with closing(sqlite3.connect(sequential_path)) as connection:
            for table, order in (
                ("sources", "source_id"), ("reconciled_posts", "post_id"),
                ("reconciled_comments", "comment_id"), ("op_labels", "post_id"),
                ("pairs", "pair_id"), ("threads", "thread_id"),
                ("thread_comments", "thread_id, position"),
            ):
                with self.subTest(table=table):
                    expected = connection.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
                    self.assertEqual(self.query(f"SELECT * FROM {table} ORDER BY {order}"), expected)

    def test_read_ahead_json_error_preserves_prior_successful_records(self):
        good = "all/train_period_data.jsonlist.bz2"
        bad = "all/heldout_period_data.jsonlist.bz2"
        self.write_archive(order=[good, bad], overrides={bad: b"\n{broken json}\n"})
        with self.assertRaises(ValueError) as error:
            self.runner().run()
        self.assertIn(f"{bad}, line 2", str(error.exception))
        self.assertEqual(self.query("SELECT post_id FROM post_splits"), [("t3_train",)])
        self.assertEqual(self.query("SELECT count(*) FROM candidates"), [(2,)])

    def test_submission_is_bounded_and_does_not_exhaust_the_input(self):
        class ImmediatePool:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, function, job):
                future = Future()
                future.set_result(function(job))
                return future

        member = "all/train_period_data.jsonlist.bz2"
        raw = fixtures.sample_records()[member]
        reads = []

        def records():
            for line in range(1, 11):
                reads.append(line)
                yield member, line, raw

        runner = self.runner()
        with mock.patch("cmv_data.MetadataRunner.ProcessPoolExecutor", ImmediatePool):
            with closing(runner._prepared_records(records(), fixtures.SOURCE_MAP)) as prepared:
                self.assertEqual(next(prepared)[1], 1)
                self.assertEqual(len(reads), 4)
                self.assertEqual(next(prepared)[1], 2)
                self.assertEqual(len(reads), 5)

    def test_cli_parallel_export_returns_clean_json(self):
        self.write_archive()
        snapshot = self.root / "source_map.json"
        snapshot.write_text(json.dumps(fixtures.SOURCE_MAP))
        result = subprocess.run([
            sys.executable, str(fixtures.PROJECT_DIR / "database" / "MetadataRunner.py"),
            "--archive", str(self.archive), "--source-map", str(snapshot),
            "--staging", str(self.staging), "--workers", "2",
            "--output", str(self.root / "v1"), "--rows-per-file", "2",
        ], text=True, capture_output=True, check=True, timeout=60)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "exported_metadata")
        self.assertEqual(report["validation"]["tables"]["comments"], 8)
        self.assertTrue((self.root / "v1" / "manifest.json").exists())

    def test_invalid_worker_counts_are_rejected(self):
        for workers in (0, -1, True, 1.5, "2"):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "workers"):
                    self.runner(workers=workers)


if __name__ == "__main__":
    import unittest
    unittest.main()
