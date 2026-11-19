"""Offline dataset checks and opt-in, rolled-back PostgreSQL COPY integration."""

import bz2
import copy
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import psycopg
from psycopg import sql


PROJECT_DIR = Path(__file__).resolve().parents[1]
from cmv_data.MetadataRunner import MetadataRunner
from cmv_data.ParquetDatabaseLoader import LOADERS, ParquetDatabaseLoader
from cmv_data.db_loaders.comments import CommentsLoader
from cmv_data.db_loaders.dataset import MetadataDataset
from cmv_data.db_loaders.op_labels import OpLabelsLoader
from cmv_data.db_loaders.posts import PostsLoader
from cmv_data.db_loaders.sources import SourcesLoader
from cmv_data.db_loaders.thread_comments import ThreadCommentsLoader
from cmv_data.parquet_loaders.metadata_staging import TABLE_SCHEMAS
from test_metadata_runner import SOURCE_MAP, sample_records


def make_release(root):
    archive_path = root / "sample.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        for member, record in sample_records().items():
            data = bz2.compress(b"\n" + json.dumps(record).encode() + b"\n")
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    output = root / "v1"
    MetadataRunner(archive_path, SOURCE_MAP, root / "sample.sqlite", output_path=output,
                   batch_size=1, rows_per_file=1).run()
    return output


class MetadataDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.release = make_release(cls.root)

    def reader(self):
        reader = MetadataDataset(self.release, batch_size=1)
        reader.validate()
        return reader

    def test_all_tables_emit_exact_database_columns_and_expected_counts(self):
        reader = self.reader()
        for table, schema in TABLE_SCHEMAS.items():
            with self.subTest(table=table):
                rows = list(reader.iter_rows(table))
                self.assertEqual(len(rows), reader.expected_counts()[table])
                self.assertTrue(all(set(row) == set(schema.names) for row in rows))

    def test_one_discussion_filters_shared_files_and_keeps_all_sources(self):
        reader = self.reader()
        for table in TABLE_SCHEMAS:
            rows = list(reader.iter_rows(table, post_id="t3_train"))
            self.assertEqual(len(rows), reader.expected_counts("t3_train")[table])
            if table != "sources":
                self.assertTrue(all(row["post_id"] == "t3_train" for row in rows))
        labels = list(reader.iter_rows("op_labels", post_id="t3_train"))
        self.assertIs(labels[0]["delta_label"], False)

    def test_requires_validation_and_rejects_unknown_posts_tables_and_sizes(self):
        reader = MetadataDataset(self.release)
        with self.assertRaises(RuntimeError):
            list(reader.iter_rows("posts"))
        reader.validate()
        with self.assertRaises(ValueError):
            list(reader.iter_rows("posts", post_id="t3_absent"))
        with self.assertRaises(ValueError):
            list(reader.iter_rows("unknown"))
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                MetadataDataset(self.release, batch_size=size)

    def test_corruption_fails_before_any_database_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copied"
            shutil.copytree(self.release, copied)
            with (copied / "sources.parquet").open("ab") as stream:
                stream.write(b"corruption")
            connection = mock.Mock()
            loader = ParquetDatabaseLoader(SimpleNamespace(connection=connection), copied)
            with self.assertRaises(ValueError):
                loader.load()
            connection.transaction.assert_not_called()


class PostgreSQLLoaderTests(unittest.TestCase):
    """Only private test schemas are written; every test transaction rolls back."""

    @classmethod
    def setUpClass(cls):
        dsn = os.environ.get("CMV_TEST_DATABASE_DSN")
        if not dsn:
            raise unittest.SkipTest("Set CMV_TEST_DATABASE_DSN to run PostgreSQL COPY integration tests.")
        cls.connection = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
        cls.addClassCleanup(cls.connection.close)
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        cls.release = make_release(cls.root)
        cls.reader = MetadataDataset(cls.release, batch_size=1)
        cls.reader.validate()
        cls.fixture = {name: list(cls.reader.iter_rows(name)) for name in TABLE_SCHEMAS}

    def setUp(self):
        self.transaction = self.connection.transaction(force_rollback=True)
        self.transaction.__enter__()
        self.addCleanup(self.transaction.__exit__, None, None, None)
        self.schema = "cmv_loader_test_" + uuid4().hex
        quoted = sql.Identifier(self.schema).as_string(self.connection)
        ddl = (PROJECT_DIR / "database/queries/create_schema.sql").read_text()
        ddl = ddl.replace("CREATE SCHEMA IF NOT EXISTS cmv;", f"CREATE SCHEMA {quoted};")
        ddl = ddl.replace("SET LOCAL search_path TO cmv;", f"SET LOCAL search_path TO {quoted};")
        self.connection.execute(ddl)

    def loader(self, cls):
        return cls(self.connection, schema=self.schema)

    def rows(self, table):
        return self.connection.execute(sql.SQL("SELECT {} FROM {} ORDER BY {}").format(
            sql.SQL(", ").join(map(sql.Identifier, TABLE_SCHEMAS[table].names)),
            sql.Identifier(self.schema, table), sql.Identifier(TABLE_SCHEMAS[table].names[0]),
        )).fetchall()

    def count(self, table):
        return self.connection.execute(sql.SQL("SELECT count(*) FROM {}").format(
            sql.Identifier(self.schema, table))).fetchone()[0]

    def parents(self):
        self.loader(SourcesLoader).load(self.fixture["sources"])
        self.loader(PostsLoader).load(self.fixture["posts"])

    def orchestrator(self):
        return ParquetDatabaseLoader(self.connection, self.release, schema=self.schema, batch_size=1)

    def test_each_table_loader_copies_and_reloads_idempotently(self):
        for cls in LOADERS:
            with self.subTest(table=cls.table):
                expected = len(self.fixture[cls.table])
                report = self.loader(cls).load(iter(self.fixture[cls.table]))
                self.assertEqual(report, {"processed": expected, "inserted": expected,
                                          "updated": 0, "unchanged": 0})
                report = self.loader(cls).load(iter(self.fixture[cls.table]))
                self.assertEqual(report["unchanged"], expected)
                self.assertEqual(report["inserted"], 0)
                self.assertEqual(self.count(cls.table), expected)

    def test_source_identity_sequence_advances_and_mapping_conflicts_rollback(self):
        source_loader = self.loader(SourcesLoader)
        source_loader.load(self.fixture["sources"])
        first = self.fixture["sources"][0]
        for row in ({**first, "source_id": 999}, {**first, "archive_member": "different"}):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    source_loader.load([row])
        self.assertEqual(self.count("sources"), 6)
        next_id = self.connection.execute(sql.SQL("""
            INSERT INTO {} (archive_member, task, original_split)
            VALUES ('custom_source', 'all', 'train') RETURNING source_id
        """).format(sql.Identifier(self.schema, "sources"))).fetchone()[0]
        self.assertEqual(next_id, 7)

    def test_child_before_parent_across_shards_loads_with_foreign_keys_enabled(self):
        self.parents()
        rows = list(self.reader.iter_rows("comments", post_id="t3_train"))
        rows.sort(key=lambda row: row["comment_id"] != "t1_train_child")
        self.assertEqual(rows[0]["parent_comment_id"], "t1_train_parent")
        self.loader(CommentsLoader).load(iter(rows))
        self.assertEqual(self.count("comments"), 4)

    def test_missing_parent_rejects_and_rolls_back_the_whole_comment_table_load(self):
        self.parents()
        row = {**self.fixture["comments"][0], "comment_id": "t1_bad",
               "parent_comment_id": "t1_absent", "original_parent_id": "t1_absent"}
        with self.assertRaises(psycopg.errors.ForeignKeyViolation):
            self.loader(CommentsLoader).load([row])
        self.assertEqual(self.count("comments"), 0)
        self.assertEqual(self.count("posts"), 2)

    def test_comment_placeholder_enrichment_and_no_downgrade_of_available_content(self):
        self.parents()
        actual = copy.deepcopy(self.fixture["comments"][0])
        actual.update(comment_id="t1_enrich", parent_comment_id=None,
                      original_parent_id=actual["post_id"], author="Café 🌱", score=0)
        placeholder = {name: None for name in actual}
        placeholder.update(comment_id=actual["comment_id"], post_id=actual["post_id"], content_available=False)
        loader = self.loader(CommentsLoader)
        self.assertEqual(loader.load([placeholder])["inserted"], 1)
        self.assertEqual(loader.load([actual])["updated"], 1)
        self.assertEqual(loader.load([placeholder])["unchanged"], 1)
        names = TABLE_SCHEMAS["comments"].names
        self.assertEqual(dict(zip(names, self.rows("comments")[0])), actual)

    def test_post_placeholder_enrichment_and_no_downgrade(self):
        self.loader(SourcesLoader).load(self.fixture["sources"])
        actual = copy.deepcopy(self.fixture["posts"][0])
        placeholder = {name: None for name in actual}
        placeholder.update(post_id=actual["post_id"], original_split=actual["original_split"], content_available=False)
        loader = self.loader(PostsLoader)
        self.assertEqual(loader.load([placeholder])["inserted"], 1)
        self.assertEqual(loader.load([actual])["updated"], 1)
        self.assertEqual(loader.load([placeholder])["unchanged"], 1)

    def test_conflicting_comment_ownership_and_parent_are_rejected(self):
        self.parents()
        row = next(row for row in self.fixture["comments"] if row["comment_id"] == "t1_train_root")
        self.loader(CommentsLoader).load([row])
        for changed in ({**row, "post_id": "t3_heldout", "original_parent_id": "t3_heldout"},
                        {**row, "parent_comment_id": "t1_other", "original_parent_id": "t1_other"}):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    self.loader(CommentsLoader).load([changed])
        self.assertEqual(self.count("comments"), 1)

    def test_false_labels_remain_false_and_conflicting_labels_are_rejected(self):
        self.parents()
        row = next(row for row in self.fixture["op_labels"] if row["post_id"] == "t3_train")
        self.assertIs(row["delta_label"], False)
        self.loader(OpLabelsLoader).load([row])
        with self.assertRaises(ValueError):
            self.loader(OpLabelsLoader).load([{**row, "delta_label": True}])
        self.assertIs(self.rows("op_labels")[0][1], False)

    def test_duplicate_input_keys_fail_before_inserting_target_rows(self):
        self.loader(SourcesLoader).load(self.fixture["sources"])
        row = self.fixture["posts"][0]
        with self.assertRaises(psycopg.errors.UniqueViolation):
            self.loader(PostsLoader).load([row, row])
        self.assertEqual(self.count("posts"), 0)

    def test_immutable_pair_thread_and_membership_conflicts_are_rejected(self):
        self.orchestrator().load()
        changes = {"pairs": {"source_line": 999}, "threads": {"author": "changed"},
                   "thread_comments": {"position": 999}}
        for cls in LOADERS:
            if cls.table in changes:
                with self.subTest(table=cls.table):
                    row = {**self.fixture[cls.table][0], **changes[cls.table]}
                    with self.assertRaises(ValueError):
                        self.loader(cls).load([row])

    def test_one_discussion_then_full_import_and_rerun(self):
        loader = self.orchestrator()
        check = loader.check(post_id="t3_train")
        self.assertEqual(check["status"], "ready")
        self.assertEqual(self.count("sources"), 0)
        first = loader.load(post_id="t3_train")
        self.assertEqual(first["tables"]["comments"]["processed"], 4)
        self.assertEqual(first["database_rows"]["posts"], 1)
        full = loader.load()
        repeated = loader.load()
        for name in TABLE_SCHEMAS:
            self.assertEqual(full["database_rows"][name], len(self.fixture[name]))
            self.assertEqual(repeated["tables"][name]["inserted"], 0)
            self.assertEqual(repeated["tables"][name]["updated"], 0)

    def test_late_failure_rolls_back_all_tables_and_source_sequence_restart(self):
        with mock.patch.object(ThreadCommentsLoader, "load", side_effect=ValueError("late failure")):
            with self.assertRaisesRegex(ValueError, "late failure"):
                self.orchestrator().load()
        self.assertTrue(all(self.count(name) == 0 for name in TABLE_SCHEMAS))
        first_id = self.connection.execute(sql.SQL("""
            INSERT INTO {} (archive_member, task, original_split)
            VALUES ('after_rollback', 'all', 'train') RETURNING source_id
        """).format(sql.Identifier(self.schema, "sources"))).fetchone()[0]
        self.assertEqual(first_id, 1)

    def test_extra_nullable_or_defaulted_columns_are_preserved(self):
        self.connection.execute(sql.SQL("ALTER TABLE {} ADD COLUMN body text").format(
            sql.Identifier(self.schema, "posts")))
        self.connection.execute(sql.SQL("ALTER TABLE {} ADD COLUMN extra_json jsonb NOT NULL DEFAULT '{{}}'").format(
            sql.Identifier(self.schema, "op_labels")))
        self.parents()
        self.connection.execute(sql.SQL("UPDATE {} SET body = 'existing text'").format(
            sql.Identifier(self.schema, "posts")))
        self.orchestrator().load()
        self.assertEqual(self.connection.execute(sql.SQL("SELECT DISTINCT body FROM {}").format(
            sql.Identifier(self.schema, "posts"))).fetchall(), [("existing text",)])

    def test_schema_mismatch_and_source_mapping_mismatch_fail_before_loading(self):
        self.loader(SourcesLoader).load(self.fixture["sources"])
        self.connection.execute(sql.SQL("UPDATE {} SET archive_member = 'wrong' WHERE source_id = 1").format(
            sql.Identifier(self.schema, "sources")))
        with self.assertRaisesRegex(ValueError, "source mapping differs"):
            self.orchestrator().load()
        self.assertEqual(self.count("posts"), 0)
        self.connection.execute(sql.SQL("ALTER TABLE {} ALTER COLUMN title TYPE integer USING NULL").format(
            sql.Identifier(self.schema, "posts")))
        with self.assertRaisesRegex(ValueError, "wrong type"):
            self.orchestrator().check()


if __name__ == "__main__":
    unittest.main()
