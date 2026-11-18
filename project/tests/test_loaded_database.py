"""Opt-in, read-only verification of the real loaded cmv database."""

import json
import os
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from psycopg import sql


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from SupabaseConnection import Database
from parquet_loaders.metadata_staging import TABLE_KEYS, TABLE_SCHEMAS


class LoadedDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CMV_VERIFY_LIVE_DB") != "1":
            raise unittest.SkipTest("Run this file directly to verify the real database read-only.")
        cls.root = Path(os.environ.get("CMV_METADATA_DATASET", PROJECT_DIR / "data/processed/cmv-metadata/v1"))
        cls.manifest = json.loads((cls.root / "manifest.json").read_text())
        cls.index = json.loads((cls.root / "discussion_index.json").read_text())
        cls.sample = "t3_37i8f6"
        cls.paired = next(post_id for post_id, location in cls.index.items()
                          if location["rows"]["pairs"] and location["rows"]["op_labels"]
                          and post_id != cls.sample)
        cls.db = Database()
        cls.addClassCleanup(cls.db.close)
        cls.transaction = cls.db.connection.transaction()
        cls.transaction.__enter__()
        cls.addClassCleanup(cls.transaction.__exit__, None, None, None)
        cls.db.connection.execute("SET TRANSACTION READ ONLY")

    def query(self, statement, params=None):
        return self.db.connection.execute(statement, params).fetchall()

    def expected(self, table, post_id=None):
        if table == "sources":
            paths = ["sources.parquet"]
        else:
            paths = self.index[post_id]["files"][table]
        result = []
        for relative in paths:
            with pq.ParquetFile(self.root / relative) as parquet:
                for batch in parquet.iter_batches(batch_size=10000, columns=TABLE_SCHEMAS[table].names):
                    if post_id is not None:
                        batch = batch.filter(pc.equal(batch.column("post_id"), post_id))
                    result.extend(batch.to_pylist())
        return result

    def assert_matches_parquet(self, table, post_id=None):
        names = TABLE_SCHEMAS[table].names
        statement = sql.SQL("SELECT {} FROM {}").format(
            sql.SQL(", ").join(map(sql.Identifier, names)), sql.Identifier("cmv", table))
        if post_id is not None:
            statement += sql.SQL(" WHERE post_id = %s")
        actual = [dict(zip(names, row)) for row in self.query(
            statement, (post_id,) if post_id is not None else None)]
        expected = self.expected(table, post_id)
        keys = TABLE_KEYS[table]
        key = lambda row: tuple(row[name] for name in keys)
        expected = {key(row): row for row in expected}
        self.assertEqual(len(actual), len(expected))
        for row in actual:
            with self.subTest(table=table, key=key(row)):
                self.assertIn(key(row), expected)
                self.assertEqual(row, expected[key(row)])

    def test_all_table_counts_match_published_manifest(self):
        for table, count in self.manifest["validation"]["tables"].items():
            with self.subTest(table=table):
                actual = self.query(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier("cmv", table)))
                self.assertEqual(actual[0][0], count)

    def test_exact_source_ids_and_original_splits_match_parquet(self):
        self.assert_matches_parquet("sources")

    def test_large_discussion_and_op_label_match_parquet_exactly(self):
        for table in ("posts", "comments", "op_labels"):
            self.assert_matches_parquet(table, self.sample)
        comments = self.query("SELECT count(*) FROM cmv.comments WHERE post_id = %s", (self.sample,))
        self.assertEqual(comments[0][0], 1367)
        label = self.query("SELECT delta_label FROM cmv.op_labels WHERE post_id = %s", (self.sample,))
        self.assertIs(label[0][0], True)

    def test_sample_reply_tree_has_real_same_post_parents(self):
        rows = self.query("""
            SELECT count(*) FILTER (WHERE c.parent_comment_id IS NOT NULL),
                   count(*) FILTER (WHERE c.parent_comment_id IS NOT NULL AND p.comment_id IS NULL),
                   count(*) FILTER (WHERE c.parent_comment_id = c.comment_id),
                   count(*) FILTER (WHERE c.content_available AND c.parent_comment_id IS NULL
                                    AND c.original_parent_id IS DISTINCT FROM c.post_id)
            FROM cmv.comments c LEFT JOIN cmv.comments p
              ON p.comment_id = c.parent_comment_id AND p.post_id = c.post_id
            WHERE c.post_id = %s
        """, (self.sample,))[0]
        self.assertGreater(rows[0], 0)
        self.assertEqual(rows[1:], (0, 0, 0))

    def test_pair_threads_and_ordered_memberships_match_parquet(self):
        for table in ("pairs", "threads", "thread_comments"):
            self.assert_matches_parquet(table, self.paired)
        outcomes = defaultdict(set)
        for pair_id, outcome in self.query(
                "SELECT pair_id, outcome FROM cmv.threads WHERE post_id = %s", (self.paired,)):
            outcomes[pair_id].add(outcome)
        self.assertTrue(outcomes)
        self.assertTrue(all(value == {"positive", "negative"} for value in outcomes.values()))
        for thread_id, positions in self.query("""
            SELECT thread_id, array_agg(position ORDER BY position)
            FROM cmv.thread_comments WHERE post_id = %s GROUP BY thread_id
        """, (self.paired,)):
            with self.subTest(thread_id=thread_id):
                self.assertEqual(positions, list(range(len(positions))))
        missing = self.query("""
            SELECT count(*) FROM cmv.thread_comments m
            LEFT JOIN cmv.comments c ON c.comment_id = m.comment_id AND c.post_id = m.post_id
            LEFT JOIN cmv.threads t ON t.thread_id = m.thread_id AND t.post_id = m.post_id
            WHERE m.post_id = %s AND (c.comment_id IS NULL OR t.thread_id IS NULL)
        """, (self.paired,))
        self.assertEqual(missing[0][0], 0)

    def test_missing_parent_placeholders_remain_unknown_and_referenced(self):
        rows = self.query("""
            SELECT c.comment_id, c.parent_comment_id, c.original_parent_id,
                   c.author, c.created_utc, c.score, c.source_id,
                   EXISTS (SELECT 1 FROM cmv.comments child
                           WHERE child.parent_comment_id = c.comment_id AND child.post_id = c.post_id)
            FROM cmv.comments c WHERE NOT c.content_available
        """)
        self.assertEqual(len(rows), self.manifest["validation"]["placeholders"]["comments"])
        for row in rows:
            with self.subTest(comment_id=row[0]):
                self.assertTrue(all(value is None for value in row[1:7]))
                self.assertIs(row[7], True)

    def test_false_op_label_is_preserved(self):
        selected = self.query("SELECT post_id FROM cmv.op_labels WHERE delta_label = FALSE ORDER BY post_id LIMIT 1")
        self.assertTrue(selected)
        self.assert_matches_parquet("op_labels", selected[0][0])
        self.assertIs(self.expected("op_labels", selected[0][0])[0]["delta_label"], False)


if __name__ == "__main__":
    os.environ["CMV_VERIFY_LIVE_DB"] = "1"
    unittest.main()
