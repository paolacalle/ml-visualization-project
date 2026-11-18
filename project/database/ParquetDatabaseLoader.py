"""Import verified metadata Parquet through table-specific PostgreSQL loaders."""

import argparse
import json
from pathlib import Path

from psycopg import sql

from db_loaders._bulk import PG_TYPES
from db_loaders.comments import CommentsLoader
from db_loaders.dataset import MetadataDataset
from db_loaders.op_labels import OpLabelsLoader
from db_loaders.pairs import PairsLoader
from db_loaders.posts import PostsLoader
from db_loaders.sources import SourcesLoader
from db_loaders.thread_comments import ThreadCommentsLoader
from db_loaders.threads import ThreadsLoader
from parquet_loaders.metadata_staging import TABLE_KEYS, TABLE_SCHEMAS
from parquet_loaders.source_map import map_sources


LOADERS = (SourcesLoader, PostsLoader, CommentsLoader, OpLabelsLoader,
           PairsLoader, ThreadsLoader, ThreadCommentsLoader)


class ParquetDatabaseLoader:
    """Load one discussion or the whole release in a single transaction.

    No raw archive access, table creation, deletion, or source-ID remapping is
    performed. The caller supplies an existing Database wrapper or psycopg
    connection and owns closing it. If that connection already has an outer
    transaction, this load is a savepoint and the caller must commit it.
    """

    def __init__(self, db, dataset_path, *, schema="cmv", batch_size=10000):
        self.connection = getattr(db, "connection", db)
        self.schema = schema
        self.dataset = MetadataDataset(dataset_path, batch_size=batch_size)

    def _preflight(self, cursor):
        cursor.execute("""
            SELECT table_name, column_name, udt_name, is_nullable, column_default, is_identity
            FROM information_schema.columns WHERE table_schema = %s
        """, (self.schema,))
        definitions = {}
        for table, name, data_type, nullable, default, identity in cursor.fetchall():
            definitions.setdefault(table, {})[name] = (data_type, nullable, default, identity)
        for table, arrow_schema in TABLE_SCHEMAS.items():
            actual = definitions.get(table, {})
            for field in arrow_schema:
                if field.name not in actual or actual[field.name][0] != PG_TYPES[str(field.type)]:
                    raise ValueError(f"Database column {self.schema}.{table}.{field.name} is missing or has the wrong type.")
            for name, (_, nullable, default, identity) in actual.items():
                if name not in arrow_schema.names and nullable == "NO" and default is None and identity != "YES":
                    raise ValueError(f"Extra required database column needs a default: {table}.{name}")
            cursor.execute("""
                SELECT a.attname FROM pg_constraint c
                JOIN pg_class t ON t.oid = c.conrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, position)
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
                WHERE n.nspname = %s AND t.relname = %s AND c.contype = 'p'
                ORDER BY k.position
            """, (self.schema, table))
            if tuple(row[0] for row in cursor.fetchall()) != TABLE_KEYS[table]:
                raise ValueError(f"Unexpected primary key on {self.schema}.{table}")

        cursor.execute(sql.SQL("SELECT source_id, archive_member, task, original_split FROM {}").format(
            sql.Identifier(self.schema, "sources")))
        existing = cursor.fetchall()
        wanted = map_sources(self.dataset.manifest["source_map"])
        by_id = {row["source_id"]: row for row in wanted}
        by_member = {row["archive_member"]: row for row in wanted}
        for source_id, member, task, split in existing:
            row = {"source_id": source_id, "archive_member": member, "task": task, "original_split": split}
            if ((source_id in by_id and by_id[source_id] != row)
                    or (member in by_member and by_member[member] != row)):
                raise ValueError(f"Database source mapping differs for {member}; remap explicitly before loading.")
        return len(existing)

    def check(self, *, post_id=None):
        """Verify the release, selected discussion, schema, and existing sources."""
        self.dataset.validate()
        expected = self.dataset.expected_counts(post_id)
        with self.connection.transaction(), self.connection.cursor() as cursor:
            existing = self._preflight(cursor)
        return {"status": "ready", "expected_rows": expected, "existing_sources": existing}

    def load(self, *, post_id=None):
        self.dataset.validate()
        expected = self.dataset.expected_counts(post_id)
        results = {}
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                self._preflight(cursor)
            for loader_class in LOADERS:
                loader = loader_class(self.connection, schema=self.schema)
                results[loader.table] = loader.load(self.dataset.iter_rows(loader.table, post_id=post_id))
                if results[loader.table]["processed"] != expected[loader.table]:
                    raise ValueError(f"Unexpected imported count for {loader.table}")
            with self.connection.cursor() as cursor:
                cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
                counts = {}
                for name in TABLE_SCHEMAS:
                    cursor.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(self.schema, name)))
                    counts[name] = cursor.fetchone()[0]
        return {"status": "loaded", "dataset": str(self.dataset.path), "post_id": post_id,
                "tables": results, "database_rows": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--post-id", help="Import one complete discussion first.")
    parser.add_argument("--schema", default="cmv")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--check", action="store_true", help="Read-only release and database preflight.")
    args = parser.parse_args()
    from SupabaseConnection import Database
    db = Database()
    try:
        loader = ParquetDatabaseLoader(db, args.dataset, schema=args.schema, batch_size=args.batch_size)
        result = loader.check(post_id=args.post_id) if args.check else loader.load(post_id=args.post_id)
        print(json.dumps(result, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
