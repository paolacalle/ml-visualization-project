"""Stage all seven metadata tables locally before Parquet export."""

import sqlite3

import pyarrow as pa

from .comments import COMMENT_SCHEMA
from .op_labels import OP_LABEL_SCHEMA
from .pairs import PAIR_SCHEMA, THREAD_SCHEMA, THREAD_COMMENT_SCHEMA
from .posts import POST_SCHEMA
from .source_map import SOURCE_SCHEMA, map_sources
from .staging import CommentStaging


TABLE_SCHEMAS = {
    "sources": SOURCE_SCHEMA,
    "posts": POST_SCHEMA,
    "comments": COMMENT_SCHEMA,
    "op_labels": OP_LABEL_SCHEMA,
    "pairs": PAIR_SCHEMA,
    "threads": THREAD_SCHEMA,
    "thread_comments": THREAD_COMMENT_SCHEMA,
}
TABLE_KEYS = {
    "sources": ("source_id",), "posts": ("post_id",),
    "comments": ("comment_id",), "op_labels": ("post_id",),
    "pairs": ("pair_id",), "threads": ("thread_id",),
    "thread_comments": ("thread_id", "comment_id"),
}
SQLITE_TYPES = {
    pa.string(): "TEXT", pa.int32(): "INTEGER", pa.int64(): "INTEGER",
    pa.float64(): "REAL", pa.bool_(): "INTEGER",
}


def _columns(schema):
    # Names and types come exclusively from the fixed Arrow schemas above.
    return ", ".join(
        f"{field.name} {SQLITE_TYPES[field.type]}"
        + (" NOT NULL" if not field.nullable else "")
        for field in schema
    )


class MetadataStaging(CommentStaging):
    """Reconcile all mapped tables using a validated, offline source snapshot.

    Full-discussion post metadata takes precedence over OP-task metadata, then
    pair-task metadata. Referenced posts/comments remain explicit placeholders
    until actual records arrive. The caller retains/removes the SQLite file.
    """

    def __init__(self, path, source_map):
        """Create new local staging with the validated six-source snapshot.

        ``path`` must be new and have an existing parent directory. The source
        mapping is local input; no PostgreSQL connection is opened.
        """
        sources = map_sources(source_map)
        self._sources = {row["source_id"]: row for row in sources}
        super().__init__(path)
        try:
            self._create_tables()
            with self.transaction():
                self.connection.executemany(
                    "INSERT INTO sources VALUES (?, ?, ?, ?)",
                    [tuple(row[name] for name in SOURCE_SCHEMA.names) for row in sources],
                )
        except BaseException:
            self.close()
            raise

    def _create_tables(self):
        """Create typed staging tables, relationship constraints, and canonical views."""
        self.connection.executescript(f"""
            CREATE TABLE sources ({_columns(SOURCE_SCHEMA)},
                PRIMARY KEY (source_id), UNIQUE (archive_member));
            CREATE TABLE post_candidates ({_columns(POST_SCHEMA)},
                task TEXT NOT NULL CHECK (task IN ('all', 'op_task', 'pair_task')),
                PRIMARY KEY (post_id, task),
                FOREIGN KEY (source_id) REFERENCES sources(source_id),
                FOREIGN KEY (post_id, original_split)
                    REFERENCES post_splits(post_id, original_split));
            CREATE TABLE op_labels ({_columns(OP_LABEL_SCHEMA)},
                PRIMARY KEY (post_id),
                FOREIGN KEY (post_id) REFERENCES post_splits(post_id),
                FOREIGN KEY (source_id) REFERENCES sources(source_id));
            CREATE TABLE pairs ({_columns(PAIR_SCHEMA)},
                PRIMARY KEY (pair_id), UNIQUE (source_id, source_line),
                UNIQUE (pair_id, post_id), CHECK (source_line > 0),
                FOREIGN KEY (post_id) REFERENCES post_splits(post_id),
                FOREIGN KEY (source_id) REFERENCES sources(source_id));
            CREATE TABLE threads ({_columns(THREAD_SCHEMA)},
                PRIMARY KEY (thread_id), UNIQUE (pair_id, outcome),
                UNIQUE (thread_id, post_id),
                CHECK (outcome IN ('positive', 'negative')),
                FOREIGN KEY (pair_id, post_id) REFERENCES pairs(pair_id, post_id));
            CREATE TABLE thread_comments ({_columns(THREAD_COMMENT_SCHEMA)},
                PRIMARY KEY (thread_id, comment_id), UNIQUE (thread_id, position),
                CHECK (position >= 0),
                FOREIGN KEY (thread_id, post_id) REFERENCES threads(thread_id, post_id),
                FOREIGN KEY (comment_id, post_id) REFERENCES comment_keys(comment_id, post_id));
            CREATE INDEX thread_comments_comment_idx ON thread_comments(comment_id, post_id);
        """)

        expressions = []
        for name in POST_SCHEMA.names:
            if name in ("post_id", "original_split"):
                expression = f"k.{name}"
            elif name == "content_available":
                expression = "(a.post_id IS NOT NULL OR o.post_id IS NOT NULL OR p.post_id IS NOT NULL)"
            elif name == "edited_utc":
                expression = ("CASE WHEN a.is_edited IS NOT NULL THEN a.edited_utc "
                              "WHEN o.is_edited IS NOT NULL THEN o.edited_utc ELSE p.edited_utc END")
            else:
                expression = f"COALESCE(a.{name}, o.{name}, p.{name})"
            expressions.append(f"{expression} AS {name}")
        self.connection.execute(f"""
            CREATE VIEW reconciled_posts AS SELECT {', '.join(expressions)}
            FROM post_splits k
            LEFT JOIN post_candidates a ON a.post_id = k.post_id AND a.task = 'all'
            LEFT JOIN post_candidates o ON o.post_id = k.post_id AND o.task = 'op_task'
            LEFT JOIN post_candidates p ON p.post_id = k.post_id AND p.task = 'pair_task'
        """)

    @staticmethod
    def _check_row(table, row):
        """Reject incorrect columns, required NULLs, booleans/integers, or ranges."""
        schema = TABLE_SCHEMAS[table]
        if set(row) != set(schema.names):
            raise ValueError(f"Expected {table} columns matching its Arrow schema.")
        for field in schema:
            value = row[field.name]
            if value is None:
                if not field.nullable:
                    raise ValueError(f"{table}.{field.name} cannot be NULL.")
            elif pa.types.is_boolean(field.type) and type(value) is not bool:
                raise ValueError(f"{table}.{field.name} must be boolean.")
            elif pa.types.is_integer(field.type) and type(value) is not int:
                raise ValueError(f"{table}.{field.name} must be an integer.")
        # Check strings and numeric ranges before inserting into SQLite.
        pa.Table.from_pylist([row], schema=schema)

    def _source(self, source_id, *, task=None, split=None):
        """Resolve an existing snapshot ID and optionally enforce task/split ownership."""
        source = self._sources.get(source_id) if type(source_id) is int else None
        if source is None:
            raise ValueError(f"Unknown source_id: {source_id}")
        if task is not None and source["task"] != task:
            raise ValueError(f"Source {source_id} is not a {task} source.")
        if split is not None and source["original_split"] != split:
            raise ValueError(f"Source {source_id} does not match split {split}.")
        return source

    def add_comments(self, rows, *, task, split):
        """Validate comment schema/provenance, then use atomic comment reconciliation."""
        def checked():
            for row in rows:
                self._check_row("comments", row)
                self._source(row["source_id"], task=task, split=split)
                yield row
        super().add_comments(checked(), task=task, split=split)

    def add_posts(self, rows, *, task="all"):
        """Merge actual post candidates while preserving split and source provenance.

        Canonical precedence is full discussion, OP task, then pair task. NULL
        attributes may be filled from fallback sources; contradictory same-task
        values fail. Referenced post placeholders are created automatically by
        labels/comments/pairs and become available when a candidate arrives.
        """
        with self.transaction():
            for row in rows:
                self._check_row("posts", row)
                self._source(row["source_id"], task=task, split=row["original_split"])
                if row["content_available"] is not True:
                    raise ValueError("Supply actual posts; referenced post placeholders are automatic.")
                if row["edited_utc"] is not None and row["is_edited"] is not True:
                    raise ValueError("An edit timestamp requires is_edited=True.")
                self._ensure_post(row["post_id"], row["original_split"])
                previous = self.connection.execute(
                    "SELECT * FROM post_candidates WHERE post_id = ? AND task = ?",
                    (row["post_id"], task),
                ).fetchone()
                merged = dict(row)
                if previous is not None:
                    for field in POST_SCHEMA.names:
                        old, new = previous[field], row[field]
                        if old is not None and new is not None and old != new:
                            raise ValueError(f"Conflicting {task} {field} for {row['post_id']}")
                        merged[field] = old if old is not None else new
                names = [*POST_SCHEMA.names, "task"]
                updates = ", ".join(f"{name} = excluded.{name}" for name in POST_SCHEMA.names)
                self.connection.execute(
                    f"INSERT INTO post_candidates ({', '.join(names)}) "
                    f"VALUES ({', '.join('?' for _ in names)}) "
                    f"ON CONFLICT (post_id, task) DO UPDATE SET {updates}",
                    (*[merged[name] for name in POST_SCHEMA.names], task),
                )

    def _add_rows(self, table, rows):
        """Insert relational rows idempotently; conflicting repeats are errors."""
        schema = TABLE_SCHEMAS[table]
        keys = TABLE_KEYS[table]
        with self.transaction():
            for row in rows:
                self._check_row(table, row)
                if table in ("op_labels", "pairs"):
                    task = "op_task" if table == "op_labels" else "pair_task"
                    source = self._source(row["source_id"], task=task)
                    self._ensure_post(row["post_id"], source["original_split"])
                    if table == "pairs":
                        expected = f"{source['archive_member']}#{row['source_line']}"
                        if row["source_line"] <= 0 or row["pair_id"] != expected:
                            raise ValueError(f"Pair ID must match source member/line: {expected}")
                elif table == "threads":
                    if row["outcome"] not in ("positive", "negative"):
                        raise ValueError(f"Invalid thread outcome: {row['outcome']}")
                    if row["thread_id"] != f"{row['pair_id']}:{row['outcome']}":
                        raise ValueError("Thread ID must match pair/outcome.")
                elif table == "thread_comments":
                    thread = self.connection.execute("""
                        SELECT t.post_id, p.original_split FROM threads t
                        JOIN post_splits p ON p.post_id = t.post_id WHERE t.thread_id = ?
                    """, (row["thread_id"],)).fetchone()
                    if thread is None or thread["post_id"] != row["post_id"]:
                        raise ValueError("Membership must reference an existing thread in the same post.")
                    self._ensure_key(row["comment_id"], row["post_id"], thread["original_split"])

                where = " AND ".join(f"{key} = ?" for key in keys)
                previous = self.connection.execute(
                    f"SELECT * FROM {table} WHERE {where}", tuple(row[key] for key in keys),
                ).fetchone()
                if previous is not None:
                    if dict(previous) != row:
                        raise ValueError(f"Conflicting {table} row: {tuple(row[key] for key in keys)}")
                    continue
                try:
                    self.connection.execute(
                        f"INSERT INTO {table} ({', '.join(schema.names)}) "
                        f"VALUES ({', '.join('?' for _ in schema)})",
                        tuple(row[name] for name in schema.names),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"Invalid {table} relationship or unique key: {exc}") from exc

    def add_op_labels(self, rows):
        """Stage supplied boolean labels; create referenced post identities if needed."""
        self._add_rows("op_labels", rows)

    def add_pairs(self, rows):
        """Stage stable pair IDs tied to the snapshot source and physical JSON line."""
        self._add_rows("pairs", rows)

    def add_threads(self, rows):
        """Stage positive/negative threads after their pair rows are present."""
        self._add_rows("threads", rows)

    def add_thread_comments(self, rows):
        """Stage ordered memberships; register missing comment identities as placeholders."""
        self._add_rows("thread_comments", rows)

    def add_pair(self, mapped):
        """Atomically stage the four outputs of map_pair(), including comments."""
        if set(mapped) != {"pair", "threads", "thread_comments", "comments"}:
            raise ValueError("Expected all four outputs from map_pair().")
        pair = mapped["pair"]
        self._check_row("pairs", pair)
        source = self._source(pair["source_id"], task="pair_task")
        expected_threads = {f"{pair['pair_id']}:{outcome}" for outcome in ("positive", "negative")}
        if (len(mapped["threads"]) != 2
                or {row["thread_id"] for row in mapped["threads"]} != expected_threads):
            raise ValueError("A mapped pair must contain its positive and negative threads.")
        for row in [*mapped["threads"], *mapped["thread_comments"], *mapped["comments"]]:
            if row["post_id"] != pair["post_id"]:
                raise ValueError("Every mapped pair row must belong to the same post.")
        for row in mapped["thread_comments"]:
            if row["thread_id"] not in expected_threads:
                raise ValueError("Membership belongs to another pair.")
        if any(row["source_id"] != pair["source_id"] for row in mapped["comments"]):
            raise ValueError("Pair comment provenance must match the pair source.")
        with self.transaction():
            self.add_pairs([pair])
            self.add_comments(mapped["comments"], task="pair_task", split=source["original_split"])
            self.add_threads(mapped["threads"])
            self.add_thread_comments(mapped["thread_comments"])

    def iter_table_batches(self, table, *, batch_size=1000, split=None):
        """Yield lists of DB-shaped dictionaries, optionally filtered by split.

        All seven table names are accepted. Each batch has at most ``batch_size``
        rows; posts/comments include their explicit unavailable placeholders.
        Partition hints stay internal and never become extra DB columns.
        """
        if table not in TABLE_SCHEMAS:
            raise ValueError(f"Unknown metadata table: {table}")
        if table == "comments":
            yield from self.iter_batches(batch_size=batch_size, split=split)
            return
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if split is not None and split not in ("train", "heldout"):
            raise ValueError(f"Invalid original split: {split}")
        schema = TABLE_SCHEMAS[table]
        storage = "reconciled_posts" if table == "posts" else table
        query = f"SELECT {', '.join('t.' + name for name in schema.names)} FROM {storage} t"
        params = ()
        if split is not None:
            if table in ("posts", "sources"):
                query += " WHERE t.original_split = ?"
            else:
                query += " JOIN post_splits p ON p.post_id = t.post_id WHERE p.original_split = ?"
            params = (split,)
        query += " ORDER BY " + ", ".join("t." + key for key in TABLE_KEYS[table])
        cursor = self.connection.execute(query, params)
        try:
            while batch := cursor.fetchmany(batch_size):
                result = [dict(row) for row in batch]
                for row in result:
                    for field in schema:
                        if pa.types.is_boolean(field.type) and row[field.name] is not None:
                            row[field.name] = bool(row[field.name])
                yield result
        finally:
            cursor.close()

    def validate(self):
        """Validate every staged table and return table/placeholder counts.

        Check foreign keys, source/split consistency, exactly two pair outcomes,
        contiguous membership positions, Arrow field values, and parent cycles.
        Indexed temporary graph traversal stays on SQLite rather than storing
        the full graph as Python objects. Invalid state raises ``ValueError``.
        """
        bad = self.connection.execute("PRAGMA foreign_key_check").fetchone()
        if bad is not None:
            raise ValueError(f"Foreign-key violation: {tuple(bad)}")

        for table, task in (("op_labels", "op_task"), ("pairs", "pair_task")):
            bad = self.connection.execute(f"""
                SELECT r.post_id FROM {table} r JOIN sources s ON s.source_id = r.source_id
                JOIN post_splits p ON p.post_id = r.post_id
                WHERE s.task <> ? OR s.original_split <> p.original_split LIMIT 1
            """, (task,)).fetchone()
            if bad is not None:
                raise ValueError(f"Invalid {table} source/split for {bad[0]}")
        for table, keys in (("candidates", "comment_keys"), ("post_candidates", "post_splits")):
            join = "k.comment_id = c.comment_id" if table == "candidates" else "k.post_id = c.post_id"
            bad = self.connection.execute(f"""
                SELECT c.post_id FROM {table} c JOIN {keys} k ON {join}
                LEFT JOIN sources s ON s.source_id = c.source_id
                WHERE s.source_id IS NULL OR s.task <> c.task
                    OR s.original_split <> k.original_split LIMIT 1
            """).fetchone()
            if bad is not None:
                raise ValueError(f"Invalid {table} source/split for {bad[0]}")

        bad = self.connection.execute("""
            SELECT p.pair_id FROM pairs p LEFT JOIN threads t ON t.pair_id = p.pair_id
            GROUP BY p.pair_id HAVING count(t.thread_id) <> 2 LIMIT 1
        """).fetchone()
        if bad is not None:
            raise ValueError(f"Pair requires positive and negative threads: {bad[0]}")
        bad = self.connection.execute("""
            SELECT thread_id FROM thread_comments GROUP BY thread_id
            HAVING min(position) <> 0 OR max(position) <> count(*) - 1 LIMIT 1
        """).fetchone()
        if bad is not None:
            raise ValueError(f"Non-contiguous membership positions: {bad[0]}")

        # Build an indexed graph on disk. Traverse outward from known or unknown
        # roots; a component without any reachable root contains a parent cycle.
        self.connection.execute("""
            CREATE TEMP TABLE validation_graph AS
            SELECT comment_id, parent_comment_id FROM reconciled_comments
        """)
        try:
            self.connection.execute("CREATE INDEX validation_parent_idx ON validation_graph(parent_comment_id)")
            bad = self.connection.execute("""
                WITH RECURSIVE reached(comment_id) AS (
                    SELECT comment_id FROM validation_graph WHERE parent_comment_id IS NULL
                    UNION ALL
                    SELECT g.comment_id FROM validation_graph g
                    JOIN reached r ON g.parent_comment_id = r.comment_id
                )
                SELECT g.comment_id FROM validation_graph g
                LEFT JOIN reached r ON r.comment_id = g.comment_id
                WHERE r.comment_id IS NULL LIMIT 1
            """).fetchone()
            if bad is not None:
                raise ValueError(f"Comment parent cycle involving {bad[0]}")
        finally:
            self.connection.execute("DROP TABLE validation_graph")

        counts = {}
        for table, schema in TABLE_SCHEMAS.items():
            count = 0
            for batch in self.iter_table_batches(table):
                pa.Table.from_pylist(batch, schema=schema)
                count += len(batch)
            counts[table] = count
        placeholders = {
            table: self.connection.execute(
                f"SELECT count(*) FROM reconciled_{table} WHERE content_available = 0"
            ).fetchone()[0]
            for table in ("posts", "comments")
        }
        return {"tables": counts, "placeholders": placeholders}
