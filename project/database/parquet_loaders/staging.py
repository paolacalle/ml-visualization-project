"""Local, disk-backed reconciliation of mapped comment metadata."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from parquet_loaders.comments import COMMENT_SCHEMA


COMMENT_COLUMNS = tuple(COMMENT_SCHEMA.names)
SELECT_COLUMNS = ", ".join(COMMENT_COLUMNS)

# add index to speed up lookup

class CommentStaging:
    """Merge comment candidates without retaining the dataset in Python memory.

    The file must be new. Closing retains it for inspection; the caller removes
    it after successful export validation. Use batches for archive ingestion.
    This class does not open the archive or connect to PostgreSQL.
    """

    def __init__(self, path):
        """Create a new SQLite file in an existing parent directory.

        Existing files raise ``FileExistsError``. The file has indexed identity
        and parent-reference lookups and stays on disk after ``close()``.
        """
        self.path = Path(path)
        # Refuse to reuse or overwrite a previous preparation run.
        with self.path.open("xb"):
            pass
        self.connection = sqlite3.connect(self.path)
        self._savepoint_number = 0
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE post_splits (
                post_id TEXT PRIMARY KEY,
                original_split TEXT NOT NULL,
                UNIQUE (post_id, original_split)
            );
            CREATE TABLE comment_keys (
                comment_id TEXT PRIMARY KEY,
                post_id TEXT NOT NULL,
                original_split TEXT NOT NULL,
                UNIQUE (comment_id, post_id),
                FOREIGN KEY (post_id, original_split)
                    REFERENCES post_splits(post_id, original_split)
            );
            CREATE INDEX comment_keys_post_idx ON comment_keys(post_id, comment_id);
            CREATE TABLE candidates (
                comment_id TEXT NOT NULL,
                post_id TEXT NOT NULL,
                task TEXT NOT NULL CHECK (task IN ('all', 'pair_task')),
                parent_comment_id TEXT,
                original_parent_id TEXT,
                author TEXT,
                created_utc REAL,
                score INTEGER,
                content_available INTEGER NOT NULL CHECK (content_available = 1),
                source_id INTEGER,
                PRIMARY KEY (comment_id, task),
                FOREIGN KEY (comment_id, post_id)
                    REFERENCES comment_keys(comment_id, post_id),
                FOREIGN KEY (parent_comment_id, post_id)
                    REFERENCES comment_keys(comment_id, post_id)
            );
            CREATE INDEX candidates_parent_idx ON candidates(parent_comment_id, post_id);
            CREATE VIEW reconciled_comments AS
            SELECT k.comment_id, k.post_id, k.original_split,
                CASE WHEN a.original_parent_id IS NOT NULL
                     THEN a.parent_comment_id ELSE p.parent_comment_id
                END AS parent_comment_id,
                COALESCE(a.original_parent_id, p.original_parent_id) AS original_parent_id,
                COALESCE(a.author, p.author) AS author,
                COALESCE(a.created_utc, p.created_utc) AS created_utc,
                COALESCE(a.score, p.score) AS score,
                (a.comment_id IS NOT NULL OR p.comment_id IS NOT NULL) AS content_available,
                COALESCE(a.source_id, p.source_id) AS source_id
            FROM comment_keys k
            LEFT JOIN candidates a ON a.comment_id = k.comment_id AND a.task = 'all'
            LEFT JOIN candidates p ON p.comment_id = k.comment_id AND p.task = 'pair_task';
        """)

    @contextmanager
    def transaction(self):
        """Allow a whole mapped pair to contain smaller atomic batch writes."""
        self._savepoint_number += 1
        name = f"batch_{self._savepoint_number}"
        self.connection.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self.connection.execute(f"ROLLBACK TO {name}")
            self.connection.execute(f"RELEASE {name}")
            raise
        else:
            self.connection.execute(f"RELEASE {name}")

    def _ensure_post(self, post_id, split):
        """Insert an unknown post identity or verify its immutable split by index."""
        if not isinstance(post_id, str) or not post_id.startswith("t3_") or len(post_id) <= 3:
            raise ValueError(f"Invalid post ID: {post_id}")
        if split not in ("train", "heldout"):
            raise ValueError(f"Invalid original split: {split}")
        existing = self.connection.execute(
            "SELECT original_split FROM post_splits WHERE post_id = ?", (post_id,)
        ).fetchone()
        if existing is not None:
            if existing["original_split"] != split:
                raise ValueError(f"Conflicting original split for {post_id}")
            return
        self.connection.execute("INSERT INTO post_splits VALUES (?, ?)", (post_id, split))

    def _ensure_key(self, comment_id, post_id, split):
        """Verify/insert an identity without updating an already-correct key.

        The early return avoids unnecessary incoming foreign-key checks. The
        same helper registers missing parents as metadata placeholders.
        """
        if not isinstance(comment_id, str) or not comment_id.startswith("t1_") or len(comment_id) <= 3:
            raise ValueError(f"Invalid comment ID: {comment_id}")
        self._ensure_post(post_id, split)
        # Keys are immutable. A no-op UPDATE still makes SQLite check incoming
        # foreign-key references; use an indexed read for an existing identity.
        existing = self.connection.execute(
            "SELECT post_id FROM comment_keys WHERE comment_id = ?", (comment_id,)
        ).fetchone()
        if existing is not None:
            if existing["post_id"] != post_id:
                raise ValueError(f"{comment_id} belongs to another post")
            return
        self.connection.execute("INSERT INTO comment_keys VALUES (?, ?, ?)",
                                (comment_id, post_id, split))

    def add_comments(self, rows, *, task, split):
        """Atomically merge mapped ``COMMENT_SCHEMA`` rows from one task/split.

        Accept ``task='all'`` or ``'pair_task'`` and an original train/heldout
        split. Prefer full-discussion values, fill NULLs from pair metadata, and
        keep placeholders for absent parents. Contradictory ownership, splits,
        parent links, or same-task values raise and roll back this batch.
        """
        if task not in ("all", "pair_task"):
            raise ValueError("Expected task 'all' or 'pair_task'.")
        if split not in ("train", "heldout"):
            raise ValueError(f"Invalid original split: {split}")

        with self.transaction():
            for row in rows:
                if set(row) != set(COMMENT_COLUMNS):
                    raise ValueError("Expected a mapped row matching COMMENT_SCHEMA.")
                if row["content_available"] is not True:
                    raise ValueError("Supply actual comment rows; parent placeholders are automatic.")
                comment_id, post_id = row["comment_id"], row["post_id"]
                parent, original = row["parent_comment_id"], row["original_parent_id"]
                if parent == comment_id:
                    raise ValueError(f"{comment_id} cannot reply to itself.")
                if parent is not None and original != parent:
                    raise ValueError(f"Inconsistent parent fields for {comment_id}")
                if parent is None and original not in (None, post_id):
                    raise ValueError(f"Inconsistent parent fields for {comment_id}")

                self._ensure_key(comment_id, post_id, split)
                if parent is not None:
                    self._ensure_key(parent, post_id, split)

                # A parent relationship must agree wherever it is known.
                known = self.connection.execute("""
                    SELECT original_parent_id FROM candidates
                    WHERE comment_id = ? AND original_parent_id IS NOT NULL
                """, (comment_id,)).fetchall()
                if original is not None and any(r[0] != original for r in known):
                    raise ValueError(f"Conflicting parent for {comment_id}")

                previous = self.connection.execute(
                    "SELECT * FROM candidates WHERE comment_id = ? AND task = ?",
                    (comment_id, task),
                ).fetchone()
                merged = dict(row)
                if previous is not None:
                    for field in COMMENT_COLUMNS:
                        old, new = previous[field], row[field]
                        if old is not None and new is not None and old != new:
                            raise ValueError(f"Conflicting {task} {field} for {comment_id}")
                        merged[field] = old if old is not None else new

                self.connection.execute(f"""
                    INSERT INTO candidates ({SELECT_COLUMNS}, task)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (comment_id, task) DO UPDATE SET
                        parent_comment_id = excluded.parent_comment_id,
                        original_parent_id = excluded.original_parent_id,
                        author = excluded.author,
                        created_utc = excluded.created_utc,
                        score = excluded.score,
                        source_id = excluded.source_id
                """, (*[merged[field] for field in COMMENT_COLUMNS], task))

    @staticmethod
    def _as_comment(row):
        result = dict(row)
        result["content_available"] = bool(result["content_available"])
        return result

    def get_comment(self, comment_id):
        """Return one reconciled comment/placeholder dictionary, or None if unknown."""
        row = self.connection.execute(
            f"SELECT {SELECT_COLUMNS} FROM reconciled_comments WHERE comment_id = ?",
            (comment_id,),
        ).fetchone()
        return None if row is None else self._as_comment(row)

    def iter_batches(self, *, batch_size=1000, split=None):
        """Yield DB-shaped rows, including placeholders, in bounded batches."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if split is not None and split not in ("train", "heldout"):
            raise ValueError(f"Invalid original split: {split}")
        query = f"SELECT {SELECT_COLUMNS} FROM reconciled_comments"
        params = ()
        if split is not None:
            query += " WHERE original_split = ?"
            params = (split,)
        query += " ORDER BY post_id, comment_id"
        cursor = self.connection.execute(query, params)
        try:
            while batch := cursor.fetchmany(batch_size):
                yield [self._as_comment(row) for row in batch]
        finally:
            cursor.close()

    def close(self):
        """Close SQLite while retaining its file for inspection or later cleanup."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
