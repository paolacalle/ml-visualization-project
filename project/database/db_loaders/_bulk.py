"""Shared PostgreSQL COPY/merge implementation for mapped metadata rows."""

from uuid import uuid4

from psycopg import sql

from parquet_loaders.metadata_staging import TABLE_KEYS, TABLE_SCHEMAS


PG_TYPES = {"string": "text", "int64": "int8", "int32": "int4",
            "double": "float8", "bool": "bool"}


def columns(names, alias=None):
    return sql.SQL(", ").join(
        sql.Identifier(name) if alias is None else sql.Identifier(alias, name)
        for name in names
    )


def row_values(names, alias):
    return sql.SQL("ROW({})").format(columns(names, alias))


class TableLoader:
    """COPY bounded client rows to a private temp table, then merge on the server.

    The caller owns the connection. Each load is atomic; within a caller-owned
    transaction this becomes a savepoint. Identity/ownership columns never change.
    Only posts/comments replace available base metadata; placeholders cannot
    overwrite an available target. Other supplied records are immutable.
    """

    table = None
    immutable = None
    protect_available = False

    def __init__(self, db, *, schema="cmv"):
        """Bind a table subclass to a caller-owned Database/psycopg connection."""
        self.connection = getattr(db, "connection", db)
        self.schema = schema
        self.arrow_schema = TABLE_SCHEMAS[self.table]
        self.names = tuple(self.arrow_schema.names)
        self.keys = TABLE_KEYS[self.table]
        self.target = sql.Identifier(schema, self.table)

    def _join(self, left="t", right="s"):
        return sql.SQL(" AND ").join(
            sql.SQL("{} = {}").format(sql.Identifier(left, key), sql.Identifier(right, key))
            for key in self.keys
        )

    def _before_merge(self, cursor, temporary):
        """Reject conflicting immutable values before any target rows are changed."""
        immutable = self.immutable
        if immutable is None:
            immutable = tuple(name for name in self.names if name not in self.keys)
        if immutable:
            cursor.execute(sql.SQL("""
                SELECT {} FROM {} t JOIN {} s ON {}
                WHERE {} IS DISTINCT FROM {} LIMIT 1
            """).format(columns(self.keys, "s"), self.target, temporary, self._join(),
                         row_values(immutable, "t"), row_values(immutable, "s")))
            conflict = cursor.fetchone()
            if conflict is not None:
                raise ValueError(f"Conflicting {self.table} identity/relationship: {conflict}")

    def _after_merge(self, cursor, inserted):
        """Subclass hook for sequence adjustment or deferred relationship checks."""
        pass

    def _merge(self, cursor, temporary):
        """Merge one staged table and return the affected insert/update row count."""
        query = sql.SQL("INSERT INTO {} AS t ({}) {} SELECT {} FROM {} s ").format(
            self.target, columns(self.names),
            sql.SQL("OVERRIDING SYSTEM VALUE") if self.table == "sources" else sql.SQL(""),
            columns(self.names, "s"), temporary,
        )
        query += sql.SQL("ON CONFLICT ({}) ").format(columns(self.keys))
        if not self.protect_available:
            cursor.execute(query + sql.SQL("DO NOTHING"))
            return cursor.rowcount

        mutable = tuple(name for name in self.names if name not in (*self.keys, *self.immutable))
        assignments = sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(name), sql.Identifier(name))
            for name in mutable
        )
        cursor.execute(query + sql.SQL("""
            DO UPDATE SET {} WHERE
                (NOT t.content_available OR EXCLUDED.content_available)
                AND {} IS DISTINCT FROM {}
        """).format(assignments, row_values(mutable, "t"), row_values(mutable, "excluded")))
        return cursor.rowcount

    def load(self, rows):
        """COPY and merge an iterable of dictionaries matching this table's schema.

        A private temporary table receives every selected row before merging;
        the client holds only the incoming batch. Primary-key duplicates fail.
        Return processed/inserted/updated/unchanged counts. All writes are atomic
        or form a savepoint within the caller's outer transaction. The input
        generator is closed after consumption or failure; the DB stays open.
        """
        temporary_name = f"cmv_import_{self.table}_{uuid4().hex}"
        temporary = sql.Identifier("pg_temp", temporary_name)
        names_set = set(self.names)
        definitions = sql.SQL(", ").join(
            sql.SQL("{} {} {}").format(
                sql.Identifier(field.name), sql.SQL(PG_TYPES[str(field.type)]),
                sql.SQL("NOT NULL") if not field.nullable else sql.SQL(""),
            ) for field in self.arrow_schema
        )
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE TEMP TABLE {} ({}, PRIMARY KEY ({})) ON COMMIT DROP").format(
                sql.Identifier(temporary_name), definitions, columns(self.keys)))
            copied = 0
            with cursor.copy(sql.SQL("COPY {} ({}) FROM STDIN").format(temporary, columns(self.names))) as copy:
                try:
                    for row in rows:
                        if set(row) != names_set:
                            raise ValueError(f"Expected {self.table} columns: {self.names}")
                        copy.write_row(tuple(row[name] for name in self.names))
                        copied += 1
                finally:
                    close = getattr(rows, "close", None)
                    if close is not None:
                        close()
            # Refresh estimates for the server-side joins on larger tables.
            cursor.execute(sql.SQL("ANALYZE {}").format(temporary))
            self._before_merge(cursor, temporary)
            cursor.execute(sql.SQL("SELECT count(*) FROM {} t JOIN {} s ON {}").format(
                self.target, temporary, self._join()))
            existed = cursor.fetchone()[0]
            affected = self._merge(cursor, temporary)
            inserted = copied - existed
            self._after_merge(cursor, inserted)
            cursor.execute(sql.SQL("SELECT count(*) FROM {} t JOIN {} s ON {}").format(
                self.target, temporary, self._join()))
            if cursor.fetchone()[0] != copied:
                raise ValueError(f"Not all {self.table} keys were stored.")
            # Full row verification, except intentional preservation of an
            # existing available record against an incoming placeholder.
            condition = sql.SQL("{} IS DISTINCT FROM {}").format(
                row_values(self.names, "t"), row_values(self.names, "s"))
            if self.protect_available:
                condition = sql.SQL("({}) AND NOT (t.content_available AND NOT s.content_available)").format(condition)
            cursor.execute(sql.SQL("SELECT 1 FROM {} t JOIN {} s ON {} WHERE {} LIMIT 1").format(
                self.target, temporary, self._join(), condition))
            if cursor.fetchone() is not None:
                raise ValueError(f"Stored {self.table} values do not match the prepared rows.")
            cursor.execute(sql.SQL("DROP TABLE {}").format(temporary))
        return {"processed": copied, "inserted": inserted, "updated": affected - inserted,
                "unchanged": copied - affected}
