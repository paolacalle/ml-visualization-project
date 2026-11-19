import tarfile
from psycopg import sql

from ._bulk import TableLoader


class SourcesLoader(TableLoader):
    """Preserve snapshot IDs; reject conflicting IDs or archive-member mappings."""

    table = "sources"

    def _before_merge(self, cursor, temporary):
        cursor.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(self.target))
        super()._before_merge(cursor, temporary)
        cursor.execute(sql.SQL("""
            SELECT s.archive_member FROM {} s JOIN {} t USING (archive_member)
            WHERE s.source_id <> t.source_id LIMIT 1
        """).format(temporary, self.target))
        if cursor.fetchone() is not None:
            raise ValueError("Source mapping differs; remap by archive_member before importing.")

    def _after_merge(self, cursor, inserted):
        if not inserted:
            return
        cursor.execute("""
            SELECT n.nspname, c.relname FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.oid = pg_get_serial_sequence(%s, 'source_id')::regclass
        """, (self.target.as_string(self.connection),))
        sequence = cursor.fetchone()
        if sequence is None:
            return
        identifier = sql.Identifier(*sequence)
        cursor.execute(sql.SQL("SELECT last_value, is_called FROM {}").format(identifier))
        last_value, is_called = cursor.fetchone()
        cursor.execute(sql.SQL("SELECT max(source_id) FROM {}").format(self.target))
        next_id = max(cursor.fetchone()[0] + 1, last_value + int(is_called))
        if next_id > 2**63 - 1:
            raise ValueError("No remaining BIGINT source IDs after this snapshot.")
        # ALTER SEQUENCE RESTART is transactional, unlike setval(). Never rewind
        # a sequence that has already allocated IDs beyond the supplied snapshot.
        cursor.execute(sql.SQL("ALTER SEQUENCE {} RESTART WITH {}").format(identifier, sql.Literal(next_id)))


def load_sources(archive_path, db):
    """Legacy archive-member registration; new imports preserve IDs with SourcesLoader."""
    with tarfile.open(archive_path, "r|bz2") as archive:
        for member in archive:
            if not member.isfile():
                continue

            if not member.name.endswith(".jsonlist.bz2"):
                continue

            task = member.name.split("/")[0]
            filename = member.name.rsplit("/", 1)[-1]
            split = filename.split("_")[0]

            result = db.execute(
                "insert_source.sql",
                (member.name, task, split),
            )
            
            if result:
                print(f"Inserted: {member.name}")
            else:
                print(f"Already exists: {member.name}")
