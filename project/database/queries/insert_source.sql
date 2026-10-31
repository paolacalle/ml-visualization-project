INSERT INTO cmv.sources (
    archive_member,
    task,
    original_split
)
VALUES (%s, %s, %s)
ON CONFLICT (archive_member) DO NOTHING
RETURNING (source_id, archive_member);