INSERT INTO cmv.sources (
    archive_member,
    task,
    original_split
)
VALUES (%s, %s, %s)
ON CONFLICT (source_id) DO NOTHING
RETURNING source_id;