INSERT INTO cmv.op_labels (
    post_id,
    delta_label,
    source_id
)
VALUES (%s, %s, %s)
ON CONFLICT (post_id) DO NOTHING
RETURNING post_id;