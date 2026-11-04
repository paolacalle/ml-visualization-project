INSERT INTO cmv.comments (
    comment_id,
    post_id,
    parent_comment_id,
    original_parent_id,
    author,
    created_utc,
    score,
    content_available,
    source_id
)
VALUES (
    %(comment_id)s,
    %(post_id)s,
    %(parent_comment_id)s,
    %(original_parent_id)s,
    %(author)s,
    %(created_utc)s,
    %(score)s,
    TRUE,
    %(source_id)s
)
ON CONFLICT (comment_id) DO UPDATE SET
    parent_comment_id = EXCLUDED.parent_comment_id,
    original_parent_id = EXCLUDED.original_parent_id,
    author = EXCLUDED.author,
    created_utc = EXCLUDED.created_utc,
    score = EXCLUDED.score,
    content_available = TRUE,
    source_id = EXCLUDED.source_id
WHERE cmv.comments.content_available = FALSE
  AND cmv.comments.post_id = EXCLUDED.post_id;