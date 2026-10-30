INSERT INTO cmv.posts (
    post_id, original_split, title, body, author, created_utc,
    content_available, source_id,
    score, ups, downs, num_comments, gilded, permalink,
    over_18, archived, stickied, distinguished,
    link_flair_text, author_flair_text, is_edited, edited_utc
)
VALUES (
    %(post_id)s, %(original_split)s, %(title)s, %(body)s,
    %(author)s, %(created_utc)s, TRUE, %(source_id)s,
    %(score)s, %(ups)s, %(downs)s, %(num_comments)s,
    %(gilded)s, %(permalink)s, %(over_18)s, %(archived)s,
    %(stickied)s, %(distinguished)s, %(link_flair_text)s,
    %(author_flair_text)s, %(is_edited)s, %(edited_utc)s
)
ON CONFLICT (post_id) DO NOTHING
RETURNING post_id;