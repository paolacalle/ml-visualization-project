WITH ranked AS (
    SELECT
        post_id,
        ROW_NUMBER() OVER (
            PARTITION BY original_split
            ORDER BY created_utc DESC, post_id
        ) AS sample_rank
    FROM cmv.posts
)
SELECT p.*
FROM cmv.posts p
JOIN ranked r ON r.post_id = p.post_id
WHERE r.sample_rank <= 2
ORDER BY p.original_split, r.sample_rank;