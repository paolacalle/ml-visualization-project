SELECT
    COUNT(*) AS comment_count,
    pg_size_pretty(
        pg_total_relation_size('cmv.comments')
    ) AS size_with_indexes
FROM cmv.comments;