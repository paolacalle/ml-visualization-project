SELECT
    (SELECT COUNT(*) FROM cmv.posts) AS post_count,
    pg_total_relation_size('cmv.posts') AS table_bytes,
    pg_size_pretty(
        pg_total_relation_size('cmv.posts')
    ) AS table_size_with_indexes,
    pg_size_pretty(
        pg_database_size(current_database())
    ) AS database_size;