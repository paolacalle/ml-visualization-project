SET LOCAL search_path TO cmv;

-- DO NOT RUN UNLESS YOU REALLY NEED TO 

-- Drop child tables before their parents.
-- DROP TABLE IF EXISTS thread_comments;
-- DROP TABLE IF EXISTS threads;
-- DROP TABLE IF EXISTS pairs;
-- DROP TABLE IF EXISTS op_labels;
-- DROP TABLE IF EXISTS comments;
-- DROP TABLE IF EXISTS posts;
-- DROP TABLE IF EXISTS sources;

ALTER TABLE cmv.comments DROP COLUMN extra_json;
