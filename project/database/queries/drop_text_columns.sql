ALTER TABLE cmv.posts
    DROP COLUMN body;

ALTER TABLe cmv.comments
    DROP COLUMN body;

ALTER TABLE cmv.op_labels
    DROP COLUMN extra_json; 

ALTER TABLE cmv.pairs
    DROP COLUMN extra_json; 

ALTER TABLE cmv.threads
    DROP COLUMN extra_json;
