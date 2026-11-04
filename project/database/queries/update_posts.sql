ALTER TABLE cmv.posts
    ADD COLUMN score INTEGER,
    ADD COLUMN ups INTEGER,
    ADD COLUMN downs INTEGER,
    ADD COLUMN num_comments INTEGER,
    ADD COLUMN gilded INTEGER,
    ADD COLUMN permalink TEXT,
    ADD COLUMN over_18 BOOLEAN,
    ADD COLUMN archived BOOLEAN,
    ADD COLUMN stickied BOOLEAN,
    ADD COLUMN distinguished TEXT,
    ADD COLUMN link_flair_text TEXT,
    ADD COLUMN author_flair_text TEXT,
    ADD COLUMN is_edited BOOLEAN,
    ADD COLUMN edited_utc DOUBLE PRECISION;

UPDATE cmv.posts
SET
    score = (extra_json ->> 'score')::INTEGER,
    ups = (extra_json ->> 'ups')::INTEGER,
    downs = (extra_json ->> 'downs')::INTEGER,
    num_comments = (extra_json ->> 'num_comments')::INTEGER,
    gilded = (extra_json ->> 'gilded')::INTEGER,
    permalink = extra_json ->> 'permalink',
    over_18 = (extra_json ->> 'over_18')::BOOLEAN,
    archived = (extra_json ->> 'archived')::BOOLEAN,
    stickied = (extra_json ->> 'stickied')::BOOLEAN,
    distinguished = extra_json ->> 'distinguished',
    link_flair_text = extra_json ->> 'link_flair_text',
    author_flair_text = extra_json ->> 'author_flair_text',
    is_edited = CASE
        WHEN jsonb_typeof(extra_json -> 'edited') = 'number' THEN TRUE
        WHEN jsonb_typeof(extra_json -> 'edited') = 'boolean'
            THEN (extra_json ->> 'edited')::BOOLEAN
        ELSE NULL
    END,
    edited_utc = CASE
        WHEN jsonb_typeof(extra_json -> 'edited') = 'number'
            THEN (extra_json ->> 'edited')::DOUBLE PRECISION
        ELSE NULL
    END;

ALTER TABLE cmv.posts DROP COLUMN extra_json;