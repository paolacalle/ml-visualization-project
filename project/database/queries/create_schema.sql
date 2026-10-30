CREATE SCHEMA IF NOT EXISTS cmv;

SET LOCAL search_path TO cmv;

CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    archive_member TEXT NOT NULL UNIQUE,
    task TEXT NOT NULL
        CHECK (task IN ('all', 'op_task', 'pair_task')),
    original_split TEXT NOT NULL
        CHECK (original_split IN ('train', 'heldout'))
);

CREATE TABLE posts (
    post_id TEXT PRIMARY KEY,
    original_split TEXT NOT NULL
        CHECK (original_split IN ('train', 'heldout')),
    title TEXT,
    body TEXT,
    author TEXT,
    created_utc DOUBLE PRECISION,
    content_available BOOLEAN NOT NULL DEFAULT FALSE,
    source_id TEXT REFERENCES sources(source_id),
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL REFERENCES posts(post_id),
    parent_comment_id TEXT,
    original_parent_id TEXT,
    author TEXT,
    body TEXT,
    created_utc DOUBLE PRECISION,
    score INTEGER,
    content_available BOOLEAN NOT NULL DEFAULT FALSE,
    source_id TEXT REFERENCES sources(source_id),
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    UNIQUE (comment_id, post_id),

    CHECK (
        parent_comment_id IS NULL
        OR parent_comment_id <> comment_id
    ),

    FOREIGN KEY (parent_comment_id, post_id)
        REFERENCES comments(comment_id, post_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE op_labels (
    post_id TEXT PRIMARY KEY REFERENCES posts(post_id),
    delta_label BOOLEAN NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE pairs (
    pair_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL REFERENCES posts(post_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_line INTEGER NOT NULL CHECK (source_line > 0),
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    UNIQUE (source_id, source_line),
    UNIQUE (pair_id, post_id)
);

CREATE TABLE threads (
    thread_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL,
    post_id TEXT NOT NULL,
    outcome TEXT NOT NULL
        CHECK (outcome IN ('positive', 'negative')),
    ancestor_id TEXT,
    author TEXT,
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    FOREIGN KEY (pair_id, post_id)
        REFERENCES pairs(pair_id, post_id),

    UNIQUE (pair_id, outcome),
    UNIQUE (thread_id, post_id)
);

CREATE TABLE thread_comments (
    thread_id TEXT NOT NULL,
    comment_id TEXT NOT NULL,
    post_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),

    PRIMARY KEY (thread_id, comment_id),
    UNIQUE (thread_id, position),

    FOREIGN KEY (thread_id, post_id)
        REFERENCES threads(thread_id, post_id),

    FOREIGN KEY (comment_id, post_id)
        REFERENCES comments(comment_id, post_id)
);

CREATE INDEX posts_split_idx ON posts(original_split);
CREATE INDEX comments_post_idx ON comments(post_id);
CREATE INDEX comments_parent_idx ON comments(parent_comment_id);
CREATE INDEX pairs_post_idx ON pairs(post_id);
CREATE INDEX thread_comments_comment_idx ON thread_comments(comment_id);
