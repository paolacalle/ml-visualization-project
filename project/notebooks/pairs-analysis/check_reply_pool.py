"""Check which archived opening replies are available for rematching one post."""

import argparse
from contextlib import closing
import json
from pathlib import Path

from cmv_data import Database, TextStore
from psycopg.rows import dict_row


REPO_ROOT = Path(__file__).resolve().parents[3]
TEXT_DIR = REPO_ROOT / "project/data/processed/cmv-text/v1"
MIN_OPENING_WORDS = 50
MIN_DISCUSSION_CHALLENGERS = 10

REPLY_POOL_SQL = """
WITH RECURSIVE discussion AS (
    SELECT * FROM cmv.comments WHERE post_id = %(post_id)s
), descendants AS (
    SELECT root.comment_id AS root_id, child.comment_id
    FROM discussion AS root
    JOIN discussion AS child ON child.parent_comment_id = root.comment_id
    WHERE root.original_parent_id = root.post_id
    UNION
    SELECT d.root_id, child.comment_id
    FROM descendants AS d
    JOIN discussion AS child ON child.parent_comment_id = d.comment_id
), branch_sizes AS (
    SELECT root_id, count(*) AS descendant_count
    FROM descendants
    WHERE comment_id <> root_id
    GROUP BY root_id
), discussion_activity AS (
    SELECT max(c.created_utc) FILTER (WHERE lower(c.author) = lower(p.author))
               AS last_op_reply_utc,
           count(DISTINCT lower(c.author)) FILTER (
               WHERE lower(c.author) <> lower(p.author)
                 AND lower(c.author) NOT IN ('', '[deleted]', '[removed]', 'deltabot', 'automoderator')
           ) AS observed_challenger_count
    FROM cmv.posts AS p
    JOIN discussion AS c ON c.post_id = p.post_id
    WHERE p.post_id = %(post_id)s
), pair_openings AS (
    SELECT tc.comment_id,
           jsonb_agg(jsonb_build_object('pair_id', t.pair_id, 'outcome', t.outcome)
                     ORDER BY t.pair_id, t.outcome) AS memberships
    FROM cmv.threads AS t
    JOIN cmv.thread_comments AS tc
      ON tc.thread_id = t.thread_id AND tc.position = 0
    WHERE t.post_id = %(post_id)s
    GROUP BY tc.comment_id
)
SELECT c.post_id, c.comment_id, c.content_available,
       (c.original_parent_id IS NOT DISTINCT FROM c.post_id) AS is_top_level,
       s.task AS source_task,
       COALESCE(b.descendant_count, 0) AS descendant_count,
       c.created_utc < a.last_op_reply_utc AS before_last_op_reply,
       a.observed_challenger_count,
       COALESCE(p.memberships, '[]'::jsonb) AS pair_opening_memberships
FROM discussion AS c
CROSS JOIN discussion_activity AS a
LEFT JOIN cmv.sources AS s ON s.source_id = c.source_id
LEFT JOIN pair_openings AS p ON p.comment_id = c.comment_id
LEFT JOIN branch_sizes AS b ON b.root_id = c.comment_id
WHERE c.original_parent_id = c.post_id OR p.comment_id IS NOT NULL
ORDER BY c.created_utc NULLS LAST, c.comment_id
"""


def summarize(post_id, rows, store):
    """Check coverage from one Parquet lookup and report IDs without reply text."""
    bodies = {row["comment_id"]: row["body"] for row in store.get_comments(post_id)}
    top_level_ids = {row["comment_id"] for row in rows if row["is_top_level"]}
    paired_ids = {row["comment_id"] for row in rows if row["pair_opening_memberships"]}
    usable_ids = {
        row["comment_id"] for row in rows
        if row["content_available"]
        and (bodies.get(row["comment_id"]) or "").strip().lower()
        not in {"", "[deleted]", "[removed]"}
    }
    extra_ids = (top_level_ids - paired_ids) & usable_ids
    reply_counts = []
    for row in rows:
        if not row["is_top_level"]:
            continue
        comment_id = row["comment_id"]
        word_count = len(bodies[comment_id].split()) if comment_id in usable_ids else None
        reply_counts.append({
            "comment_id": comment_id,
            "descendant_count": row["descendant_count"],
            "opening_word_count": word_count,
            "meets_minimum_words": word_count is not None and word_count >= MIN_OPENING_WORDS,
            "before_last_op_reply": row["before_last_op_reply"],
        })
    challenger_count = rows[0]["observed_challenger_count"] if rows else None
    return {
        "post_id": post_id,
        "observed_challenger_count": challenger_count,
        "meets_minimum_challengers": (
            challenger_count is not None and challenger_count >= MIN_DISCUSSION_CHALLENGERS
        ),
        "top_level_reply_counts": reply_counts,
        "top_level_replies_in_db": len(top_level_ids),
        "top_level_replies_with_text": len(top_level_ids & usable_ids),
        "top_level_replies_without_usable_text": sorted(top_level_ids - usable_ids),
        "matched_openings": [
            {"comment_id": row["comment_id"],
             "memberships": row["pair_opening_memberships"],
             "is_top_level": row["is_top_level"],
             "has_usable_text": row["comment_id"] in usable_ids}
            for row in rows if row["pair_opening_memberships"]
        ],
        "unpaired_top_level_replies_with_text": len(extra_ids),
        "unpaired_replies_from_all_source": sum(
            row["source_task"] == "all" and row["comment_id"] in extra_ids for row in rows
        ),
        "example_unpaired_reply_ids": sorted(extra_ids)[:5],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("post_id", help="Reddit post ID including the t3_ prefix")
    args = parser.parse_args()

    with closing(Database(env_file=REPO_ROOT / ".env")) as db:
        with db.connection.transaction(), db.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(REPLY_POOL_SQL, {"post_id": args.post_id})
            rows = cursor.fetchall()

    report = summarize(args.post_id, rows, TextStore(TEXT_DIR, expected_version="v1"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
