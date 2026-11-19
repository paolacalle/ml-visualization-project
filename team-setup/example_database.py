"""Read-only starter: verify SQL access and fetch metadata for one discussion."""

import argparse
import json

from _common import load_environment
from cmv_data import Database


def query_comments(db, post_id, limit=5):
    """Return available comments with IDs needed by TextStore, in stable order."""
    rows = db.execute("""
        SELECT comment_id, post_id, parent_comment_id, author, score
        FROM cmv.comments
        WHERE post_id = %s AND content_available
        ORDER BY comment_id
        LIMIT %s;
    """, (post_id, limit))
    columns = ("comment_id", "post_id", "parent_comment_id", "author", "score")
    return [dict(zip(columns, row)) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-id", default="t3_37i8f6")
    args = parser.parse_args()
    load_environment()
    db = Database()
    try:
        with db.connection.transaction():
            db.connection.execute("SET TRANSACTION READ ONLY")
            print("Connection check:", db.execute("SELECT 1;")[0][0])
            posts = db.execute("""
                SELECT post_id, original_split, title
                FROM cmv.posts WHERE post_id = %s;
            """, (args.post_id,))
            if not posts:
                raise ValueError(f"Post not found in the database: {args.post_id}")
            print("Post:", posts[0])
            print(json.dumps(query_comments(db, args.post_id), indent=2, ensure_ascii=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
