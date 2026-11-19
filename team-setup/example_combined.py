"""Starter workflow: query shared metadata, then attach local text by IDs."""

import argparse
import json

from _common import TEXT_DIR, load_environment
from example_database import query_comments
from cmv_data import Database
from cmv_data import TextStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-id", default="t3_37i8f6")
    args = parser.parse_args()
    load_environment()
    store = TextStore(TEXT_DIR, expected_version="v1")
    db = Database()
    try:
        with db.connection.transaction():
            db.connection.execute("SET TRANSACTION READ ONLY")
            rows = query_comments(db, args.post_id)
    finally:
        db.close()
    if not rows:
        raise ValueError(f"No available comments in the database for {args.post_id}")
    # SQL chooses rows; TextStore returns copies with body added, preserving order.
    enriched = store.attach_comment_text(rows)
    print("Title:", store.get_post(args.post_id)["title"])
    for row in enriched:
        preview = {**row, "body": (row["body"] or "")[:200]}
        print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
