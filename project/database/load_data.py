import argparse
from pathlib import Path

from SupabaseConnection import Database
from loaders import posts


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_DIR / "data" / "raw" / "cmv.tar.bz2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--migrate-posts",
        action="store_true",
        help="Backfill new post columns and remove extra_json; run once.",
    )
    args = parser.parse_args()

    db = Database()

    try:
        print("Before:", db.execute("posts_size.sql"))

        if args.migrate_posts:
            db.execute("update_posts.sql")
            print("Posts migration completed.")

        # Keep the archive sample small while reviewing the schema.
        posts.load_posts(
            ARCHIVE_PATH,
            db,
            limit_per_source=2,
        )

        rows = db.execute("sample_posts.sql")

        for row in rows:
            post_id, split, title = row[:3]
            print(f"\n{post_id} | {split}\n{title}")

        print("\nAfter:", db.execute("posts_size.sql"))
    finally:
        db.close()


if __name__ == "__main__":
    main()