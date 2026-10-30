import argparse
from pathlib import Path

from SupabaseConnection import Database
from loaders import op_labels


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
        op_labels.load_op_label(
            ARCHIVE_PATH,
            db,
            post_id="t3_37i8f6",
        )

        print(db.execute(
            """
            SELECT post_id, delta_label, source_id
            FROM cmv.op_labels
            WHERE post_id = %s;
            """,
            ("t3_37i8f6",),
        ))
    finally:
        db.close()


if __name__ == "__main__":
    main()