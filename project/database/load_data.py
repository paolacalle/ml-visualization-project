import argparse
from pathlib import Path
import json
from SupabaseConnection import Database
from parquet_loaders.source_map import snapshot_sources


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_DIR / "data" / "raw" / "cmv.tar.bz2"


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--migrate-posts",
    #     action="store_true",
    #     help="Backfill new post columns and remove extra_json; run once.",
    # )
    # args = parser.parse_args()

    db = Database()

    try:
        source_map = snapshot_sources(
            db,
            PROJECT_DIR / "data/processed/cmv-metadata/source_map.json",
        )

        print(json.dumps(source_map, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()