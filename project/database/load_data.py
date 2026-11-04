import argparse
from pathlib import Path
import json
from SupabaseConnection import Database
from parquet_loaders.source_map import snapshot_sources
from contextlib import closing
from ArchiveReader import ArchiveReader
from parquet_loaders.posts import map_post

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
        # upload all sources to DB 
        
        # load all of sources in DB
        source_map = snapshot_sources(
            db,
            PROJECT_DIR / "data/processed/cmv-metadata/source_map.json",
        )
        
        # load the reader 
        reader = ArchiveReader(
            archive_path=ARCHIVE_PATH,
            source_map=source_map
            
        )
        
        # auto closes the iterator once all records are processed
        with closing(reader.iter_records(task="all")) as records:
            member_name, line_num, record = next(records)
            row = map_post(
                record=record,
                source=source_map[member_name]
            )
            
            print(f"Source: {source_map[member_name]}, Line#: {line_num}")
            print(json.dumps(row, indent=2, ensure_ascii=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()