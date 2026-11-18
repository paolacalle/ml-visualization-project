"""Manual setup helpers; direct execution creates the database schema.

Use ``MetadataRunner.py`` for raw-to-Parquet preparation and
``ParquetDatabaseLoader.py`` for imports. Running this file is a schema-creation
operation and is not the normal way to reload or query an existing database.
"""

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


def add_snap_shots():
    """Cache/read source IDs, then print the first mapped discussion post.

    Opens the configured PostgreSQL connection and reuses a saved snapshot when
    present. It reads the raw archive for a sample and does not import its data.
    It is available for manual use but is not the script's default entrypoint.
    """
    
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


def create_schema():
    """Connect and execute ``queries/create_schema.sql`` in a transaction.

    Destination tables must not already exist. This does not drop old tables or
    import data. The connection is closed on success or error.
    """

    db = Database()

    try:
        db.execute("create_schema.sql")
    
    finally:
        db.close()

if __name__ == "__main__":
    create_schema()
