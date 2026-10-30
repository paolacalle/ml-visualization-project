from pathlib import Path
import tarfile
from SupabaseConnection import Database


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = PROJECT_DIR / "data" / "raw" / "cmv.tar.bz2"


def load_sources(archive_path, db):
    with tarfile.open(archive_path, "r|bz2") as archive:
        for member in archive:
            if not member.isfile():
                continue

            if not member.name.endswith(".jsonlist.bz2"):
                continue

            task = member.name.split("/")[0]
            filename = member.name.rsplit("/", 1)[-1]
            split = filename.split("_")[0]

            result = db.execute(
                "insert_source.sql",
                (member.name, task, split),
            )
            
            if result:
                print(f"Inserted: {member.name}")
            else:
                print(f"Already exists: {member.name}")


if __name__ == "__main__":
    db = Database()

    try:
        load_sources(ARCHIVE_PATH, db)
    finally:
        db.close()