import json
from pathlib import Path
import pyarrow.parquet as pq
from SupabaseConnection import Database


class TextStore:
    """Read versioned post and comment text from local Parquet files."""

    def __init__(self, dataset_dir, expected_version="v1"):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()

        manifest_path = self.dataset_dir / "manifest.json"
        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        self.version = self.manifest["version"]

        if self.version != expected_version:
            raise ValueError(
                f"Expected dataset {expected_version}, "
                f"but found {self.version}."
            )

        index_path = (
            self.dataset_dir
            / self.manifest["discussion_index"]["path"]
        )
        self.index = json.loads(
            index_path.read_text(encoding="utf-8")
        )
    
    def get_post(self, post_id):
        location = self.index.get(post_id)
        
        if location is None: 
            raise KeyError(f"Unknown post ID: {post_id}")
        
        table = pq.read_table(
            self.dataset_dir / location["post_file"],
            columns=["post_id", "title", "body"],
            filters=[("post_id", "=", post_id)],
        )
        
        if table.num_rows != 1:
            raise ValueError(
                f"Expected one post for {post_id}, "
                f"found {table.num_rows}."
            )
            
        return table.to_pylist()[0]
    
    def get_comments(self, post_id):
        location = self.index.get(post_id)
        
        if location is None:
            raise KeyError(f"Unknown post ID: {post_id}")
        
        if location["comment_file"] is None:
            return []
        
        table = pq.read_table(
            self.dataset_dir / location["comment_file"],
            columns=["comment_id", "post_id", "body"],
            filters=[("post_id", "=", post_id)],
        )
        
        if table.num_rows != location["comment_count"]:
            raise ValueError(f"Comment count mismatch for {post_id}")

        return table.to_pylist()
    
    def attach_comment_text(self, rows):
        rows = list(rows)
        files = {}
        
        # group requests so each file is read once
        for row in rows:
            post_id = row["post_id"]
            comment_id = row["comment_id"]

            location = self.index.get(post_id)

            if location is None:
                raise KeyError(f"Unknown post ID: {post_id}")

            filename = location["comment_file"]

            if filename is None:
                raise KeyError(f"No comment text available for {post_id}")

            files.setdefault(filename, set()).add(comment_id)
            
        # load the bodies needed
        bodies = {}

        for filename, comment_ids in files.items():
            table = pq.read_table(
                self.dataset_dir / filename,
                columns=["comment_id", "post_id", "body"],
                filters=[("comment_id", "in", list(comment_ids))],
            )

            for record in table.to_pylist():
                key = (record["post_id"], record["comment_id"])
                bodies[key] = record["body"]
                
        # enrich the data
        # preserve the input order and duplicate requests
        enriched = []
        
        for row in rows:
            key = (row["post_id"], row["comment_id"])

            if key not in bodies:
                raise KeyError(f"Comment text not found: {key}")

            enriched.append({**row, "body": bodies[key]})

        return enriched

        
if __name__ == "__main__":
    """Example of how to use this class."""
    project_dir = Path(__file__).resolve().parents[1]
    
    print(project_dir)

    store = TextStore(
        project_dir / "data/processed/cmv-text/v1"
    )
    

    print("Dataset version:", store.version)
    print("Posts indexed:", len(store.index))
    print("Sample location:", store.index["t3_37i8f6"])
    
    post = store.get_post("t3_37i8f6")

    print("Post:", post["post_id"])
    print("Title:", post["title"])
    print("Body preview:", (post["body"] or "")[:200])
    
    comments = store.get_comments("t3_37i8f6")

    print("Comments:", len(comments))  # expect: 1367

    if comments:
        print(json.dumps(comments[0], indent=2, ensure_ascii=False))
        
    db = Database()

    try:
        result = db.execute(
            """
            SELECT comment_id, post_id, score, parent_comment_id
            FROM cmv.comments
            WHERE post_id = %s AND content_available = TRUE
            ORDER BY created_utc, comment_id
            LIMIT 5;
            """,
            ("t3_37i8f6",),
        )

        columns = (
            "comment_id",
            "post_id",
            "score",
            "parent_comment_id",
        )
        rows = [dict(zip(columns, row)) for row in result]

        enriched = store.attach_comment_text(rows)

        for row in enriched:
            print(row["comment_id"], "score:", row["score"])
            print((row["body"] or "")[:3])
            print()
    finally:
        db.close()