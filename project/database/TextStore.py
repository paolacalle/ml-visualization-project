import json
from pathlib import Path
import pyarrow.parquet as pq


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
        
if __name__ == "__main__":
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