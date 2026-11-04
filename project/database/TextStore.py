import json
import hashlib
from collections import Counter
from pathlib import Path
import pyarrow as pa
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

    def _read_table(self, filename, columns, filters):
        path = self.dataset_dir / filename

        if not path.is_file():
            raise FileNotFoundError(f"Missing Parquet file: {path}")

        try:
            return pq.read_table(path, columns=columns, filters=filters)
        except (pa.ArrowInvalid, OSError) as exc:
            raise ValueError(f"Cannot read Parquet file {path}: {exc}") from exc

    @staticmethod
    def _verify_file(path, expected):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")

        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"File size mismatch: {path}")

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)

        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")

    def validate(self):
        """Check the complete local release; normal lookups do not hash files."""
        index_info = self.manifest["discussion_index"]
        index_path = self.dataset_dir / index_info["path"]
        self._verify_file(index_path, index_info)
        disk_index = json.loads(index_path.read_text(encoding="utf-8"))
        if disk_index != self.index:
            raise ValueError("In-memory discussion index differs from the release index")

        files = {}
        totals = Counter()
        for entry in self.manifest["files"]:
            filename = entry["path"]
            if filename in files:
                raise ValueError(f"Duplicate file in manifest: {filename}")
            files[filename] = entry
            path = self.dataset_dir / filename
            self._verify_file(path, entry)
            try:
                metadata = pq.read_metadata(path)
                schema = pq.read_schema(path)
            except (pa.ArrowInvalid, OSError) as exc:
                raise ValueError(f"Cannot read Parquet file {path}: {exc}") from exc

            if metadata.num_rows != entry["rows"]:
                raise ValueError(f"Parquet row count mismatch: {path}")

            kind = Path(filename).parts[0]
            expected_columns = {
                "posts": ["post_id", "title", "body"],
                "comments": ["comment_id", "post_id", "body"],
            }.get(kind)
            if expected_columns is None or schema.names != expected_columns:
                raise ValueError(f"Unexpected Parquet columns: {path}")
            if any(field.type != pa.string() for field in schema):
                raise ValueError(f"Unexpected Parquet column types: {path}")
            totals[kind] += metadata.num_rows

        # Validate every index pointer and reconcile its counts with each file.
        indexed_rows = Counter()
        for post_id, location in self.index.items():
            split = location["split"]
            if split not in {"train", "heldout"}:
                raise ValueError(f"Invalid split in index for {post_id}")

            post_file = location["post_file"]
            if post_file not in files or not post_file.startswith(f"posts/split={split}/"):
                raise ValueError(f"Invalid post file reference for {post_id}: {post_file}")
            indexed_rows[post_file] += 1

            count = location["comment_count"]
            comment_file = location["comment_file"]
            if type(count) is not int or count < 0:
                raise ValueError(f"Invalid comment count for {post_id}")
            if comment_file is None:
                if count != 0:
                    raise ValueError(f"Missing comment file for nonempty post {post_id}")
            else:
                if comment_file not in files or not comment_file.startswith(f"comments/split={split}/"):
                    raise ValueError(f"Invalid comment file reference for {post_id}: {comment_file}")
                indexed_rows[comment_file] += count

        for filename, entry in files.items():
            if indexed_rows[filename] != entry["rows"]:
                raise ValueError(f"Index row count mismatch: {filename}")

        for kind in ("posts", "comments"):
            if totals[kind] != sum(source[kind] for source in self.manifest["sources"]):
                raise ValueError(f"Source/Parquet count mismatch: {kind}")

        return {
            "version": self.version,
            "parquet_files": len(files),
            "post_rows": totals["posts"],
            "comment_rows": totals["comments"],
            "index_posts": len(self.index),
            "index_checksum_verified": True,
        }
    
    def get_post(self, post_id):
        location = self.index.get(post_id)
        
        if location is None: 
            raise KeyError(f"Unknown post ID: {post_id}")
        
        table = self._read_table(
            location["post_file"],
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
            if location["comment_count"] != 0:
                raise ValueError(f"Missing comment file for nonempty post {post_id}")
            return []
        
        table = self._read_table(
            location["comment_file"],
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
            table = self._read_table(
                filename,
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

