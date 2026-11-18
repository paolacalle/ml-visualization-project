"""Bounded reads of declared database columns from a verified metadata release."""

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from parquet_loaders.metadata_parquet import _safe_path, validate_metadata_dataset
from parquet_loaders.metadata_staging import TABLE_SCHEMAS


class MetadataDataset:
    def __init__(self, path, *, batch_size=10000):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.path = Path(path).resolve()
        self.batch_size = batch_size
        self.manifest = None
        self.index = None

    def validate(self):
        report = validate_metadata_dataset(self.path, batch_size=self.batch_size)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.index = json.loads((self.path / self.manifest["discussion_index"]["path"]).read_text(encoding="utf-8"))
        return report

    def expected_counts(self, post_id=None):
        if self.manifest is None:
            raise RuntimeError("Validate the release before reading rows.")
        if post_id is None:
            return dict(self.manifest["validation"]["tables"])
        if post_id not in self.index:
            raise ValueError(f"Post is not in this metadata release: {post_id}")
        return {"sources": 6, **self.index[post_id]["rows"]}

    def iter_rows(self, table, *, post_id=None):
        expected = self.expected_counts(post_id)
        if table not in TABLE_SCHEMAS:
            raise ValueError(f"Unknown metadata table: {table}")
        if post_id is None or table == "sources":
            paths = [info["path"] for info in self.manifest["files"] if info["table"] == table]
        else:
            paths = self.index[post_id]["files"][table]
        seen = 0
        for relative in paths:
            with pq.ParquetFile(_safe_path(self.path, relative)) as parquet:
                for batch in parquet.iter_batches(batch_size=self.batch_size, columns=TABLE_SCHEMAS[table].names):
                    if post_id is not None and table != "sources":
                        batch = batch.filter(pc.equal(batch.column("post_id"), post_id))
                    for row in batch.to_pylist():
                        seen += 1
                        yield row
        if seen != expected[table]:
            raise ValueError(f"Unexpected {table} row count: {seen}, expected {expected[table]}")
