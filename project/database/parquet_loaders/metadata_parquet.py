"""Write and independently verify versioned metadata Parquet releases."""

import hashlib
import json
from collections import Counter
from contextlib import closing
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from .metadata_staging import TABLE_KEYS, TABLE_SCHEMAS
from .source_map import map_sources


DISCUSSION_TABLES = tuple(name for name in TABLE_SCHEMAS if name != "sources")


def _checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest_batch(digest, batch):
    for row in batch.to_pylist():
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")


def _schema_description(schema):
    return [{"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema]


def _new_location(split):
    return {"split": split, "files": {name: [] for name in DISCUSSION_TABLES},
            "rows": {name: 0 for name in DISCUSSION_TABLES}}


def _index_file(index, name, split, relative, counts):
    for post_id, count in counts.items():
        if name == "posts":
            if post_id in index or count != 1:
                raise ValueError(f"Duplicate post ID in Parquet: {post_id}")
            index[post_id] = _new_location(split)
        if post_id not in index or index[post_id]["split"] != split:
            raise ValueError(f"Unknown post or conflicting split in {relative}: {post_id}")
        index[post_id]["files"][name].append(relative)
        index[post_id]["rows"][name] += count


def _sort_index(index):
    for location in index.values():
        for paths in location["files"].values():
            paths.sort()


def _safe_path(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Manifest paths must be strings.")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid manifest path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Manifest path escapes the dataset: {relative}")
    return result


class MetadataParquetWriter:
    """Write all seven metadata tables, an index, and a provenance manifest.

    The caller supplies open ``MetadataStaging``. A new ``.partial`` sibling
    directory holds candidate files until offline verification succeeds; then
    it is renamed to the final release directory. The writer owns output files,
    but does not close staging, reparse the archive, or contact PostgreSQL.
    """

    def __init__(self, output_path, *, rows_per_file=50000, batch_size=1000):
        """Store a new release path and positive shard/read-batch row limits.

        One shard plus an incoming batch and the discussion index are kept in
        memory. Files use Zstandard level 9; source rows share one global file.
        This constructor creates no directories.
        """
        for name, value in (("rows_per_file", rows_per_file), ("batch_size", batch_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.output_path = Path(output_path).resolve()
        self.partial_path = self.output_path.with_name(self.output_path.name + ".partial")
        self.rows_per_file = rows_per_file
        self.batch_size = batch_size

    def check_destination(self):
        """Raise ``FileExistsError`` if the release or its partial sibling exists."""
        for path in (self.output_path, self.partial_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {path}; choose a new output path.")

    def _write_file(self, name, split, part, buffers, files, index):
        """Sort/write a shard, record byte/row hashes, and update discussion pointers."""
        schema = TABLE_SCHEMAS[name]
        table = pa.concat_tables(buffers) if buffers else pa.Table.from_pylist([], schema=schema)
        sorting = list(TABLE_KEYS[name])
        if "post_id" in schema.names:
            sorting = ["post_id", *[key for key in sorting if key != "post_id"]]
        if name == "thread_comments":
            sorting = ["post_id", "thread_id", "position"]
        table = table.sort_by([(key, "ascending") for key in sorting])
        relative = ("sources.parquet" if name == "sources" else
                    f"{name}/split={split}/part-{part:04d}.parquet")
        path = self.partial_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        logical = hashlib.sha256()
        for batch in table.to_batches(max_chunksize=self.batch_size):
            _digest_batch(logical, batch)
        pq.write_table(table, path, compression="zstd", compression_level=9,
                       row_group_size=4096, write_statistics=True)
        files.append({"table": name, "split": split, "path": relative,
                      "rows": table.num_rows, "bytes": path.stat().st_size,
                      "sha256": _checksum(path), "rows_sha256": logical.hexdigest()})
        if name != "sources":
            _index_file(index, name, split, relative, Counter(table["post_id"].to_pylist()))

    def _write_partition(self, stage, name, split, files, index):
        """Flush bounded shards for a table/split, including an explicit empty file."""
        buffers, buffered_rows, part = [], 0, 0
        # Only this shard and one incoming batch are held in memory.
        with closing(stage.iter_table_batches(name, batch_size=self.batch_size, split=split)) as batches:
            for rows in batches:
                incoming = pa.Table.from_pylist(rows, schema=TABLE_SCHEMAS[name])
                offset = 0
                while offset < incoming.num_rows:
                    count = min(incoming.num_rows - offset, self.rows_per_file - buffered_rows)
                    buffers.append(incoming.slice(offset, count))
                    offset += count
                    buffered_rows += count
                    if buffered_rows == self.rows_per_file:
                        self._write_file(name, split, part, buffers, files, index)
                        buffers, buffered_rows = [], 0
                        part += 1
        # Keep an explicit-schema empty file for an empty partition.
        if buffers or part == 0:
            self._write_file(name, split, part, buffers, files, index)

    def write(self, stage, *, archive_path, input_by_source):
        """Validate staging, export every table, verify the release, and publish.

        Args:
            stage: Open ``MetadataStaging`` with all mapped records present.
            archive_path: Existing raw archive, read for its provenance checksum.
            input_by_source: All six member names mapped to nonnegative
                ``records`` and ``comments`` occurrence counts from the runner.

        Return output path, validation totals, file/byte counts, and discussion
        count. A failed write/verification leaves ``.partial`` for inspection and
        never publishes it. Existing release/partial directories are refused.
        """
        self.check_destination()
        archive = Path(archive_path).resolve()
        if not archive.is_file():
            raise FileNotFoundError(archive)
        validation = stage.validate()
        sources = [row for batch in stage.iter_table_batches("sources") for row in batch]
        source_map = {row["archive_member"]: {key: row[key] for key in
                      ("source_id", "task", "original_split")} for row in sources}
        map_sources(source_map)
        if set(input_by_source) != set(source_map):
            raise ValueError("Input counts must describe all six source members.")
        for counts in input_by_source.values():
            if (set(counts) != {"records", "comments"}
                    or any(type(value) is not int or value < 0 for value in counts.values())):
                raise ValueError("Input counts must be nonnegative record/comment integers.")

        self.partial_path.mkdir(parents=True)
        files, index = [], {}
        # The six-row source snapshot is always a single global file.
        self._write_file("sources", None, 0,
                         [pa.Table.from_pylist(sources, schema=TABLE_SCHEMAS["sources"])], files, index)
        for name in DISCUSSION_TABLES:
            for split in ("train", "heldout"):
                self._write_partition(stage, name, split, files, index)
        _sort_index(index)
        index_path = self.partial_path / "discussion_index.json"
        index_path.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")) + "\n", encoding="utf-8")
        manifest = {
            "format": "cmv-metadata", "format_version": 1, "version": self.output_path.name,
            "pyarrow_version": pa.__version__,
            "compression": {"codec": "zstd", "level": 9}, "row_group_size": 4096,
            "raw_archive": {"name": archive.name, "bytes": archive.stat().st_size,
                            "sha256": _checksum(archive)},
            "source_map": source_map, "input_by_source": input_by_source,
            "schemas": {name: _schema_description(schema) for name, schema in TABLE_SCHEMAS.items()},
            "primary_keys": {name: list(keys) for name, keys in TABLE_KEYS.items()},
            "validation": validation, "files": files,
            "discussion_index": {"path": index_path.name, "bytes": index_path.stat().st_size,
                                 "sha256": _checksum(index_path)},
        }
        (self.partial_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verified = validate_metadata_dataset(self.partial_path, batch_size=self.batch_size)
        if self.output_path.exists():
            raise FileExistsError(self.output_path)
        self.partial_path.rename(self.output_path)
        return {"output_path": str(self.output_path), "validation": validation, **verified}


def validate_metadata_dataset(root, *, batch_size=1000):
    """Verify all files and rebuild the discussion index without staging/raw data.

    Logical row hashes originate from staging before each file is written. They
    check exact round trips; stage.validate() supplies the relational validation.
    This function independently checks the published bytes, schemas, counts,
    source snapshot, original splits, placeholders, and complete file index.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "cmv-metadata" or manifest.get("format_version") != 1:
        raise ValueError("Unsupported metadata manifest format.")
    expected_schemas = {name: _schema_description(schema) for name, schema in TABLE_SCHEMAS.items()}
    if manifest["schemas"] != expected_schemas or manifest["primary_keys"] != {
            name: list(keys) for name, keys in TABLE_KEYS.items()}:
        raise ValueError("Manifest schemas/primary keys do not match the metadata contract.")
    map_sources(manifest["source_map"])
    index_info = manifest["discussion_index"]
    index_path = _safe_path(root, index_info["path"])
    if index_path.name != "discussion_index.json":
        raise ValueError("Unexpected discussion index path.")
    if index_path.stat().st_size != index_info["bytes"] or _checksum(index_path) != index_info["sha256"]:
        raise ValueError("Discussion index checksum/size mismatch.")
    expected_index = json.loads(index_path.read_text(encoding="utf-8"))
    index, paths, sources = {}, set(), []
    totals = Counter({name: 0 for name in TABLE_SCHEMAS})
    placeholders = Counter({"posts": 0, "comments": 0})
    files = manifest["files"]
    # Post locations must exist before child tables are indexed.
    order = {name: position for position, name in enumerate(TABLE_SCHEMAS)}
    if any(file.get("table") not in order for file in files):
        raise ValueError("Unknown table in manifest files.")
    for info in sorted(files, key=lambda file: (order[file["table"]], file["path"])):
        name, relative, split = info["table"], info["path"], info["split"]
        path = _safe_path(root, relative)
        if relative in paths:
            raise ValueError(f"Duplicate manifest file: {relative}")
        paths.add(relative)
        if name == "sources":
            if relative != "sources.parquet" or split is not None:
                raise ValueError("Invalid sources file location.")
        elif (split not in ("train", "heldout")
              or not relative.startswith(f"{name}/split={split}/") or path.suffix != ".parquet"):
            raise ValueError(f"Invalid table/split location: {relative}")
        if path.stat().st_size != info["bytes"] or _checksum(path) != info["sha256"]:
            raise ValueError(f"Parquet checksum/size mismatch: {relative}")
        with pq.ParquetFile(path) as parquet:
            if parquet.schema_arrow != TABLE_SCHEMAS[name] or parquet.metadata.num_rows != info["rows"]:
                raise ValueError(f"Parquet schema/count mismatch: {relative}")
            logical, counts = hashlib.sha256(), Counter()
            rows = 0
            for batch in parquet.iter_batches(batch_size=batch_size):
                _digest_batch(logical, batch)
                rows += batch.num_rows
                if name == "sources":
                    sources.extend(batch.to_pylist())
                else:
                    counts.update(batch.column("post_id").to_pylist())
                    if name == "posts" and any(value != split for value in batch.column("original_split").to_pylist()):
                        raise ValueError(f"Post split mismatch: {relative}")
                    if name in placeholders:
                        placeholders[name] += batch.column("content_available").to_pylist().count(False)
            if logical.hexdigest() != info["rows_sha256"] or rows != info["rows"]:
                raise ValueError(f"Parquet row round-trip mismatch: {relative}")
        totals[name] += rows
        if name != "sources":
            _index_file(index, name, split, relative, counts)

    if {path.relative_to(root).as_posix() for path in root.rglob("*.parquet")} != paths:
        raise ValueError("Parquet files do not match the manifest inventory.")
    source_map = {row["archive_member"]: {key: row[key] for key in
                  ("source_id", "task", "original_split")} for row in sources}
    if len(sources) != 6 or source_map != manifest["source_map"]:
        raise ValueError("Exported source IDs do not match the manifest snapshot.")
    _sort_index(index)
    if index != expected_index:
        raise ValueError("Discussion index does not match actual Parquet rows/files.")
    if dict(totals) != manifest["validation"]["tables"]:
        raise ValueError("Parquet table totals do not match validated staging.")
    if dict(placeholders) != manifest["validation"]["placeholders"]:
        raise ValueError("Parquet placeholder totals do not match validated staging.")
    return {"parquet_files": len(paths), "parquet_bytes": sum(file["bytes"] for file in files),
            "discussions": len(index)}
