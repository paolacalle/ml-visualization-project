"""Prepare metadata in one archive pass, optionally mapping with worker processes.

Run this file with ``--help`` for CLI arguments. ``guides/RUNNERS.md`` shows the
extraction command and distinguishes local preparation from PostgreSQL loading.
"""

import argparse
import copy
import json
import multiprocessing
import sqlite3
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from itertools import islice
from pathlib import Path

import pyarrow as pa

from cmv_data.ArchiveReader import ArchiveReader
from cmv_data.parquet_loaders.comments import map_comment
from cmv_data.parquet_loaders.metadata_staging import MetadataStaging
from cmv_data.parquet_loaders.metadata_parquet import MetadataParquetWriter
from cmv_data.parquet_loaders.op_labels import map_op_label
from cmv_data.parquet_loaders.pairs import map_pair
from cmv_data.parquet_loaders.posts import map_post, map_task_post
from cmv_data.parquet_loaders.source_map import map_sources


def _discussion_comments(record, source):
    """Yield mapped flat-list comments, attaching their positions to errors."""
    comments = record.get("comments", [])
    if not isinstance(comments, list):
        raise ValueError("Expected a flat comments list in an all/ record.")
    for position, comment in enumerate(comments):
        if not isinstance(comment, dict):
            raise ValueError(f"Comment position {position}: expected a comment object.")
        try:
            yield map_comment(comment, record["name"], source)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Comment position {position}: {exc}") from exc


def prepare_record(job):
    """Map ``(member, line, decoded_record, source)`` into a picklable result.

    Returns ``(member, line, source, post_row, task_rows)``. The task rows contain
    comments, an OP label, or a pair bundle. Comments are materialized per job.
    Workers receive ordinary data, never an open archive or SQLite connection.
    Parsing/decompression have already happened in the parent process.
    """
    member, line, record, source = job
    try:
        task = source["task"]
        if task == "all":
            post = map_post(record, source)
            data = {"comments": list(_discussion_comments(record, source))}
        elif task == "op_task":
            post = map_task_post(record, source)
            data = {"label": map_op_label(record, source)}
        else:
            post = map_task_post(record, source)
            data = map_pair(record, source, member, line)
        return member, line, source, post, data
    except (ValueError, TypeError, KeyError, pa.ArrowException) as exc:
        raise ValueError(f"{member}, line {line}: {exc}") from exc


class MetadataRunner:
    """Read all six sources once, stage mapped rows, and validate relationships.

    source_map accepts a snapshot dictionary or its JSON file path. Each raw
    record is atomic. Optional workers map records, while the main process alone
    writes SQLite in archive order. Earlier successful records remain in the
    SQLite file if a later record fails. Existing staging files are never
    overwritten or resumed.
    An optional output_path publishes a verified local Parquet release after
    staging. Neither mode uploads data or connects to a cloud database.
    """

    def __init__(self, archive_path, source_map, staging_path, *, batch_size=1000,
                 output_path=None, rows_per_file=50000, workers=1):
        """Configure preparation without opening the archive or creating staging.

        Args:
            archive_path: Existing raw ``cmv.tar.bz2`` path.
            source_map: Six-source snapshot dictionary or its JSON file path;
                actual database source IDs are preserved.
            staging_path: New SQLite file; existing files cannot be resumed.
            batch_size: Positive comment-write/export-read batch limit in rows.
            output_path: Optional new metadata release directory. ``None``
                prepares/validates staging without publishing Parquet.
            rows_per_file: Positive maximum metadata rows per export shard.
                The six-row global source file is written separately.
            workers: Positive mapping-process count; ``1`` runs in the parent.
                Larger values require a caller script's main guard.

        Relative paths resolve against the current working directory. File
        creation and overwrite checks happen when ``run()`` starts.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if type(workers) is not int or workers <= 0:
            raise ValueError("workers must be a positive integer.")
        self.workers = workers
        if isinstance(source_map, dict):
            snapshot = copy.deepcopy(source_map)
        else:
            snapshot = json.loads(Path(source_map).read_text(encoding="utf-8"))
        map_sources(snapshot)
        self.archive_path = Path(archive_path).resolve()
        self.staging_path = Path(staging_path).resolve()
        self.source_map = snapshot
        self.batch_size = batch_size
        if type(rows_per_file) is not int or rows_per_file <= 0:
            raise ValueError("rows_per_file must be a positive integer.")
        self.output_path = None if output_path is None else Path(output_path).resolve()
        self.rows_per_file = rows_per_file

    def _write_prepared(self, stage, prepared):
        """Write one mapped result atomically; return supplied comment occurrences.

        Only the parent process calls this method. Nested staging savepoints
        keep a post, its labels/comments, and any pair relationships together.
        """
        member, line, source, post, data = prepared
        task = source["task"]
        # Nested staging calls use savepoints; nothing from a failed raw record
        # survives, including a task's original-post fallback metadata.
        with stage.transaction():
            stage.add_posts([post], task=task)
            if task == "all":
                comments = iter(data["comments"])
                while batch := list(islice(comments, self.batch_size)):
                    stage.add_comments(batch, task=task, split=source["original_split"])
            elif task == "op_task":
                stage.add_op_labels([data["label"]])
            else:
                stage.add_pair(data)
        return len(data.get("comments", []))

    def _prepared_records(self, records, source_map):
        """Yield mapping results in input order with at most ``2 * workers`` queued.

        Spawned children cannot inherit the open SQLite connection. Read-ahead
        errors are deferred until the successful queued prefix has been written.
        Closing this generator cancels pending work and shuts down the pool.
        """
        if self.workers == 1:
            for member, line, record in records:
                yield prepare_record((member, line, record, source_map[member]))
            return

        # Spawn keeps the already-open SQLite connection out of child processes.
        with ProcessPoolExecutor(max_workers=self.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = deque()
            exhausted, reader_error = False, None
            try:
                while pending or not exhausted:
                    while not exhausted and len(pending) < self.workers * 2:
                        try:
                            member, line, record = next(records)
                        except StopIteration:
                            exhausted = True
                        except Exception as exc:
                            # Commit the successful prefix before surfacing a
                            # read-ahead error, just as sequential ingestion does.
                            exhausted, reader_error = True, exc
                        else:
                            pending.append(pool.submit(
                                prepare_record, (member, line, record, source_map[member])
                            ))
                    if pending:
                        yield pending.popleft().result()
                if reader_error is not None:
                    raise reader_error
            finally:
                for future in pending:
                    future.cancel()

    def run(self):
        """Prepare all recognized sources once, validate, and optionally export.

        Returns:
            A dictionary with ``status``, ``staging_path``, ``input_by_source``,
            and ``validation``. Export mode adds ``export`` and returns status
            ``exported_metadata``; otherwise status is ``validated_staging``.
            Input comments count supplied occurrences, while validation reports
            canonical rows and placeholders separately.

        Raises on missing/duplicate source members, mapping conflicts, invalid
        relationships, or existing output paths. A failed record rolls back;
        earlier staging records remain for inspection. Staging closes on exit.
        Neither mode modifies PostgreSQL or uploads files to Storage.
        """
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        # Take another private copy so caller edits cannot change a running pass.
        source_map = copy.deepcopy(self.source_map)
        map_sources(source_map)
        writer = None
        if self.output_path is not None:
            writer = MetadataParquetWriter(self.output_path, batch_size=self.batch_size,
                                           rows_per_file=self.rows_per_file)
            writer.check_destination()
            
        self.staging_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {member: {"records": 0, "comments": 0} for member in source_map}
        reader = ArchiveReader(self.archive_path, source_map)
        
        progress = {}
        
        with MetadataStaging(self.staging_path, source_map) as stage:

            with closing(reader.iter_records()) as records, closing(
                self._prepared_records(records, source_map)
            ) as prepared_records:
                for prepared in prepared_records:
                    member, line = prepared[:2]
                    try:
                        comments = self._write_prepared(stage, prepared)
                    except (ValueError, TypeError, KeyError, sqlite3.Error, pa.ArrowException) as exc:
                        raise ValueError(f"{member}, line {line}: {exc}") from exc
                    counts[member]["records"] += 1
                    counts[member]["comments"] += comments
                    
                    if not progress.get(member):
                        progress[member] = {"line" : 1, "c" : comments}
                    else: 
                        progress[member]["line"] += 1
                        progress[member]["c"] += comments
                        
                        if progress[member]["line"] % 1000 == 0:
                            print(progress, file=sys.stderr)
                    
                    
            missing = set(source_map) - reader.seen_members
            if missing:
                raise ValueError(f"Missing archive sources: {', '.join(sorted(missing))}")
            export = None
            if writer is None:
                validation = stage.validate()
            else:
                export = writer.write(stage, archive_path=self.archive_path, input_by_source=counts)
                validation = export["validation"]

        report = {
            "status": "validated_staging" if export is None else "exported_metadata",
            "staging_path": str(self.staging_path),
            "input_by_source": counts,
            "validation": validation,
        }
        if export is not None:
            report["export"] = export
        return report


def main():
    """Run the CLI; print periodic progress to stderr and the final JSON to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True,
                        help="New SQLite file; an existing path is refused.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1,
                        help="Mapping processes; 1 runs sequentially. SQLite writes stay in the parent.")
    parser.add_argument("--output", type=Path,
                        help="Optional new directory for a verified Parquet release.")
    parser.add_argument("--rows-per-file", type=int, default=50000)
    args = parser.parse_args()
    runner = MetadataRunner(args.archive, args.source_map, args.staging, batch_size=args.batch_size,
                            output_path=args.output, rows_per_file=args.rows_per_file, workers=args.workers)
    print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
