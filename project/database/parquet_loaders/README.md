# Metadata preparation

`MetadataRunner` connects `ArchiveReader`, the table mappers, and
`MetadataStaging` in one sequential pass. It uses the saved source snapshot and
requires no database connection.

From the repository root, with requirements installed, run:

```bash
python project/database/MetadataRunner.py \
  --archive project/data/raw/cmv.tar.bz2 \
  --source-map project/data/processed/cmv-metadata/source_map.json \
  --staging project/data/processed/cmv-metadata/staging/run-001.sqlite \
  --output project/data/processed/cmv-metadata/v1 \
  --rows-per-file 50000 \
  --batch-size 1000
```

The staging path, output directory, and output's `.partial` directory must be
new. Parent directories are created if needed. Omit `--output` to prepare and
validate SQLite staging only. Database loading remains a separate step.

To use the class from another script under `project/database`:

```python
from MetadataRunner import MetadataRunner

runner = MetadataRunner(
    archive_path="project/data/raw/cmv.tar.bz2",
    source_map="project/data/processed/cmv-metadata/source_map.json",
    staging_path="project/data/processed/cmv-metadata/staging/run-001.sqlite",
    output_path="project/data/processed/cmv-metadata/v1",
    batch_size=1000,
)
report = runner.run()
print(report["validation"])
```

Both examples assume the current working directory is the repository root.
`source_map` also accepts a snapshot dictionary. Actual database source IDs are
preserved; archive order never assigns them.

## Processing and result

1. Validate the six-source snapshot and create new local staging.
2. Open the compressed archive once and process each JSON record.
3. Map full-discussion posts/comments, OP labels, and pair/thread memberships.
4. Merge repeated metadata and enrich referenced post/comment placeholders.
5. Verify that every expected source member was encountered, then validate
   source/split consistency, relationships, outcomes, positions, and parent trees.
6. If an output path was supplied, write typed, Zstandard-compressed Parquet
   shards, a discussion index, and a manifest into a new `.partial` directory.
7. Verify the files against staging row fingerprints, schemas, checksums,
   canonical counts, placeholder counts, original splits, and index entries.
8. Publish by renaming the verified `.partial` directory. Close staging and
   return status `exported_metadata`, or `validated_staging` in staging-only mode.

The report includes `staging_path`, `input_by_source`, and `validation`. Input
counts describe JSON records and supplied comment occurrences per source;
pair-task occurrences can repeat comments. Validation counts describe canonical
table rows and separately report post/comment placeholders. Reported Reddit
`num_comments` values do not determine how many comment rows are ingested.

The archive reader holds one decoded JSON record at a time. `batch_size` bounds
additional mapped comment batches for full discussions; a pair is mapped as one
unit. Each raw record is atomic, including any task-post fallback metadata.
During export, memory is bounded by one `rows_per_file` shard plus an incoming
batch and the discussion index. The six sources remain in one global file.

If ingestion or validation fails, no successful report is returned. The SQLite
file remains available for inspection, containing earlier successful records.
Choose another new staging path when retrying; resume and overwrite are not
supported. Missing source members and duplicate source members are errors;
present but empty source files have zero input counts.

An export failure retains `.partial` for inspection and leaves the final output
unpublished. Use fresh paths when retrying. The existing text release is separate
and is not modified by metadata export.

## Export layout and lookup

```text
v1/
├── sources.parquet
├── posts/split=train/part-0000.parquet
├── posts/split=heldout/part-0000.parquet
├── comments/split=.../part-....parquet
├── op_labels/split=.../part-....parquet
├── pairs/split=.../part-....parquet
├── threads/split=.../part-....parquet
├── thread_comments/split=.../part-....parquet
├── discussion_index.json
└── manifest.json
```

Each discussion-index entry has its `split`, per-table `files` lists, and
per-table `rows` counts. Every shard containing a discussion is recorded; a
discussion can span multiple files. Empty partitions retain schema-bearing
Parquet files, while their index file lists remain empty.

The manifest records the exact source snapshot, raw archive checksum, input
counts, Arrow field types/nullability, primary keys, compression settings, and
each file's row count, byte count, checksum, and logical row fingerprint.

For example, from a script under `project/database` run at the repository root:

```python
import json
from pathlib import Path
import pyarrow.parquet as pq
from parquet_loaders.metadata_staging import TABLE_SCHEMAS

root = Path("project/data/processed/cmv-metadata/v1")
index = json.loads((root / "discussion_index.json").read_text())
post_id = "t3_37i8f6"
comments = []
for relative in index[post_id]["files"]["comments"]:
    comments.extend(pq.read_table(
        root / relative,
        columns=TABLE_SCHEMAS["comments"].names,
        filters=[("post_id", "=", post_id)],
    ).to_pylist())
```

Select the declared columns explicitly: PyArrow dataset reads can otherwise
include an inferred `split` column from the directory name. To retrieve one
comment, add a `("comment_id", "=", comment_id)` filter alongside the post ID.

`validate_metadata_dataset(root)` in `parquet_loaders.metadata_parquet` verifies
a downloaded release without the raw archive, SQLite staging, or a cloud
connection. It rebuilds the index from every file and checks exact row round
trips against fingerprints created before writing. The relational graph is
validated in staging before export.

## Local verification

```bash
python -m unittest discover -s project/tests -p 'test_metadata_runner.py' -v
python -m unittest discover -s project/tests -p 'test_metadata_parquet.py' -v
```

These tests build tiny archives locally. They do not process the full dataset or
connect to Supabase. See [the test instructions](../../tests/README.md) for the
full suite.
