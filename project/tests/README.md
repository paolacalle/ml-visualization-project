# Database data tests

After importing the real metadata release, verify the loaded `cmv` data with:

```bash
python project/tests/test_loaded_database.py -v
```

These seven checks use a read-only transaction and compare database rows with
the published Parquet release: all table totals, source IDs/splits, the large
sample discussion and OP label, its reply tree, paired threads/memberships,
missing-parent placeholders, and a false OP label. The direct command opts into
the configured live database; normal test discovery skips this class unless
`CMV_VERIFY_LIVE_DB=1`. `CMV_METADATA_DATASET` can select another metadata release.

The prepared-data database loaders have offline and opt-in PostgreSQL tests:

```bash
python -m unittest discover -s project/tests -p 'test_db_loaders.py' -v
```

Without `CMV_TEST_DATABASE_DSN`, four dataset tests run and PostgreSQL tests skip.
With that environment variable configured, 14 additional integration tests use
real PostgreSQL COPY, private test schemas, and forced rollback. They test all
seven loaders and `ParquetDatabaseLoader` without changing production `cmv`
data. The configured role needs permission to create schemas. See the
[loader instructions](../database/db_loaders/README.md) for usage and merge rules.

The optional mapping-worker path is tested with:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_multiprocessing.py' -v
```

These tests run the runner contract with two spawned processes, compare all seven
tables against sequential output, check bounded submission and successful-prefix
preservation after read-ahead errors, and cover CLI Parquet export. SQLite writes
are checked in the parent process. All archives and output files are temporary.

From the repository root, activate your environment and install dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s project/tests -v
```

To run only the metadata post tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_posts.py' -v
```

`test_metadata_posts.py` builds a tiny doubly compressed archive in a temporary
directory, reads/maps three posts, writes Parquet, and compares the restored
values and schema. It also checks source IDs, splits, original line numbers,
missing fields, Unicode, edited flags/timestamps, invalid IDs, and integer limits.
No database connection, raw dataset download, or changes to `load_data.py` are
required. All fixture archives and output Parquet files are temporary.

To run only the metadata comment tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_comments.py' -v
```

`test_metadata_comments.py` checks top-level and nested replies, missing parent
records, ID validation, cross-post ownership, omitted text, and Parquet round
trips with filtered lookups. Missing parent references are preserved; creating
and enriching placeholder rows belongs to the later reconciliation step.
These tests use temporary local files and require no database connection.

To run only the metadata OP-label tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_op_labels.py' -v
```

`test_metadata_op_labels.py` checks supplied true/false labels, exact source IDs,
invalid tasks/IDs, missing or incorrectly typed labels, and Parquet round trips.
An absent label is never converted to false. These tests run offline.

To run only the metadata pair tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_pairs.py' -v
```

`test_metadata_pairs.py` checks stable pair/thread IDs, both outcomes, ordered
memberships, source/line provenance, invalid inputs, and cross-post ownership.
It also writes a tiny compressed archive and round-trips all three output tables
through Parquet. The mapper also returns comment metadata candidates; staging
reconciles them into canonical comment rows.

To run only comment reconciliation tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_staging.py' -v
```

`test_metadata_staging.py` checks the local SQLite `CommentStaging` class:
deduplication, source-order-independent precedence, NULL filling, missing-parent
placeholders and enrichment, ownership/split conflicts, atomic batch rollback,
and bounded batch export to the comment Parquet schema. Staging files are kept
when the class closes so callers can inspect failed exports; these tests own and
clean up their temporary directories. No cloud connection is required.

To run only the pair-comment integration tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_pair_comments.py' -v
```

`test_metadata_pair_comments.py` verifies the pair-task comment adapter, unknown
parent handling, source/line error context, and integration with staging. Shared
comments retain their memberships while producing one canonical row. Tests also
check source-order-independent full-discussion precedence, placeholder enrichment,
and comment/membership Parquet key consistency. All files are temporary and local.

To run only all-table staging and validation tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_tables.py' -v
```

`test_metadata_tables.py` exercises `MetadataStaging` with a validated six-source
snapshot. It checks all seven table schemas, post fallback precedence, post/comment
placeholders, original splits, atomic pair writes and nested rollback, idempotent
reloads, and source-order-independent output. Validation rejects missing pair
outcomes, gaps in membership positions, invalid provenance, and parent cycles.
Every table is also round-tripped through temporary Parquet files. No archive
scan, exporter run, or database connection is involved.

`MetadataStaging.iter_table_batches(table, batch_size=1000, split=None)` yields
only the declared DB columns. Call `validate()` before publishing; its report
separates table row counts from post/comment placeholder counts. `add_pair()`
accepts all four outputs of `map_pair()` as one atomic write. The original
`CommentStaging` interface remains available for focused comment tests.

To run only the archive runner tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_runner.py' -v
```

`test_metadata_runner.py` creates a tiny archive with all six sources and checks
single-pass routing, physical source lines, bounded discussion-comment batches,
source order, and final counts. It covers failed-record rollback, incomplete or
duplicate sources, malformed JSON, validation failures, resource cleanup, and
the command-line entrypoint. It does not scan the real archive. Usage is in the
[metadata preparation instructions](../database/parquet_loaders/README.md).

To run only metadata Parquet export tests:

```bash
python -m unittest discover -s project/tests -p 'test_metadata_parquet.py' -v
```

`test_metadata_parquet.py` runs the tiny six-source archive through staging and
the writer. It checks typed compressed shards, exact source IDs, NULLs,
placeholders, manifest checksums, discussions sharing/spanning shards, and indexed
lookup. It verifies downloaded copies without raw/staging data, refuses existing
output, retains failed `.partial` runs, and detects altered bytes/rows/schemas,
missing index entries, incorrect totals, and unlisted files. The CLI export path
is covered too. The full raw archive and cloud database are not used.
