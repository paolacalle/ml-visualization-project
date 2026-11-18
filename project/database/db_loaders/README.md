# Loading prepared metadata into PostgreSQL

For a quick overview and runnable commands, see the
[runner guide](../guides/RUNNERS.md#import-metadata-into-postgresql).

`ParquetDatabaseLoader` reads the published metadata release and calls seven
table-specific loaders in dependency order:

```text
SourcesLoader → PostsLoader → CommentsLoader → OpLabelsLoader
             → PairsLoader → ThreadsLoader → ThreadCommentsLoader
```

Each loader accepts an iterable of mapped dictionaries through `.load(rows)`.
The shared helper streams those rows through psycopg `COPY FROM STDIN` into a
private PostgreSQL temporary table, then merges on the server. It does not issue
one INSERT round trip per row or read `cmv.tar.bz2`.

## Check and load

Run these from the repository root with the existing `.env` connection settings.
The destination tables must already exist. Schema creation/cleanup is separate;
the loader does not drop tables, delete data, or remove old text columns.

Read-only preflight for the real release and current database:

```bash
python project/database/ParquetDatabaseLoader.py \
  --dataset project/data/processed/cmv-metadata/v1 \
  --check
```

Load one complete discussion first:

```bash
python project/database/ParquetDatabaseLoader.py \
  --dataset project/data/processed/cmv-metadata/v1 \
  --post-id t3_37i8f6
```

Load all prepared data by omitting the post filter:

```bash
python project/database/ParquetDatabaseLoader.py \
  --dataset project/data/processed/cmv-metadata/v1 \
  --batch-size 10000
```

The post filter uses the discussion index to choose files, then filters batches
within shared shards. All six sources are checked/loaded even for one discussion.
`batch_size` bounds local Parquet reads; each table's rows are streamed to one
temporary table before merging. Comment parents may appear in later shards.

## Class interface

From a script under `project/database`, run at the repository root:

```python
from SupabaseConnection import Database
from ParquetDatabaseLoader import ParquetDatabaseLoader

db = Database()
try:
    loader = ParquetDatabaseLoader(
        db,
        "project/data/processed/cmv-metadata/v1",
        batch_size=10000,
    )
    print(loader.check(post_id="t3_37i8f6"))  # Read-only.
    print(loader.load(post_id="t3_37i8f6"))
finally:
    db.close()
```

The class also accepts a psycopg connection directly. Tests can select a separate
destination with `schema=...`. The caller owns connection closing.

## Merge and transaction rules

- Validate the release's checksums, schemas, row counts, and index before copying.
- Verify required database columns/types and primary keys. Extra nullable columns
  or columns with defaults are allowed and left untouched.
- Preserve snapshot source IDs. A conflicting ID/archive-member mapping fails;
  it is never silently reassigned. New source IDs use identity override, and the
  identity sequence is advanced transactionally without rewinding it.
- Preserve post splits and comment ownership. Conflicting parent references fail.
- Available post/comment metadata can update older base metadata and enrich
  placeholders. An incoming unavailable placeholder cannot replace an existing
  available record. Identical rows cause no UPDATE.
- Labels, pairs, threads, and memberships are immutable for their existing keys;
  conflicting values fail instead of being silently skipped.
- The complete import is one transaction. A failure rolls back earlier table
  writes too. Comments are merged with deferred parent checks and checked before
  importing dependent memberships. Foreign keys stay enabled.
- If the caller already owns an outer transaction, the import is a savepoint;
  the caller remains responsible for committing that outer transaction.

Reports give processed/inserted/updated/unchanged counts for every table, plus
total destination row counts. Unrelated existing rows are preserved, so those
destination totals can exceed the selected release or discussion counts.

The older `load_posts`, `load_comments`, `load_sources`, and `load_op_label`
functions remain for compatibility. New imports should use the classes above.

## Verification

Offline reader/integrity tests:

```bash
python -m unittest discover -s project/tests -p 'test_db_loaders.py' -v
```

To include PostgreSQL integration tests, supply `CMV_TEST_DATABASE_DSN` through
your local environment or secret manager. Do not put credentials in source files.
The same command then creates unique `cmv_loader_test_...` schemas inside
forced-rollback transactions. It never loads test data into `cmv`, and its
connection requires permission to create schemas. Tests cover each real COPY
loader, idempotence, cross-shard parents, source identities, immutable conflicts,
placeholder enrichment, one-discussion/full orchestration, and all-table rollback.

The real metadata release passed read-only preflight. The 14 PostgreSQL
integration tests passed against isolated rolled-back schemas; production `cmv`
row counts were checked afterwards and remained unchanged.

After the real import, run its seven read-only verification cases with:

```bash
python project/tests/test_loaded_database.py -v
```

These compare real table totals and selected discussions against Parquet, and
check labels, source IDs, parent links, missing-parent placeholders, pair outcomes,
and ordered memberships. They do not insert, update, or delete rows.
