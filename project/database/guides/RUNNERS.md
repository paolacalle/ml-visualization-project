# Runners: preparation, import, and verification

Commands run from the repository root. Preparation uses local files; database
commands use the existing `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and
`DB_PASSWORD` environment settings loaded by `SupabaseConnection.Database`.

## Prepare metadata

Logic: `MetadataRunner` reads each source once, maps records, reconciles duplicate
metadata and missing references in SQLite, validates the graph, and optionally
publishes metadata Parquet through `MetadataParquetWriter`.

Choose new staging and output paths. The sample names below are alternatives to
the already-published `v1`; use other names if these also exist.

```bash
./project/venv/bin/python project/database/MetadataRunner.py \
  --archive project/data/raw/cmv.tar.bz2 \
  --source-map project/data/processed/cmv-metadata/source_map.json \
  --staging project/data/processed/cmv-metadata/staging/run-doc-example.sqlite \
  --output project/data/processed/cmv-metadata/v2 \
  --workers 4 --batch-size 5000 --rows-per-file 100000
```

| Argument | Meaning |
| --- | --- |
| `--source-map` | Saved mapping of all six members to actual database IDs |
| `--staging` | New local SQLite file; interrupted runs are retained, not resumed |
| `--output` | New release directory; omit for staging/validation only |
| `--workers` | Mapping processes; default 1, with at most twice this count queued |
| `--batch-size` | Comment staging/export-read batch rows, independent of workers |
| `--rows-per-file` | Metadata shard row cap; sources use one global file |

Decompression and JSON parsing remain in the main reader. Mapping may run in
spawned processes; SQLite writes remain in the parent and follow archive order.
Keep `if __name__ == "__main__":` around class-based runner calls in a script.

Output is built in `.partial`, checked, then renamed to the release directory.
Progress goes to stderr; the final JSON reports input counts, canonical table
counts, and placeholders. The writer has no separate CLI: use the runner's
`--output`, or call `writer.write(open_stage, ...)` from Python when managing
staging yourself. [Detailed preparation guide](../parquet_loaders/README.md).

## Export original text

The separate text exporter streams only the two full-discussion sources. It
keeps each discussion's comment text together and publishes its own index and
manifest. It does not connect to PostgreSQL.

```bash
./project/venv/bin/python project/database/parquet_loaders/export_text_parquet.py \
  --archive project/data/raw/cmv.tar.bz2 \
  --output project/data/processed/cmv-text/v2 \
  --verify-post t3_37i8f6
```

Use an unused version directory. Teammates using the existing text release only
need its one-time download and `TextStore`. [Text-export guide](PARQUET.md).

## Import metadata into PostgreSQL

Logic: `ParquetDatabaseLoader` validates the release and destination, streams rows
through the seven table-specific COPY loaders, and merges by key in dependency
order. All tables share one transaction. The destination schema must exist.

Start with read-only preflight:

```bash
./project/venv/bin/python project/database/ParquetDatabaseLoader.py \
  --dataset project/data/processed/cmv-metadata/v1 --check
```

Load a complete sample by replacing `--check` with `--post-id t3_37i8f6`. Load the
whole release by omitting both selection/check flags:

```bash
./project/venv/bin/python project/database/ParquetDatabaseLoader.py \
  --dataset project/data/processed/cmv-metadata/v1 --batch-size 10000
```

The command prints its report after completion. Tables viewed from another
connection may still show their old rows while the transaction is active.
`SET CONSTRAINTS ALL IMMEDIATE` can take time checking the complete comment tree.
Repeated identical imports report unchanged rows; conflicting source IDs or
immutable relationships fail. Placeholder enrichment is allowed, while available
content is protected from unavailable incoming rows. See
[merge/transaction rules and the class example](../db_loaders/README.md).

`SupabaseConnection.Database` owns the connection: create it, pass it to the
loader, and close it in `finally`. Its `execute("filename.sql", params)` resolves
query files under `database/queries`; other strings are executed as parameterized
SQL. `load_data.py` and the direct `SupabaseConnection.py` entrypoint currently
execute schema creation. They are setup helpers, not normal data-import commands,
and should not be rerun against existing tables as a query-only check.

## Verify an import

```bash
./project/venv/bin/python project/tests/test_loaded_database.py -v
```

The live checks use a read-only transaction to compare counts/source IDs and
selected discussions with Parquet, including true/false labels, parent links,
missing-parent placeholders, pair outcomes, and membership order. Offline tests
do not automatically connect to this database. [Test guide](../../tests/README.md).

For a long command, start `tmux new -s cmv-data`, run the command inside it, and
detach with Ctrl-B then D. Reattach with `tmux attach -t cmv-data`. This keeps a
normal shell available after the final report so errors/results can be reviewed.
