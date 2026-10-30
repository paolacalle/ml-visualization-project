# Export CMV text to Parquet

Using the existing dataset? Follow the
[one-time team download guide](project/database/TEAM_DATA.md) instead of rerunning
the exporter. It documents the exact private bucket/prefix, local layout, and
complete release validation.

`export_text_parquet.py` streams the raw archive, writes post/comment text with
Zstandard compression, and creates an index mapping each discussion to its files.
It does not connect to PostgreSQL. IDs and the original train/heldout split are
preserved; text is unchanged. Metadata and labels are not included in this export.

From the repository root, with your project environment activated:

```bash
python -m pip install -r requirements.txt
python project/database/export_text_parquet.py
```

Default paths are resolved relative to the script, so the working directory does
not affect them:

- Input: `project/data/raw/cmv.tar.bz2`.
- Output: `project/data/processed/cmv-text/v1`.

The existing `v1` export is retained. To reproduce the export in a new directory:

```bash
python project/database/export_text_parquet.py \
  --output project/data/processed/cmv-text/v2
```

Use `--archive PATH` for a different archive and `--verify-post ID` to select the
discussion used for exact text round-trip verification. The default test post is
`t3_37i8f6`. Both train and heldout discussion files must be present in the archive.

Files are written to an adjacent `.partial` directory and renamed only after
validation succeeds. Existing output or partial directories are never overwritten.
If an export is interrupted, inspect the partial directory and choose a new output
path or remove the incomplete directory yourself before retrying.

The export buffers about 24 MiB per table, plus the current discussion, sorting
buffers, and the in-memory discussion index. A discussion is kept together in a
single comment file. If a file exceeds 50 MB, the exporter stops rather than
silently creating an oversized upload. This bound is checked after compression;
an unusually large individual discussion may require a different policy.

Outputs:

| File/directory | Content |
| --- | --- |
| `posts/split=.../*.parquet` | `post_id`, `title`, `body` |
| `comments/split=.../*.parquet` | `comment_id`, `post_id`, `body` |
| `discussion_index.json` | Post ID to split, post file, comment file, and comment count |
| `manifest.json` | Schemas, source counts, versions, file sizes, checksums, and validation results |

A null comment-file entry means the source provides no comments for that post.
Identifiers are the durable references; Parquet row positions are not IDs.
Validation checks every shard's row count and compares the selected discussion's
exported post/comment text against hashes calculated from the source record.

The original measured export contains 20,626 posts and 1,260,266 comments in
33 Parquet files, approximately 235 MB including its index and manifest. Results
can vary with the input archive and dependency version. See the generated export's
README for a retrieval example. Cloud uploading is a separate step.

## Local validation and tests

```bash
python -m unittest discover -s project/tests -v
```

Use `TextStore.validate()` as shown in the
[team download guide](project/database/TEAM_DATA.md#4-verify-the-complete-download-and-retrieve-the-sample)
to check all local Parquet checksums, sizes, schemas, row counts, and index
references. These operations do not connect to PostgreSQL or cloud storage.
See `project/tests/README.md` for the test command.
