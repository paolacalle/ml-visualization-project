# Readers: choose the input you have

All commands run from the repository root. `PYTHONPATH=project/database` makes
the existing database-module imports available to inline Python snippets.

## ArchiveReader: raw JSON

Logic: stream the outer tar/bzip2 archive, open each recognized inner bzip2 file,
and decode one nonempty JSON line at a time. Yield the member name, physical
one-based line number, and raw dictionary. The dictionary may contain an entire
discussion's comments. Nothing is extracted to disk or inserted into a database.

```bash
PYTHONPATH=project/database ./project/venv/bin/python - <<'PY'
import json
from contextlib import closing
from pathlib import Path
from ArchiveReader import ArchiveReader

source_map = json.loads(Path(
    "project/data/processed/cmv-metadata/source_map.json"
).read_text())
reader = ArchiveReader("project/data/raw/cmv.tar.bz2", source_map)
with closing(reader.iter_records(task="all")) as records:
    member, line, record = next(records)
    print(member, line, record["name"])
    print("Supplied comments:", len(record.get("comments", [])))
PY
```

Use `task="op_task"`, `"pair_task"`, or `None` for other source selections.
Blank lines still count toward physical line numbers. Close an iterator when
stopping early; another call reopens the archive and starts a new sequential pass.
Malformed JSON and duplicate recognized members raise with source context.

## MetadataDataset: prepared rows

Logic: verify a published metadata release, cache its manifest/index, select the
relevant shards, and filter batches by post ID. Yield dictionaries containing the
declared database columns. This reader works without PostgreSQL or the archive.

```bash
PYTHONPATH=project/database ./project/venv/bin/python - <<'PY'
from db_loaders.dataset import MetadataDataset

dataset = MetadataDataset("project/data/processed/cmv-metadata/v1", batch_size=1000)
print(dataset.validate())
post_id = "t3_37i8f6"
print("Expected:", dataset.expected_counts(post_id))
comments = list(dataset.iter_rows("comments", post_id=post_id))
print("Retrieved:", len(comments))
print("First comment:", comments[0]["comment_id"])
PY
```

Call `validate()` once before reading and reuse the instance. Omit `post_id` to
stream a whole table. Sources always return all six snapshot rows. Exhaust the
iterator to check expected counts; use `closing(...)` for an intentional early
exit. Metadata placeholders are included and have `content_available=False`.

## TextStore: local text

Logic: load the text manifest/index, look up a discussion's shard, and filter by
its database ID. `validate()` checks the complete download separately from normal
lookups. The reader does not download files or query PostgreSQL.

```bash
PYTHONPATH=project/database ./project/venv/bin/python - <<'PY'
from TextStore import TextStore

store = TextStore("project/data/processed/cmv-text/v1", expected_version="v1")
post = store.get_post("t3_37i8f6")
comments = store.get_comments(post["post_id"])
print(post["title"])
print("Comments:", len(comments))

# Shape of query results after converting SQL tuples to dictionaries.
rows = [{"post_id": post["post_id"], "comment_id": comments[0]["comment_id"],
         "feature_value": 0.75}]
enriched = store.attach_comment_text(rows)
print(enriched[0]["body"][:120])
PY
```

`get_post` returns `{post_id, title, body}`. `get_comments` returns dictionaries
with `{comment_id, post_id, body}`. `attach_comment_text` copies input rows,
preserves their order/duplicates/metadata, and reads each requested shard once
per call. Unavailable metadata placeholders do not have invented text; omit them
from text requests, for example with `WHERE content_available` in SQL.

After downloading, run `store.validate()` once. Version mismatches, unknown IDs,
missing files, and inconsistent row counts raise errors. See the
[team setup guide](../../../TEAM_DATA.md) for the pinned release checksum and
download layout.
