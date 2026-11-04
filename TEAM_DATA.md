# One-time team download: CMV text v1

Download the shared text release once. PostgreSQL remains the shared query layer
for metadata, trees, labels, and features; `TextStore` reads text locally and joins
it to query results by ID. Teammates do not need the raw archive or to rerun the
exporter to use this release.

## Exact remote location

| Setting | Value |
| --- | --- |
| Supabase project reference | `xofhtpbusoutxtvrdynb` |
| Storage bucket | `cmv-text` |
| Bucket access | Private |
| Object prefix inside the bucket | `cmv-text/v1/` |
| Manifest object key | `cmv-text/v1/manifest.json` |
| Discussion-index object key | `cmv-text/v1/discussion_index.json` |
| Version | `v1` |
| Expected manifest SHA-256 | `324e4ef93bf16d2928abd89e1dab1ed34652c2a98ff81d4f17741a64495391c0` |

The bucket and its first folder happen to share a name. Select bucket `cmv-text`,
then open folder `cmv-text`, then folder `v1`. Do not omit that first folder from
remote object keys. Manifest/index file references are relative to the `v1` root.

Example remote object: bucket `cmv-text`, key
`cmv-text/v1/comments/split=heldout/part-0002.parquet`.

## 1. Set up the Python environment

The current workflow has been verified with Python 3.13. From the repository root,
use an existing project environment or create one:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The remaining commands run from the repository root with that environment active.
Text-only reads and validation do not require a `.env` file or database connection.

## 2. Obtain access and download the release

You need authorized access to the project's Storage dashboard. If the project is
not visible in your account, ask the project owner to arrange access before
downloading. A PostgreSQL login is separate from private-bucket download access.
Do not distribute the project's admin password or secret API key as a substitute.

1. Sign in to the [Supabase dashboard](https://supabase.com/dashboard) and select
   the project above.
2. Open Storage, select bucket `cmv-text`, and navigate into `cmv-text/v1/`.
3. Download `manifest.json` and `discussion_index.json`.
4. Download every Parquet object named in the manifest's `files` list. Use the
   dashboard download action for each file; no bulk-folder-download capability is
   assumed. Preserve the relative directories shown below.
5. Download `README.md` if desired. It is documentation, not required by `TextStore`.

Required files: **33 Parquet files plus two JSON files**, approximately **235 MB**.
The largest Parquet file is about 7.8 MB. This is a one-time transfer per release.

## 3. Put the files in the standard local directory

Create `project/data/processed/cmv-text/v1/` inside your checkout. The remote
`cmv-text/v1/` prefix maps to this local directory; do not add another nested copy
of the bucket name under `v1`.

```text
project/data/processed/cmv-text/v1/
├── manifest.json
├── discussion_index.json
├── README.md                         # optional
├── posts/
│   ├── split=train/
│   │   ├── part-0000.parquet
│   │   └── part-0001.parquet
│   └── split=heldout/
│       └── part-0000.parquet
└── comments/
    ├── split=train/
    │   └── part-0000.parquet ... part-0025.parquet
    └── split=heldout/
        └── part-0000.parquet ... part-0003.parquet
```

Identical basenames occur in different directories. Do not flatten the download
or allow browser-added suffixes such as `(1)` to become final dataset filenames.
The local manifest and index must refer to the same release as the Parquet files.
Keep these generated files out of Git; `project/data/` is already ignored.

## 4. Verify the complete download and retrieve the sample

Run this command from the repository root. It first checks the downloaded
manifest against the pinned release checksum above, then validates every listed
file and retrieves text without connecting to PostgreSQL or cloud storage:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from project.database.TextStore import TextStore

root = Path("project/data/processed/cmv-text/v1")
expected_manifest_hash = "324e4ef93bf16d2928abd89e1dab1ed34652c2a98ff81d4f17741a64495391c0"
actual_manifest_hash = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
if actual_manifest_hash != expected_manifest_hash:
    raise ValueError("Manifest does not match the published v1 release")

store = TextStore(root, expected_version="v1")
print(json.dumps(store.validate(), indent=2))

post = store.get_post("t3_37i8f6")
comments = store.get_comments("t3_37i8f6")
if len(comments) != 1367:
    raise ValueError("Expected 1,367 comments for the sample discussion")
print("Sample post:", post["post_id"])
print("Sample comments:", len(comments))
print("PASS: downloaded v1 is ready for local text loading")
PY
```

Expected validation summary:

```json
{
  "version": "v1",
  "parquet_files": 33,
  "post_rows": 20626,
  "comment_rows": 1260266,
  "index_posts": 20626,
  "index_checksum_verified": true
}
```

`validate()` checks sizes, checksums, schemas, row counts, index references, and
source totals. If it reports a missing or mismatched file, redownload that exact
object into the matching directory and validate again; do not alter the manifest
to make a failed check pass. An expired session/download link requires renewed
authorized access, not a change to the dataset.

The checksum above pins this published `v1`. A changed export belongs in a new
version directory with its own manifest and documented checksum; do not replace
`v1` in place.

## 5. Use the local text with database query results

PostgreSQL queries should return `comment_id` and `post_id` along with any desired
metadata. Convert the result tuples to dictionaries and pass them to the loader:

```python
from project.database.TextStore import TextStore

store = TextStore("project/data/processed/cmv-text/v1", expected_version="v1")

# Example shape of a database query result; these calls do not query PostgreSQL.
rows = [{"comment_id": "t1_crmwbkm", "post_id": "t3_37i8f6", "score": 9}]
rows_with_text = store.attach_comment_text(rows)
print(rows_with_text[0]["body"])
```

The loader preserves input order, repeated requests, and metadata, and reads each
requested file once per batch. Keep the `TextStore` instance for subsequent calls;
there is no need to download or hash the full release per query. PostgreSQL
credentials are only needed for the SQL portion of a real workflow.

