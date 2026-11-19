# Team data setup

Use PostgreSQL to query metadata, relationships, and labels. Use `TextStore` to
attach original text from a one-time local download. You do not need the raw
archive, schema scripts, or import runners for analysis.

Run commands from the repository root: the folder containing `requirements.txt`,
`project/`, and `team-setup/`.

## 1. Install dependencies

The pipeline has been tested with Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use your existing environment if requirements are already installed. On Windows,
activate with `.venv\Scripts\activate` instead.

Install the package once in that environment even if its dependencies are already
present. Editable installation makes this checkout's code available as
`from cmv_data import Database, TextStore` from scripts and notebooks without
editing `sys.path`. Use the same Python environment for your notebook kernel.
`pip install -r requirements.txt` performs the same installation.

## 2. Configure your database connection

Ask the owner for your database login/password with permission to read `cmv`.
A Supabase dashboard account, PostgreSQL login, and private Storage download link
are separate types of access. Creating a dashboard account does not automatically
create a personal database login.

If you have dashboard access:

1. Open the [shared project](https://supabase.com/dashboard/project/xofhtpbusoutxtvrdynb).
2. Click **Connect**, choose **Session pooler**, and view the PostgreSQL URI.
3. Copy the exact host, port, database, and username. For an owner-assigned custom
   role, use `ROLE.xofhtpbusoutxtvrdynb` and that role's password. The dashboard's
   default `postgres` URI is the administrator login, not a new personal login.

Without dashboard access, ask the owner for these fields. Do not create another
project. Session mode supports IPv4 networks; direct connections need access to
the project's IPv6 endpoint unless it has the IPv4 add-on. See the official
[connection guide](https://supabase.com/docs/guides/database/connecting-to-postgres).

Create `.env` at the repository root from the template. If `.env` already exists,
edit it rather than replacing it:

```bash
cp -n team-setup/.env.example .env
```

The Supabase URI has this shape:

```text
postgresql://ROLE.PROJECT_REF:[PASSWORD]@[POOLER_HOST]:5432/postgres
```

Our existing `Database` class accepts its components separately:

| `.env` key | Value |
| --- | --- |
| `DB_HOST` | Exact host after `@`, without a scheme or port |
| `DB_PORT` | `5432` for session mode |
| `DB_NAME` | `postgres`, the database after the final `/` |
| `DB_USER` | Complete pooler username including `.PROJECT_REF` |
| `DB_PASSWORD` | Your role's original password, inside quotes |

No full connection string or API key is required by `Database`. Use the literal
password; do not URL-encode it or encode the whole URI. TLS is required by the
wrapper. The starter scripts load the root `.env` without expanding password
characters and close their connections. `.env` is ignored by Git. In your own
notebook outside the checkout, pass an explicit dotenv path if needed:

```python
from cmv_data import Database

db = Database(env_file="/absolute/path/to/your/checkout/.env")
try:
    print(db.execute("SELECT 1;"))
finally:
    db.close()
```

Without `env_file`, `Database()` searches for `.env` from the working directory
upward. Existing environment variables take precedence.

Test the connection and query five comments:

```bash
python team-setup/example_database.py
```

Expect `Connection check: 1`, the sample post, and comment metadata. The example
runs inside a read-only transaction and does not create or reload tables.

## 3. Download the text files once

Ask the owner for **`cmv-text-download-links.json`**. Put it at
`project/data/cmv-text-download-links.json`, then run:

```bash
python team-setup/download_text.py
```

Or provide its current location:

```bash
python team-setup/download_text.py --links ~/Downloads/cmv-text-download-links.json
```

The JSON contains expiring signed links for the private bucket. It grants
temporary access to these objects; share it only within the team and keep it out
of Git. Teammates do not need an admin API key, database password, or dashboard
access to use the links. See Supabase's
[private download documentation](https://supabase.com/docs/guides/storage/serving/downloads).

The script downloads a manifest, discussion index, and 33 Parquet files
(approximately 235 MB) into this layout automatically:

```text
project/data/processed/cmv-text/v1/
├── manifest.json
├── discussion_index.json
├── posts/split=heldout/part-0000.parquet
├── posts/split=train/...
├── comments/split=heldout/...
└── comments/split=train/...
```

Files are streamed to temporary files, verified, then moved into place. Rerunning
skips correct local files and repairs missing/corrupt files. An interrupted file
restarts from the beginning; completed files stay cached. The script finishes by
running `TextStore.validate()` on the complete release. Expired links require a
fresh JSON from the owner; downloaded files remain usable after expiry.

To verify an existing/manual download entirely offline:

```bash
python team-setup/download_text.py --verify-only
```

Expected totals: **20,626 posts**, **1,260,266 comment texts**, **33 Parquet files**.
Metadata also contains missing-parent placeholders without original text, so its
comment total is slightly larger. The published manifest is pinned to SHA-256:

```text
324e4ef93bf16d2928abd89e1dab1ed34652c2a98ff81d4f17741a64495391c0
```

## 4. Query TextStore without the database

```bash
python team-setup/example_textstore.py
```

This prints the sample post title, body preview, and **1,367 comments** for
`t3_37i8f6`. It reads only local files; `.env` is not needed. The main calls are:

```python
from cmv_data import TextStore

store = TextStore("project/data/processed/cmv-text/v1", expected_version="v1")
post = store.get_post("t3_37i8f6")
comments = store.get_comments(post["post_id"])
```

## 5. Query metadata, then attach text

```bash
python team-setup/example_combined.py
```

The [database example](example_database.py) uses SQL parameters to select
`comment_id`, `post_id`, parent ID, author, and score, then converts tuples to
dictionaries. The [combined example](example_combined.py) then calls:

```python
rows_with_text = store.attach_comment_text(rows)
```

Each returned dictionary keeps the metadata and adds `body`. PostgreSQL selects
the rows; `TextStore` uses their IDs to fetch text. Include `WHERE content_available`
in comment queries to exclude missing-parent placeholders. Reuse a `TextStore`
instance and attach text to batches; downloads and full validation are not needed
on every query.

All three examples accept `--post-id`, for example:

```bash
python team-setup/example_combined.py --post-id t3_37i8f6
```

## Owner only: prepare download links

The owner needs the existing local `v1/manifest.json` and an authorized Supabase
secret API key from **Settings → API Keys**. A database password or personal
dashboard access token is not a Storage API key. Keep the secret on the owner's
computer; it is never included in the links JSON.

```bash
python team-setup/create_download_links.py
```

The script prompts for the API key with input hidden, or reads
`SUPABASE_SECRET_KEY` from the owner's environment. It signs the 35 required
objects for 24 hours and writes the ignored
`project/data/cmv-text-download-links.json`. Share only that JSON privately.
Use `--expires-in 604800` for seven days. This does not change bucket permissions.
See the [API key documentation](https://supabase.com/docs/guides/getting-started/api-keys).

| Remote setting | Exact value |
| --- | --- |
| Project | `xofhtpbusoutxtvrdynb` |
| Bucket | `cmv-text` (private) |
| Prefix **inside** the bucket | `cmv-text/v1/` |
| Manifest object key | `cmv-text/v1/manifest.json` |

The bucket and first folder share a name. Keep both in object paths. Published
`v1` bytes must stay unchanged; a new export needs a new release and checksum.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing `DB_*` setting | Fill in the root `.env` using the template's names |
| No module named `cmv_data` | Run `python -m pip install -e .` with the Python environment/kernel you use for analysis |
| Cannot resolve/reach DB host | Copy the exact Session pooler host; check network access |
| Password authentication fails | Check role, complete username, and original password; API keys do not work |
| Permission denied for `cmv` | Ask the owner for database read access |
| HTTP 400/401/403/404 downloading | Ask the owner to check uploaded objects and regenerate links |
| Checksum mismatch | Rerun; do not edit the manifest to bypass checks |
| Unknown text ID | Exclude unavailable placeholders; confirm matching releases |

Further detail: [reader examples](../project/database/guides/READERS.md),
[pipeline guides](../project/database/guides/README.md), and
[text export](../project/database/guides/PARQUET.md).
