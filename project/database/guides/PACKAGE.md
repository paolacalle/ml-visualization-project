# Using the cmv_data package

From the repository root, activate your Python environment and install:

```bash
python -m pip install -e .
```

The distribution is named `cmv-data`; its Python import is `cmv_data`. Dependencies
are pinned in `pyproject.toml`, and `requirements.txt` delegates to the same
package. Editable installation uses the code in your checkout, so Python sees
subsequent edits without reinstalling. Restart an already-running notebook kernel
after editing imported modules. Reinstall after changing dependency metadata or
console commands. The pipeline is tested with Python 3.13.

## Main interface

```python
from cmv_data import Database, TextStore

store = TextStore("/absolute/path/to/cmv-text/v1", expected_version="v1")
post = store.get_post("t3_37i8f6")
print(post["title"])

db = Database(env_file="/absolute/path/to/checkout/.env")
try:
    rows = db.execute("SELECT comment_id, post_id, score FROM cmv.comments "
                      "WHERE post_id = %s AND content_available "
                      "ORDER BY comment_id LIMIT %s;", (post["post_id"], 5))
    records = [dict(zip(("comment_id", "post_id", "score"), row)) for row in rows]
    enriched = store.attach_comment_text(records)
finally:
    db.close()
```

Importing the package opens no database connection and reads no dataset.
`Database()` connects on construction. It uses `DB_HOST`, `DB_PORT`, `DB_NAME`,
`DB_USER`, and `DB_PASSWORD`, with TLS required. Without an explicit `env_file`,
it looks for `.env` from the working directory upward. Environment variables
override dotenv values; passwords stay literal, including `${...}` characters.

`db.execute("test.sql")` loads the SQL resource bundled with the package, regardless
of working directory. `db.execute(sql, params)` also accepts SQL strings as before.
The package contains code, guides, and SQL resources; generated datasets and
credentials are not part of the installation. Dataset locations are supplied to
the readers explicitly.

## Preparation and import commands

Installation adds two commands to the active environment:

```bash
cmv-prepare-metadata --help
cmv-load-metadata --help
```

They call the existing runners. The original
`python project/database/MetadataRunner.py ...` and
`python project/database/ParquetDatabaseLoader.py ...` commands also work after
installation. Use [runner instructions](RUNNERS.md) for full arguments and the
read-only database preflight. Python imports for those classes are:

```python
from cmv_data.MetadataRunner import MetadataRunner
from cmv_data.ParquetDatabaseLoader import ParquetDatabaseLoader
from cmv_data.ArchiveReader import ArchiveReader
from cmv_data.db_loaders.dataset import MetadataDataset
```

Keep the main guard when using `MetadataRunner` with multiple workers.

## Source layout and verification

`pyproject.toml` maps the installed package to `project/database/`. This preserves
the existing source paths and the single `queries/` directory. Subpackages use
relative imports; executable root modules use the installed `cmv_data` namespace
so direct script execution remains available.

```bash
python -m pip check
python -m unittest discover -s project/tests
```

The package tests check public imports, dotenv lookup/password handling, bundled
SQL access from another directory, and all preparation/loader module imports.
The existing tests continue to cover mapping, multiprocessing, Parquet, text
lookups, and database-loader contracts. See [tests](../../tests/README.md).

For teammate credentials, download instructions, and runnable query examples,
use [team setup](../../../team-setup/TEAM_DATA.md).
