# ml-visualization-project

Install the shared data package from this directory in your Python environment:

```bash
python -m pip install -e .
```

Use the same imports in scripts and notebooks:

```python
from cmv_data import Database, TextStore
```

Dependencies are pinned in `pyproject.toml`; `pip install -r requirements.txt`
installs the same editable package. See the
[package guide](project/database/guides/PACKAGE.md) for configuration and runners.

For the shared text dataset, follow the
[team connection, download, and starter guide](team-setup/TEAM_DATA.md).
PostgreSQL stores shared metadata and features; local Parquet files provide text
through `TextStore`.

For reader examples and preparation/import commands, see the
[data pipeline guides](project/database/guides/README.md).
To regenerate the text export, see the
[Parquet text-export guide](project/database/guides/PARQUET.md).
