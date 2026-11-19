"""Team data interface: PostgreSQL metadata and matching local Parquet text.

Install with ``python -m pip install -e .`` from the repository root, then use
``from cmv_data import Database, TextStore`` from scripts or notebooks. Importing
this package does not connect, load credentials, or open dataset files.
"""

from .SupabaseConnection import Database
from .TextStore import TextStore

__all__ = ["Database", "TextStore"]
