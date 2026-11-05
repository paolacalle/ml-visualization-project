# Database data tests

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
through Parquet. Canonical comment rows and missing-reference reconciliation are
handled separately; this mapper only produces pair/thread/membership rows.
