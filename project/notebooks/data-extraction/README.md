# CMV tabular export

For the linked SQLite database and Supabase PostgreSQL setup, see
[the database workflow](../../database/README.md).

Run from the repository root (Python standard library only):

```bash
# Preview: three input records from each of the six source files.
python3 project/notebooks/data-extraction/parse-files.py project/data/raw/cmv.tar.bz2 --limit 3 --output project/data/processed/cmv-sample

# Export every record.
python3 project/notebooks/data-extraction/parse-files.py project/data/raw/cmv.tar.bz2 --output project/data/processed/cmv
```

Use `--target op_task/train_op_data.jsonlist.bz2` to select one archive member.
The output directories must be empty. After an interrupted run, use a new output
location or remove the incomplete output before retrying.

Each source gets its own directory, keeping training and heldout data separate:

| Source | CSV tables | Row meaning |
| --- | --- | --- |
| `all/` | `posts.csv`, `comments.csv` | One post or comment |
| `op_task/` | `original_posts.csv` | One original post with `delta_label` |
| `pair_task/` | `pairs.csv`, `threads.csv`, `arguments.csv` | One pair, outcome thread, or comment in that thread |

Join `comments.post_name` to `posts.name`. Within comments, `parent_id` identifies
the parent post or comment (`t3_` and `t1_` prefixes). Some reply references may
have no corresponding comment in the provided data. Reply metadata is retained
as JSON in the `replies` column.

Join pairs, threads, and arguments by `pair_id` within the same source directory;
join threads and arguments by both `pair_id` and `outcome`. `position` is the
zero-based order of a comment in the provided thread. The `positive` outcome is
the successful argument; `negative` is the matched unsuccessful argument.
`op_name` links pairs to posts from the corresponding train/heldout period.

Common analytical fields are individual columns. Other metadata is retained in
`extra_json`; remaining lists/dictionaries are JSON strings inside CSV cells.
Missing values become empty cells, and booleans are written as `True`/`False`.
Text is preserved as provided, including Markdown and newlines; use a CSV reader
rather than splitting output into lines. No text cleaning or model fitting is done.

The exporter streams both compression layers and keeps one input record in
memory. Selecting a file late in the archive still requires decompressing the
preceding archive content. Raw files are never extracted to disk. Only a sample
export is included; run the full command when ready to process the entire dataset.
