"""Stream the CMV archive into linked CSV tables without extracting raw files."""

import argparse
import bz2
import csv
import json
from contextlib import ExitStack
from pathlib import Path
import tarfile


POST_FIELDS = ["name", "id", "title", "selftext", "author", "created_utc",
               "score", "num_comments", "url", "permalink"]

COMMENT_FIELDS = ["name", "id", "parent_id", "link_id", "author", "body",
                  "created_utc", "score", "distinguished", "replies"]


class Tables:
    
    def __init__(self, directory, stack):
        self.directory = directory
        self.stack = stack
        self.writers = {}
        self.counts = {}

    def write(self, table, record, fields, context=None, omit=()):
        context = context or {}
        
        # create a writer for each table
        if table not in self.writers:
            path = self.directory / f"{table}.csv"
            handle = self.stack.enter_context(path.open("x", newline="", encoding="utf-8"))
            writer = csv.DictWriter(handle, fieldnames=[*context, *fields, "extra_json"])
            writer.writeheader()
            self.writers[table] = writer
            self.counts[table] = 0
            
        row = {**context, **{key: record.get(key) for key in fields}}
        
        # Preserve metadata outside the main analytical columns without losing fields.
        row["extra_json"] = json.dumps(
            {k: v for k, v in record.items() if k not in {*fields, *omit}},
            ensure_ascii=False,
        )
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
        self.writers[table].writerow(row)
        self.counts[table] += 1


def walk_comments(comments):
    """Visit nested Reddit replies without storing a whole second comment list."""
    for comment in comments:
        if not isinstance(comment, dict):
            continue  # Reply IDs refer to rows already in the flat comments list.
        # Replies use Reddit's {kind, data} envelope; top-level comments do not.
        if "kind" in comment and isinstance(comment.get("data"), dict):
            if comment["kind"] != "t1":
                continue  # Reddit 'more' placeholders are not actual comments.
            comment = comment["data"]
        yield comment
        replies = comment.get("replies")
        if isinstance(replies, dict):
            yield from walk_comments(replies.get("data", {}).get("children", []))
        elif isinstance(replies, list):
            yield from walk_comments(replies)


def export_record(tables, task, record, line_number):
    if task == "all":
        tables.write("posts", record, POST_FIELDS, omit=("comments",))
        for comment in walk_comments(record.get("comments", [])):
            tables.write("comments", comment, COMMENT_FIELDS,
                         context={"post_name": record.get("name")})
    elif task == "op_task":
        tables.write("original_posts", record, ["name", "title", "selftext", "delta_label"])
    elif task == "pair_task":
        # Line numbers uniquely identify pairs within this source file/split.
        tables.write("pairs", record, ["op_name", "op_title", "op_text", "op_author"],
                     context={"pair_id": line_number}, omit=("positive", "negative"))
        for outcome in ("positive", "negative"):
            thread = record[outcome]
            tables.write("threads", thread, ["ancestor", "author"],
                         context={"pair_id": line_number, "outcome": outcome},
                         omit=("comments",))
            for position, comment in enumerate(thread.get("comments", [])):
                tables.write("arguments", comment, COMMENT_FIELDS,
                             context={"pair_id": line_number, "op_name": record.get("op_name"),
                                      "outcome": outcome, "position": position})


def parseFile(filepath: Path, output: Path, target=None, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    matched = False
    with tarfile.open(filepath, "r|bz2") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jsonlist.bz2"):
                continue
            if target and member.name != target:
                continue
            task = member.name.split("/")[0]
            if task not in {"all", "op_task", "pair_task"}:
                continue
            matched = True
            destination = output / member.name.removesuffix(".jsonlist.bz2")
            destination.mkdir(parents=True, exist_ok=True)
            # Refuse to mix new output with a prior or interrupted export.
            if any(destination.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {destination}")
            print(f"Processing {member.name} -> {destination}", flush=True)
            with ExitStack() as stack:
                tables = Tables(destination, stack)
                compressed = stack.enter_context(archive.extractfile(member))
                lines = stack.enter_context(bz2.BZ2File(compressed, "rb"))
                # Binary iteration also works with non-seekable tar streams.
                for line_number, line in enumerate(lines, start=1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                        except (ValueError, UnicodeError) as exc:
                            raise ValueError(f"{member.name}, line {line_number}: {exc}") from exc
                        export_record(tables, task, record, line_number)
                    if limit is not None and line_number >= limit:
                        break
                print(f"Rows written: {tables.counts}", flush=True)
            if target:
                break
    if not matched:
        raise FileNotFoundError(target or "No supported CMV JSON files found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=Path, help="Path to cmv.tar.bz2")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("project/data/processed/cmv"))
    parser.add_argument("-t", "--target", help="Exact archive member to process (default: all six)")
    parser.add_argument("-n", "--limit", type=int, help="Maximum input lines per source file")
    args = parser.parse_args()
    
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
        
    parseFile(args.filename, args.output, args.target, args.limit)


if __name__ == "__main__":
    main()
