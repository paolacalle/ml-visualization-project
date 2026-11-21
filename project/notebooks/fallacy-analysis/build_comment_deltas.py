"""Derive comment-level delta labels from the raw CMV archive.

In 2013-2015, a delta was confirmed when DeltaBot replied to the awarding
comment with "Confirmed: 1 delta awarded to /u/<name>". So for every DeltaBot
confirmation:

    DeltaBot comment -> parent = awarding comment (A) -> parent = delta-receiving comment (D)

and we keep the link only if D's author matches the name DeltaBot printed.
Rejected awards ("This delta has been rejected ...") are not counted.

Output: one row per comment (metadata only; text stays in TextStore),
    project/data/processed/cmv-deltas/comments.parquet

Usage (repo root):
    .venv/bin/python project/notebooks/fallacy-analysis/build_comment_deltas.py
Expects the extracted archive members in project/data/raw/all/ (see the README).
"""

from __future__ import annotations

import argparse
import bz2
import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "project" / "data" / "raw" / "all"
OUT_PATH = REPO_ROOT / "project" / "data" / "processed" / "cmv-deltas" / "comments.parquet"

CONFIRM_RE = re.compile(r"Confirmed:\s*\d+\s+deltas?\s+awarded\s+to\s+/?u/([\w-]+)", re.I)
BOTS = {"DeltaBot", "AutoModerator"}
GONE = {"[deleted]", "[removed]"}


def word_count(body: str | None) -> int:
    """Words outside quoted ('>') lines, after unescaping the stored HTML entities."""
    text = html.unescape(html.unescape(body or ""))
    kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    return len(" ".join(kept).split())


def process_post(post: dict, split: str, stats: Counter) -> list[dict]:
    post_id = post["name"]
    op = post.get("author")
    comments = post.get("comments") or []
    by_name = {c["name"]: c for c in comments if c.get("name")}

    deltas_from_op: Counter = Counter()
    deltas_from_other: Counter = Counter()
    for c in comments:
        if c.get("author") != "DeltaBot":
            continue
        m = CONFIRM_RE.search(c.get("body") or "")
        if not m:
            continue
        stats["confirmations"] += 1
        awarder = by_name.get(c.get("parent_id"))
        receiver = by_name.get(awarder.get("parent_id")) if awarder else None
        if receiver is None:
            stats["unlinked_missing_parent"] += 1
            continue
        receiver_author = (receiver.get("author") or "").lower()
        if receiver_author in GONE:
            # Account deleted after the award: the structure still identifies D.
            stats["linked_deleted_author"] += 1
        elif receiver_author != m.group(1).lower():
            stats["unlinked_author_mismatch"] += 1
            continue
        stats["linked"] += 1
        if awarder.get("author") == op:
            deltas_from_op[receiver["name"]] += 1
        else:
            deltas_from_other[receiver["name"]] += 1

    depth_cache: dict[str, int] = {}

    def depth(name: str) -> int:
        # Top-level replies (parent t3_) have depth 0; walk up iteratively.
        chain, cur = [], name
        while cur in by_name and cur not in depth_cache:
            chain.append(cur)
            parent = by_name[cur].get("parent_id") or ""
            if not parent.startswith("t1_") or parent not in by_name:
                depth_cache[cur] = 0 if parent.startswith("t3_") else -1
                chain.pop()
                break
            cur = parent
        base = depth_cache.get(cur, -1)
        for n in reversed(chain):
            base = base + 1 if base >= 0 else -1
            depth_cache[n] = base
        return depth_cache.get(name, -1)

    rows = []
    for c in comments:
        name = c.get("name")
        if not name:
            continue
        author = c.get("author")
        body = c.get("body")
        rows.append({
            "comment_id": name,
            "post_id": post_id,
            "parent_id": c.get("parent_id"),
            "split": split,
            "author": author,
            "is_op": author == op,
            "is_bot": author in BOTS,
            # Author-deleted comments often keep their text, so only the body decides.
            "is_deleted": (body or "").strip() in GONE,
            "author_deleted": author in GONE,
            "depth": depth(name),
            "score": c.get("score"),
            "created_utc": c.get("created_utc"),
            "n_words": word_count(body),
            "deltas_from_op": deltas_from_op.get(name, 0),
            "deltas_from_other": deltas_from_other.get(name, 0),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    stats: Counter = Counter()
    rows: list[dict] = []
    for split in ("train", "heldout"):
        path = args.raw_dir / f"{split}_period_data.jsonlist.bz2"
        print(f"reading {path.name} ...", flush=True)
        with bz2.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.extend(process_post(json.loads(line), split, stats))

    df = pd.DataFrame(rows)
    df["got_delta_op"] = df["deltas_from_op"] > 0
    df["got_delta_any"] = (df["deltas_from_op"] + df["deltas_from_other"]) > 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\n{len(df):,} comments in {df['post_id'].nunique():,} posts -> {args.out}")
    print("DeltaBot confirmations:", dict(stats))
    print(f"comments with an OP delta:  {df['got_delta_op'].sum():,}")
    print(f"comments with any delta:    {df['got_delta_any'].sum():,}")
    print(f"posts with an OP delta:     {df.loc[df['got_delta_op'], 'post_id'].nunique():,}")


if __name__ == "__main__":
    main()
