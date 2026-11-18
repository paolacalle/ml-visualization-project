"""Loaders and text cleaning for the Cornell ChangeMyView Data v1.0 (Tan et al., WWW 2016).

Expected layout (after extracting cmv.tar.bz2 into data/raw/):

    data/raw/all/{train,heldout}_period_data.jsonlist.bz2
    data/raw/pair_task/{train,heldout}_pair_data.jsonlist.bz2
    data/raw/op_task/{train,heldout}_op_data.jsonlist.bz2

Download (original author link is dead; this is the Wayback Machine copy, 336,131,530 bytes,
sha256 25c4e442268e25b49e9f4d30e71394862c8ca01422d3ca219571b8c00689cd08):

    mkdir -p data/raw && cd data/raw
    curl -L -o cmv.tar.bz2 "https://web.archive.org/web/20221226140424id_/https://chenhaot.com/data/cmv/cmv.tar.bz2"
    tar -xjf cmv.tar.bz2
"""

from __future__ import annotations

import bz2
import html
import json
import re
from pathlib import Path
from typing import Iterator

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SPLITS = ("train", "heldout")

# The moderator footnote appended to most original posts, from its "_____" separator to the end.
FOOTNOTE_RE = re.compile(r"(?:_{3,}\s*)?>\s*\*?\s*Hello,? users of CMV!.*", re.S | re.I)
EDIT_LINE_RE = re.compile(r"^\W*(edit|update)\b", re.I)
# Delta markers: the ∆ symbol (U+2206), Greek Δ, "!delta", or the word delta.
DELTA_RE = re.compile(r"∆|Δ|!delta|\bdeltas?\b", re.I)


def read_jsonlist(path: Path) -> Iterator[dict]:
    """Stream one JSON object per line from a .jsonlist.bz2 file."""
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def clean_text(text: str | None, *, drop_quotes: bool = True, drop_edits: bool = False,
               strip_delta: bool = False) -> str:
    """Normalize Reddit markdown text for analysis.

    - Unescapes HTML entities first (twice; a few posts are double-escaped): the raw text
      stores '>' as '&gt;', so quote filters written against '>' silently match nothing
      without this step.
    - Removes the CMV moderator footnote.
    - drop_quotes: remove lines quoting another message ('>' lines). Challengers quote the OP
      heavily, which inflates length and word overlap.
    - drop_edits: remove lines starting with EDIT/UPDATE. In OP text these often announce who
      changed the OP's view, which leaks the label.
    - strip_delta: remove delta markers entirely. (Don't replace them with a placeholder: the
      placeholder itself would then carry the label signal.)
    """
    t = html.unescape(html.unescape(text or ""))  # some posts are double-escaped (&amp;gt;)
    t = FOOTNOTE_RE.sub("", t)
    kept = []
    for line in t.splitlines():
        s = line.strip()
        if drop_quotes and s.startswith(">"):
            continue
        if drop_edits and EDIT_LINE_RE.match(s):
            continue
        kept.append(line)
    t = "\n".join(kept).strip()
    if strip_delta:
        t = DELTA_RE.sub("", t)
    return t


def n_words(text: str) -> int:
    return len(text.split())


def load_op_task(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per original post, labeled by whether the OP changed their view (malleability task)."""
    frames = []
    for split in SPLITS:
        df = pd.DataFrame(read_jsonlist(data_dir / f"op_task/{split}_op_data.jsonlist.bz2"))
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["selftext"].map(lambda s: clean_text(s, drop_quotes=False, drop_edits=True))
    df["n_words"] = df["text"].map(n_words)
    return df


def load_pairs(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the paired persuasion task.

    Returns
    -------
    pairs : one row per pair (the OP and its winning/losing challenger threads).
    args  : long format, one row per challenger thread (two per pair), with
            is_winner = 1 if that thread earned a delta from the OP.
    """
    pair_rows, arg_rows = [], []
    for split in SPLITS:
        for i, r in enumerate(read_jsonlist(data_dir / f"pair_task/{split}_pair_data.jsonlist.bz2")):
            pair_id = f"{split}-{i:04d}"
            pair_rows.append({
                "pair_id": pair_id, "split": split, "op_name": r["op_name"],
                "op_author": r["op_author"], "op_title": html.unescape(r["op_title"]),
                "op_text_raw": r["op_text"],
                "op_text": clean_text(r["op_text"], drop_quotes=False, drop_edits=True),
            })
            for side, is_winner in (("positive", 1), ("negative", 0)):
                path = r[side]
                comments = path["comments"]
                raw = "\n\n".join(c.get("body") or "" for c in comments)
                text = clean_text(raw)
                arg_rows.append({
                    "pair_id": pair_id, "split": split, "op_name": r["op_name"],
                    "is_winner": is_winner, "author": path["author"],
                    "n_comments": len(comments), "text_raw": raw, "text": text,
                    "n_words": n_words(text),
                    # Reddit score is measured after the fact and correlates with winning;
                    # treat it as an outcome, not an input feature.
                    "root_score": comments[0].get("score") if comments else None,
                    "created_utc": comments[0].get("created_utc") if comments else None,
                })
    return pd.DataFrame(pair_rows), pd.DataFrame(arg_rows)


def iter_all_submissions(split: str = "train", data_dir: Path = DATA_DIR) -> Iterator[dict]:
    """Stream full discussion trees from all/ (train: 2.2 GB uncompressed; don't load at once).

    Each submission has Reddit API fields plus 'comments', a flat list whose items carry
    'parent_id' (t3_* = top-level reply to the post, t1_* = reply to a comment).
    """
    yield from read_jsonlist(data_dir / f"all/{split}_period_data.jsonlist.bz2")
