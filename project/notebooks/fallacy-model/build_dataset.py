"""Build the logical-fallacy training set (8 fallacy types + "none").

Sources (both use the same 8-type taxonomy):
  * CoCoLoFa (Yeh et al., EMNLP 2024) - crowd-written news comments, official
    article-level train/dev/test splits. https://github.com/Crowd-AI-Lab/cocolofa
  * Sahai et al. (ACL 2021) - real Reddit comments flagged by other users as
    fallacious, then verified by crowd workers, plus sampled negatives.
    https://github.com/sahaisaumya/informal_fallacies

Usage (from the repo root):
    python project/notebooks/fallacy-model/build_dataset.py
    python project/notebooks/fallacy-model/build_dataset.py --n-random-neg 0

Raw files are cached in project/data/raw/fallacy/ and the output parquet files
land in project/data/processed/fallacy/ (both git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "project" / "data" / "raw" / "fallacy"
OUT_DIR = REPO_ROOT / "project" / "data" / "processed" / "fallacy"

LABELS = [
    "none",
    "appeal to authority",
    "appeal to majority",
    "appeal to nature",
    "appeal to tradition",
    "appeal to worse problems",
    "false dilemma",
    "hasty generalization",
    "slippery slope",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# Pinned to specific commits so every teammate builds the same dataset.
COCOLOFA_URL = (
    "https://raw.githubusercontent.com/Crowd-AI-Lab/cocolofa/"
    "c39d45fdd57401e1f6cb674f25113dfa0304e734/{split}.json"
)
SAHAI_URL = (
    "https://raw.githubusercontent.com/sahaisaumya/informal_fallacies/"
    "630d56edb27fb0d5f32dd125ad6e886e4ca2955e/data/raw/{family}/{family}_{kind}.csv"
)
# Sahai et al. folder name -> shared label.
SAHAI_FAMILIES = {
    "authority": "appeal to authority",
    "population": "appeal to majority",
    "natural": "appeal to nature",
    "tradition": "appeal to tradition",
    "worse_problems": "appeal to worse problems",
    "blackwhite": "false dilemma",
    "hasty_generalization": "hasty generalization",
    "slippery_slope": "slippery slope",
}
# When the same text appears more than once, keep the row from the most
# trustworthy source.
SOURCE_PRIORITY = {"cocolofa": 0, "sahai_annotated": 1, "sahai_random_neg": 2}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
WS_RE = re.compile(r"\s+")
REMOVED = {"[deleted]", "[removed]", "nan", "none", ""}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def download_all() -> None:
    for split in ("train", "dev", "test"):
        fetch(COCOLOFA_URL.format(split=split), RAW_DIR / "cocolofa" / f"{split}.json")
    for family in SAHAI_FAMILIES:
        for kind in ("annotated", "negative-samples"):
            fetch(
                SAHAI_URL.format(family=family, kind=kind),
                RAW_DIR / "sahai" / f"{family}_{kind}.csv",
            )


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: object) -> str:
    """Light normalisation so training text looks like what CMV inference sees."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = MD_LINK_RE.sub(r"\1", text)  # [anchor](url) -> anchor
    text = URL_RE.sub("[URL]", text)
    # Reddit quote markers and other markdown noise.
    text = re.sub(r"(?m)^\s*>+\s?", "", text)
    text = text.replace("&nbsp;", " ").replace("**", "").replace("~~", "")
    text = re.sub(r"\\([_*\[\]()~`>#\\-])", r"\1", text)  # markdown escapes: T\_D -> T_D
    return WS_RE.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def hash_split(group: str, dev_frac: float, test_frac: float) -> str:
    """Deterministic group-level split, so one thread never spans two splits."""
    bucket = int(hashlib.md5(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + dev_frac:
        return "dev"
    return "train"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_cocolofa() -> pd.DataFrame:
    rows = []
    for split in ("train", "dev", "test"):
        articles = json.loads((RAW_DIR / "cocolofa" / f"{split}.json").read_text())
        for article in articles:
            for c in article["comments"]:
                rows.append(
                    {
                        "text": c["comment"],
                        "label": c["fallacy"].strip().lower(),
                        "source": "cocolofa",
                        "group": f"cocolofa:{article['id']}",
                        "context": article.get("title", ""),
                        "split": split,
                    }
                )
    return pd.DataFrame(rows)


def load_sahai(n_random_neg: int, seed: int, dev_frac: float, test_frac: float) -> pd.DataFrame:
    frames = []
    for family, label in SAHAI_FAMILIES.items():
        ann = pd.read_csv(RAW_DIR / "sahai" / f"{family}_annotated.csv", index_col=0)
        ann = ann[ann["fallacy_exists"].isin([0, 1])]
        # Crowd workers judged whether the flagged fallacy is really there;
        # rejected rows become hard negatives.
        frames.append(
            pd.DataFrame(
                {
                    "text": ann["parent"],
                    "label": ann["fallacy_exists"].map({1: label, 0: "none"}),
                    "source": "sahai_annotated",
                    "group": "sahai:" + ann["title"].astype(str),
                    "context": ann["title"],
                }
            )
        )
        neg = pd.read_csv(RAW_DIR / "sahai" / f"{family}_negative-samples.csv", sep="\t")
        frames.append(
            pd.DataFrame(
                {
                    "text": neg["parent"],
                    "label": "none",
                    "source": "sahai_random_neg",
                    "group": "sahai:" + neg["title"].astype(str),
                    "context": neg["title"],
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)

    # The random negatives were never human-verified, so cap how many we keep.
    is_rand = df["source"] == "sahai_random_neg"
    rand = df[is_rand]
    if n_random_neg < len(rand):
        rand = rand.sample(n=n_random_neg, random_state=seed)
    df = pd.concat([df[~is_rand], rand], ignore_index=True)

    df["split"] = df["group"].map(lambda g: hash_split(g, dev_frac, test_frac))
    return df


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse rows that share a normalised text.

    * one distinct fallacy label (possibly alongside "none") -> that fallacy
    * two or more distinct fallacy labels                    -> dropped (ambiguous)
    * only "none"                                            -> "none"
    """
    df = df.assign(
        _key=df["text"].map(dedup_key),
        _prio=df["source"].map(SOURCE_PRIORITY),
    ).sort_values(["_key", "_prio"])

    fallacy_sets = (
        df[df["label"] != "none"].groupby("_key")["label"].agg(lambda s: sorted(set(s)))
    )
    ambiguous = set(fallacy_sets[fallacy_sets.map(len) > 1].index)
    resolved = fallacy_sets[fallacy_sets.map(len) == 1].map(lambda s: s[0])

    df = df[~df["_key"].isin(ambiguous)]
    keep = df.drop_duplicates("_key", keep="first").copy()
    keep["label"] = keep["_key"].map(resolved).fillna(keep["label"])
    return keep.drop(columns=["_key", "_prio"]), len(ambiguous)


def build(args: argparse.Namespace) -> pd.DataFrame:
    print("Fetching raw data ...")
    download_all()

    coco = load_cocolofa()
    sahai = load_sahai(args.n_random_neg, args.seed, args.dev_frac, args.test_frac)
    df = pd.concat([coco, sahai], ignore_index=True)

    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unmapped labels: {unknown}")

    df["text"] = df["text"].map(clean_text)
    df["context"] = df["context"].fillna("").map(clean_text)
    n_words = df["text"].str.split().str.len()
    too_short = (n_words < args.min_words) | df["text"].str.lower().isin(REMOVED)
    print(f"Dropped {too_short.sum()} empty/removed/too-short texts")
    df = df[~too_short]

    before = len(df)
    df, n_ambiguous = resolve_duplicates(df)
    print(f"Deduplicated {before} -> {len(df)} rows ({n_ambiguous} texts dropped as ambiguous)")

    df["label_id"] = df["label"].map(LABEL2ID)
    df["n_words"] = df["text"].str.split().str.len()
    cols = ["text", "label", "label_id", "source", "group", "context", "n_words", "split"]
    return df[cols].sample(frac=1, random_state=args.seed).reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    print("\nLabel counts by split:")
    table = pd.crosstab(df["label"], df["split"])[["train", "dev", "test"]]
    table["total"] = table.sum(axis=1)
    print(table.reindex(LABELS).to_string())
    print("\nRows by source and split:")
    print(pd.crosstab(df["source"], df["split"])[["train", "dev", "test"]].to_string())
    print("\nWords per text:", df["n_words"].describe().round(1).to_dict())
    leaks = df.groupby("group")["split"].nunique().gt(1).sum()
    print(f"Groups spanning more than one split: {leaks}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-random-neg", type=int, default=3000,
                        help="Unverified Sahai negatives to keep (default 3000, of ~6250)")
    parser.add_argument("--dev-frac", type=float, default=0.1, help="Sahai dev fraction (by thread)")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Sahai test fraction (by thread)")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = build(args)
    report(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, part in df.groupby("split"):
        part.drop(columns="split").reset_index(drop=True).to_parquet(
            args.out_dir / f"{split}.parquet", index=False
        )
    (args.out_dir / "labels.json").write_text(json.dumps(LABELS, indent=2))
    print(f"\nWrote {len(df)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
