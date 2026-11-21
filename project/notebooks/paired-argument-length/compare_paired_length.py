"""Compare opening length within the paper's published training pairs."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "project/data/processed/paired-argument-length"
BUILDER = HERE.parent / "nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
LENGTH_SCRIPT = HERE.parent / "models-threads/argument-length/train_argument_length.py"
FEATURES = ["word_count", "sentence_count", "average_sentence_length"]
GAP_LABELS = ["1–10", "11–25", "26–50", "51–100", "101+"]
BOOTSTRAP_DRAWS = 1000
SEED = 42


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pairs(split="train"):
    builder = import_script("branch_builder", BUILDER)
    length = import_script("opening_length", LENGTH_SCRIPT)
    metadata, text = builder.manifest(builder.METADATA), builder.manifest(builder.TEXT)
    if metadata["raw_archive"]["sha256"] != text["raw_archive"]["sha256"]:
        raise ValueError("The metadata and text snapshots must come from the same archive.")

    def rows(table, columns):
        return builder.rows_for(metadata, table, columns, split=split)

    pairs = {row["pair_id"]: row["post_id"] for row in rows("pairs", ["pair_id", "post_id"])}
    threads = {
        row["thread_id"]: row for row in rows("threads", ["thread_id", "pair_id", "post_id", "outcome"])
        if row["pair_id"] in pairs
    }
    openings = defaultdict(dict)
    for row in rows("thread_comments", ["thread_id", "post_id", "position", "comment_id"]):
        if row["position"] != 0 or row["thread_id"] not in threads:
            continue
        thread = threads[row["thread_id"]]
        pair_id, outcome = thread["pair_id"], thread["outcome"]
        if row["post_id"] != pairs[pair_id] or thread["post_id"] != pairs[pair_id]:
            raise ValueError(f"Post mismatch for {pair_id}.")
        if outcome in openings[pair_id]:
            raise ValueError(f"Repeated {outcome} opening for {pair_id}.")
        openings[pair_id][outcome] = row["comment_id"]

    ids = {cid for sides in openings.values() for cid in sides.values()}
    parent_data = {
        row["comment_id"]: (row["post_id"], row["original_parent_id"])
        for row in rows("comments", ["comment_id", "post_id", "original_parent_id"])
        if row["comment_id"] in ids
    }
    lengths = {
        row["comment_id"]: length.length_features(row["body"], builder.opening_text_lines)
        for row in builder.text_rows(text["files"], ids, split)
    }
    records = []
    excluded = Counter()
    for pair_id, post_id in pairs.items():
        sides = openings[pair_id]
        if set(sides) != {"positive", "negative"} or sides["positive"] == sides["negative"]:
            excluded["missing or duplicate opening"] += 1
            continue
        winner, comparison = sides["positive"], sides["negative"]
        if any(parent_data.get(cid) != (post_id, post_id) for cid in (winner, comparison)):
            excluded["opening is not a verified top-level reply"] += 1
            continue
        if any(lengths.get(cid) is None for cid in (winner, comparison)):
            excluded["unusable opening text"] += 1
            continue
        record = {
            "pair_id": pair_id, "post_id": post_id,
            "winner_comment_id": winner, "comparison_comment_id": comparison,
        }
        for name, winning, losing in zip(FEATURES, lengths[winner], lengths[comparison]):
            record[f"winner_{name}"] = winning
            record[f"comparison_{name}"] = losing
            record[f"difference_{name}"] = winning - losing
        records.append(record)
    return len(pairs), dict(excluded), pd.DataFrame(records)


def rate_intervals(frame, group, shorter):
    """Resample OP discussions to retain dependence across pairs under one post."""
    grouped = frame.assign(shorter=shorter).groupby(["post_id", group], observed=True)["shorter"]
    totals = grouped.size().unstack(fill_value=0)
    wins = grouped.sum().unstack(fill_value=0)
    rng = np.random.default_rng(SEED)
    n_posts = len(totals)
    probabilities = np.full(n_posts, 1 / n_posts)
    replicates = []
    for _ in range(BOOTSTRAP_DRAWS // 50):
        draw = rng.multinomial(n_posts, probabilities, size=50)
        denominator = draw @ totals.to_numpy()
        numerator = draw @ wins.to_numpy()
        replicates.append(np.divide(
            numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0,
        ))
    lo, hi = np.nanquantile(np.concatenate(replicates), [0.025, 0.975], axis=0)
    return {str(key): [low, high] for key, low, high in zip(totals.columns, lo, hi)}


def summarize(pairs):
    rows = []
    for name in FEATURES:
        difference = pairs[f"difference_{name}"]
        shorter = int(difference.lt(0).sum())
        longer = int(difference.gt(0).sum())
        unequal = shorter + longer
        rows.append({
            "metric": name, "winner_lower": shorter, "winner_higher": longer,
            "ties": int(difference.eq(0).sum()), "winner_lower_rate": shorter / unequal,
            "winner_median": pairs[f"winner_{name}"].median(),
            "comparison_median": pairs[f"comparison_{name}"].median(),
            "median_winner_minus_comparison": difference.median(),
        })

    non_ties = pairs.loc[pairs["difference_word_count"].ne(0)].copy()
    shorter = non_ties["difference_word_count"].lt(0)
    overall = rate_intervals(non_ties.assign(group="overall"), "group", shorter)["overall"]
    non_ties["word_gap"] = pd.cut(
        non_ties["difference_word_count"].abs(), [0, 10, 25, 50, 100, np.inf], labels=GAP_LABELS,
    )
    intervals = rate_intervals(non_ties, "word_gap", shorter)
    gap_stats = []
    for label, group in non_ties.groupby("word_gap", observed=True):
        wins = int(group["difference_word_count"].lt(0).sum())
        gap_stats.append({
            "word_gap": label, "pairs": len(group), "shorter_wins": wins,
            "shorter_win_rate": wins / len(group), "ci_95": intervals[label],
        })
    return rows, overall, gap_stats


def main():
    total, excluded, pairs = load_pairs()
    summaries, interval, gap_stats = summarize(pairs)
    report = {
        "split": "train", "published_pairs": total, "analyzed_pairs": len(pairs),
        "discussions": pairs["post_id"].nunique(), "excluded": excluded,
        "length_metrics": summaries, "shorter_word_count_95_ci": interval,
        "word_gap_groups": gap_stats, "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as directory:
        staging = Path(directory)
        pairs.to_parquet(staging / "pairs.parquet", index=False)
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved paired analysis to {destination}")


if __name__ == "__main__":
    main()
