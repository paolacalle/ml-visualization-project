"""Measure published training openings' delays from their OP posts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired


OUT = paired.DATA / "reply-timing"
BUILDER = paired.NOTEBOOKS / "nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
DELAY = "reply_delay_minutes"
TIMING = "a_minus_b_log1p_reply_delay_minutes"
GAP_LABELS = ("0–5 min", ">5–30 min", ">30–120 min", ">120 min")


def score_openings(assignments, builder, metadata):
    post_ids = set(assignments["post_id"])
    ids = set(assignments["a_comment_id"]) | set(assignments["b_comment_id"])
    post_times = {
        row["post_id"]: builder.timestamp(row["created_utc"])
        for row in builder.rows_for(metadata, "posts", ["post_id", "created_utc"], split="train")
        if row["post_id"] in post_ids
    }
    openings = []
    columns = ["comment_id", "post_id", "original_parent_id", "created_utc"]
    for row in builder.rows_for(metadata, "comments", columns, split="train"):
        if row["comment_id"] not in ids:
            continue
        post_id = row["post_id"]
        if row["original_parent_id"] != post_id:
            raise ValueError(f"Opening {row['comment_id']} is not a top-level reply.")
        post_time, comment_time = post_times.get(post_id), builder.timestamp(row["created_utc"])
        delay = (comment_time - post_time) / 60 if post_time is not None and comment_time is not None else np.nan
        openings.append({
            "comment_id": row["comment_id"], "post_id": post_id, "post_created_utc": post_time,
            "reply_created_utc": comment_time, DELAY: delay if delay >= 0 else np.nan,
        })
    frame = pd.DataFrame(openings).set_index("comment_id")
    if not frame.index.is_unique:
        raise ValueError("Each opening must have a single training metadata record.")
    return frame


def analyze(assignments, openings):
    scores = assignments.copy()
    for side in ("a", "b"):
        ids = scores[f"{side}_comment_id"]
        found = ids.map(openings["post_id"])
        if (found.notna() & found.ne(scores["post_id"])).any():
            raise ValueError("An opening's timestamp belongs to a different OP discussion.")
        scores[f"{side}_{DELAY}"] = ids.map(openings[DELAY])
    a, b = f"a_{DELAY}", f"b_{DELAY}"
    scores = scores.dropna(subset=[a, b]).copy()
    if scores.empty:
        raise ValueError("No training pair has two valid delays.")
    scores[TIMING] = np.log1p(scores[a]) - np.log1p(scores[b])
    # Positive values indicate the delta-winning opening was posted first.
    earlier_winner = (scores[a] - scores[b]) * (1 - 2 * scores["a_wins"])
    scores["reply_gap_minutes"] = earlier_winner.abs()
    scores["reply_gap_group"] = pd.cut(
        scores["reply_gap_minutes"], [0, 5, 30, 120, np.inf], labels=GAP_LABELS,
    )
    _, performance = paired.evaluate_differences(scores, scores[[TIMING]])
    return scores, {
        "all_pairs": paired.paired_higher(scores, earlier_winner),
        "by_gap": {
            str(group): paired.paired_higher(rows, earlier_winner.loc[rows.index])
            for group, rows in scores.groupby("reply_gap_group", observed=True)
        },
        "prediction": performance,
    }


def main():
    assignments, audit = paired.load_training_pairs()
    builder = paired.import_script("branch_builder", BUILDER)
    metadata = builder.manifest(builder.METADATA)
    openings = score_openings(assignments, builder, metadata)
    scores, metrics = analyze(assignments, openings)
    report = {
        "split": "train", "folds": paired.FOLDS, **audit, "shared_pairs": len(assignments),
        "pairs": len(scores), "pairs_missing_valid_delay": len(assignments) - len(scores),
        "discussions": scores["post_id"].nunique(), "unique_openings": len(openings),
        "archive_sha256": metadata["raw_archive"]["sha256"], **metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        scores.to_parquet(staging / "paired_scores.parquet", index=False)
        openings.to_parquet(staging / "opening_scores.parquet")
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(f"Saved reply timing to {destination}")


if __name__ == "__main__":
    main()
