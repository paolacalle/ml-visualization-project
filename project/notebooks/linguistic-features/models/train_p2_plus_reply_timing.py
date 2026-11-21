"""Evaluate P2 plus the signed difference in log-transformed reply delay."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from paired_model_comparison import (
    CONTENT_ZIPF, DATA, IDENTIFIERS, OUT, WORD_COUNT, input_fingerprint, latest_input,
    load_differences, paired,
)


MODEL = "p2_plus_reply_timing"
BASELINE = OUT / "p2_word_count_and_content_word_zipf_differences/20261120T201405034740Z"
TIMING_SOURCE = DATA / "reply-timing"
TIMING = "a_minus_b_log1p_reply_delay_minutes"
FEATURES = [WORD_COUNT, CONTENT_ZIPF, TIMING]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-input", type=Path, help="Path to a reply timing paired_scores.parquet.")
    args = parser.parse_args()

    baseline = json.loads((BASELINE / "results.json").read_text())
    training = DATA / "word-frequency-sensitivity" / baseline["input_run"] / "paired_scores.parquet"
    if baseline["split"] != "train" or input_fingerprint(training) != baseline["input_sha256"]:
        raise ValueError("Expected the frozen P2 training data.")
    frame, differences = load_differences(training)
    baseline_predictions = pd.read_parquet(BASELINE / "oof_predictions.parquet")
    pd.testing.assert_frame_equal(frame[IDENTIFIERS].join(differences), baseline_predictions[[
        *IDENTIFIERS, WORD_COUNT, CONTENT_ZIPF,
    ]])

    timing_path = args.timing_input or latest_input(TIMING_SOURCE)
    timing_report = json.loads(timing_path.with_name("results.json").read_text())
    if timing_report["split"] != "train":
        raise ValueError("Expected training-only timing scores.")
    timing = pd.read_parquet(timing_path).set_index("pair_id")
    if not timing.index.is_unique:
        raise ValueError("Timing scores must contain one row per pair.")
    keys = [*IDENTIFIERS, "a_word_count", "b_word_count"]
    eligible = frame["pair_id"].isin(timing.index)
    if not timing.index.isin(frame["pair_id"]).all() or not eligible.any():
        raise ValueError("Expected a nonempty subset of archived training pairs.")
    frame = frame.loc[eligible].copy()
    differences = differences.loc[eligible].copy()
    matched = timing.loc[frame["pair_id"]]
    pd.testing.assert_frame_equal(frame[keys].set_index("pair_id"), matched[keys[1:]])
    differences[TIMING] = frame["pair_id"].map(matched[TIMING])
    if not np.isfinite(differences.to_numpy()).all():
        raise ValueError("Each included training pair must have complete timing scores.")

    if eligible.all():
        baseline_probability = baseline_predictions["oof_probability_a_wins"].to_numpy()
    else:
        # Give both models exactly the same fitting and validation rows.
        baseline_probability, _ = paired.evaluate_differences(frame, differences[[WORD_COUNT, CONTENT_ZIPF]])
    probability, metrics = paired.evaluate_differences(frame, differences[FEATURES])
    predictions = frame[IDENTIFIERS].join(differences)
    predictions["p2_oof_probability_a_wins"] = baseline_probability
    predictions["oof_probability_a_wins"] = probability
    report = {
        "model": MODEL, "features": FEATURES, "split": "train", "folds": paired.FOLDS,
        "pairs": len(frame), "discussions": frame["post_id"].nunique(),
        "baseline_run": BASELINE.name, "baseline_refit_on_timing_cohort": not bool(eligible.all()), "input_sha256": baseline["input_sha256"],
        "timing_input_run": timing_path.resolve().parent.name,
        "timing_input_sha256": input_fingerprint(timing_path), **metrics,
    }
    model_dir = OUT / MODEL
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=model_dir) as temporary:
        staging = Path(temporary)
        predictions.to_parquet(staging / "oof_predictions.parquet", index=False)
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved P2 plus reply timing to {destination}")


if __name__ == "__main__":
    main()
