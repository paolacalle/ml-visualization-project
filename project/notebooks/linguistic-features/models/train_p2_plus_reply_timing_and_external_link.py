"""Test external-link presence beyond P2 plus log-transformed reply delay."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from paired_model_comparison import DATA, IDENTIFIERS, OUT, input_fingerprint, latest_input, paired
from train_p2_plus_reply_timing import FEATURES as TIMING_FEATURES


MODEL = "p2_plus_reply_timing_and_external_link"
BASELINE = OUT / "p2_plus_reply_timing/20261120T215323283056Z"
LINK_SOURCE = DATA / "external-links"
LINK_FEATURE = "a_minus_b_has_external_link"
FEATURES = [*TIMING_FEATURES, LINK_FEATURE]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link-input", type=Path, help="Path to an external-link paired_scores.parquet.")
    args = parser.parse_args()
    baseline = json.loads((BASELINE / "results.json").read_text())
    if baseline["split"] != "train" or baseline["features"] != TIMING_FEATURES:
        raise ValueError("Expected the archived training-only timing model.")
    archived_path = BASELINE / "oof_predictions.parquet"
    frame = pd.read_parquet(archived_path)
    if len(frame) != baseline["pairs"] or not frame["pair_id"].is_unique:
        raise ValueError("The timing model must have one prediction per archived training pair.")
    frozen = pd.read_parquet(DATA / "training_pair_assignments.parquet").set_index("pair_id")
    pd.testing.assert_frame_equal(
        frame[IDENTIFIERS].set_index("pair_id"), frozen.loc[frame["pair_id"], IDENTIFIERS[1:]],
    )

    link_path = args.link_input or latest_input(LINK_SOURCE)
    link_metadata = json.loads(link_path.with_name("results.json").read_text())
    if link_metadata["split"] != "train":
        raise ValueError("Expected training-only external-link scores.")
    links = pd.read_parquet(link_path).set_index("pair_id")
    if not links.index.is_unique:
        raise ValueError("Link scores must contain one row per pair.")
    matched = links.loc[frame["pair_id"]]
    pd.testing.assert_frame_equal(frame[IDENTIFIERS].set_index("pair_id"), matched[IDENTIFIERS[1:]])
    frame[LINK_FEATURE] = frame["pair_id"].map(matched[LINK_FEATURE])
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Each timing pair must have a complete external-link score.")

    probability, metrics = paired.evaluate_differences(frame, frame[FEATURES])
    scores = frame[[*IDENTIFIERS, *FEATURES]].copy()
    scores["p2_plus_reply_timing_oof_probability_a_wins"] = frame["oof_probability_a_wins"]
    scores["oof_probability_a_wins"] = probability
    report = {
        "model": MODEL, "features": FEATURES, "split": "train", "folds": paired.FOLDS,
        "pairs": len(frame), "discussions": frame["post_id"].nunique(),
        "baseline_run": BASELINE.name, "baseline_predictions_sha256": input_fingerprint(archived_path),
        "link_input_run": link_path.resolve().parent.name,
        "link_input_sha256": input_fingerprint(link_path), **metrics,
    }
    model_dir = OUT / MODEL
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=model_dir) as temporary:
        staging = Path(temporary)
        scores.to_parquet(staging / "oof_predictions.parquet", index=False)
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved external-link model to {destination}")


if __name__ == "__main__":
    main()
