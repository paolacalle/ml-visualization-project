"""Evaluate P2 plus paragraph count, question presence, and Markdown list presence."""

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


MODEL = "p2_plus_reply_structure"
BASELINE = OUT / "p2_word_count_and_content_word_zipf_differences/20261120T201405034740Z"
STRUCTURE_SOURCE = DATA / "reply-structure"
STRUCTURE = ["a_minus_b_paragraph_count", "a_minus_b_has_question", "a_minus_b_has_list"]
FEATURES = [WORD_COUNT, CONTENT_ZIPF, *STRUCTURE]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-input", type=Path, help="Path to a reply structure paired_scores.parquet.")
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

    structure_path = args.structure_input or latest_input(STRUCTURE_SOURCE)
    structure_report = json.loads(structure_path.with_name("results.json").read_text())
    if structure_report["split"] != "train":
        raise ValueError("Expected training-only structure scores.")
    structure = pd.read_parquet(structure_path).set_index("pair_id")
    if not structure.index.is_unique:
        raise ValueError("Structure scores must contain one row per pair.")
    keys = [*IDENTIFIERS, "a_word_count", "b_word_count"]
    matched = structure.loc[frame["pair_id"]]
    pd.testing.assert_frame_equal(frame[keys].set_index("pair_id"), matched[keys[1:]])
    for column in STRUCTURE:
        differences[column] = frame["pair_id"].map(matched[column])
    if not np.isfinite(differences.to_numpy()).all():
        raise ValueError("Each P2 training pair must have complete structure scores.")

    probability, metrics = paired.evaluate_differences(frame, differences[FEATURES])
    predictions = frame[IDENTIFIERS].join(differences)
    predictions["oof_probability_a_wins"] = probability
    report = {
        "model": MODEL, "features": FEATURES, "split": "train", "folds": paired.FOLDS,
        "pairs": len(frame), "discussions": frame["post_id"].nunique(),
        "baseline_run": BASELINE.name, "input_sha256": baseline["input_sha256"],
        "structure_input_run": structure_path.resolve().parent.name,
        "structure_input_sha256": input_fingerprint(structure_path), **metrics,
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
    print(f"Saved P2 plus reply structure to {destination}")


if __name__ == "__main__":
    main()
