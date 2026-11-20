"""Run paired feature combinations on the frozen word-frequency sensitivity cohort."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired


DATA = paired.DATA
SOURCE = DATA / "word-frequency-sensitivity"
RARE_WORD_SOURCE = DATA / "rare-word-ratio"
AVERAGE_WORD_LENGTH_SOURCE = DATA / "average-word-length"
OUT = DATA / "paired-models"
WORD_COUNT = "a_minus_b_word_count"
CONTENT_ZIPF = "a_minus_b_mean_content_word_zipf"
RARE_WORD_RATIO = "a_minus_b_rare_word_ratio"
AVERAGE_WORD_LENGTH = "a_minus_b_average_word_length"
IDENTIFIERS = ["pair_id", "post_id", "a_comment_id", "b_comment_id", "a_wins", "fold"]


def load_differences(input_path):
    frame = pd.read_parquet(input_path)
    assignments = pd.read_parquet(DATA / "training_pair_assignments.parquet").set_index("pair_id")
    if not frame["pair_id"].is_unique or set(frame["fold"]) != set(range(1, paired.FOLDS + 1)):
        raise ValueError("Expected unique training pairs in all three discussion folds.")
    columns = [*IDENTIFIERS, "a_word_count", "b_word_count"]
    pd.testing.assert_frame_equal(
        frame[columns].set_index("pair_id"),
        assignments.loc[frame["pair_id"], columns[1:]],
    )
    differences = pd.DataFrame({
        WORD_COUNT: frame["a_word_count"] - frame["b_word_count"],
        CONTENT_ZIPF: frame["content_words_a"] - frame["content_words_b"],
    })
    if not np.isfinite(differences.to_numpy()).all():
        raise ValueError("Every input pair must have both length and content-word Zipf scores.")
    return frame, differences


def load_rare_word_ratio(input_path, frame):
    with input_path.with_name("results.json").open() as source:
        report = json.load(source)
    if report["split"] != "train" or report["rare_zipf_cutoff_exclusive"] != 3:
        raise ValueError("Expected a training run for the primary all-word Zipf < 3 ratio.")
    return load_opening_difference(input_path, frame, "rare_word_ratio")


def load_opening_difference(input_path, frame, feature):
    openings = pd.read_parquet(input_path).set_index("pair_id")
    if not openings.index.is_unique:
        raise ValueError(f"Expected unique pairs in {input_path}.")
    columns = [*IDENTIFIERS[1:], "a_word_count", "b_word_count"]
    matched = openings.loc[frame["pair_id"]]
    pd.testing.assert_frame_equal(frame.set_index("pair_id")[columns], matched[columns])
    difference = matched[f"a_{feature}"] - matched[f"b_{feature}"]
    if not np.isfinite(difference.to_numpy()).all():
        raise ValueError(f"Each baseline pair must have a finite difference for {feature}.")
    return frame["pair_id"].map(difference)


def load_average_word_length(input_path, frame):
    with input_path.with_name("results.json").open() as source:
        report = json.load(source)
    if report["split"] != "train":
        raise ValueError("Expected a training run for average word length.")
    return load_opening_difference(input_path, frame, "average_word_length")


def latest_input(root):
    runs = sorted(root.glob("*/paired_scores.parquet"))
    if not runs:
        raise FileNotFoundError(f"No paired scores found under {root}.")
    return runs[-1]


def input_fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_model(model_name, feature_names):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional path to a saved sensitivity paired_scores.parquet.")
    use_rare_words = RARE_WORD_RATIO in feature_names
    if use_rare_words:
        parser.add_argument("--rare-word-input", type=Path, help="Optional primary rare-word paired scores.")
    use_average_word_length = AVERAGE_WORD_LENGTH in feature_names
    if use_average_word_length:
        parser.add_argument("--average-word-length-input", type=Path,
                            help="Optional average word length paired scores.")
    args = parser.parse_args()
    input_path = args.input or latest_input(SOURCE)
    frame, differences = load_differences(input_path)
    extra_source = {}
    if use_rare_words:
        rare_path = args.rare_word_input or latest_input(RARE_WORD_SOURCE)
        differences[RARE_WORD_RATIO] = load_rare_word_ratio(rare_path, frame)
        extra_source = {
            "rare_word_input_run": rare_path.resolve().parent.name,
            "rare_word_input_sha256": input_fingerprint(rare_path),
        }
    if use_average_word_length:
        word_length_path = args.average_word_length_input or latest_input(AVERAGE_WORD_LENGTH_SOURCE)
        differences[AVERAGE_WORD_LENGTH] = load_average_word_length(word_length_path, frame)
        extra_source.update({
            "average_word_length_input_run": word_length_path.resolve().parent.name,
            "average_word_length_input_sha256": input_fingerprint(word_length_path),
        })
    probability, metrics = paired.evaluate_differences(frame, differences[feature_names])
    predictions = frame[IDENTIFIERS].join(differences)
    predictions["oof_probability_a_wins"] = probability
    report = {
        "model": model_name, "features": feature_names, "split": "train", "folds": paired.FOLDS,
        "pairs": len(frame), "discussions": frame["post_id"].nunique(),
        "input_run": input_path.resolve().parent.name, "input_sha256": input_fingerprint(input_path),
        **extra_source,
        **metrics,
    }

    model_dir = OUT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=model_dir) as temp:
        staging = Path(temp)
        predictions.to_parquet(staging / "oof_predictions.parquet", index=False)
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved to {destination}")
