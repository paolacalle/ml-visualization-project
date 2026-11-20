"""Compare average letters per word in the published training pairs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyze_word_frequency import WORDS, minimal_text, paired, score_openings


OUT = paired.DATA / "average-word-length"
ZIPF_RUNS = paired.DATA / "word-frequency-sensitivity"
FEATURE = "average_word_length"
WORD_COUNT = "word_count"
ZIPF = "mean_content_word_zipf"
CORRELATION_FEATURES = (WORD_COUNT, ZIPF, FEATURE)


def measure_opening(body):
    """Average alphabetic characters per token occurrence after Minimal cleaning."""
    text = minimal_text(body or "")
    words = WORDS.findall(text)
    letters = sum(character.isalpha() for word in words for character in word)
    return {
        FEATURE: letters / len(words) if words else np.nan,
        "measured_tokens": len(words),
        "letters": letters,
        "opening_preview": " ".join(text.split())[:240],
    }


def analyze(assignments, openings, content_zipf):
    # Use the archived opening word count so correlations refer to P2's exact input.
    lengths = pd.concat([
        assignments[[f"{side}_comment_id", f"{side}_word_count"]].rename(
            columns={f"{side}_comment_id": "comment_id", f"{side}_word_count": WORD_COUNT}
        )
        for side in ("a", "b")
    ]).drop_duplicates().set_index("comment_id")
    if not lengths.index.is_unique:
        raise ValueError("Opening word count differs across assignments.")
    openings = openings.join(lengths, how="inner").join(content_zipf.rename(ZIPF))
    scores = assignments.copy()
    for side in ("a", "b"):
        for feature in (FEATURE, ZIPF):
            scores[f"{side}_{feature}"] = scores[f"{side}_comment_id"].map(openings[feature])

    a, b = f"a_{FEATURE}", f"b_{FEATURE}"
    scores = scores.dropna(subset=[a, b]).copy()
    if scores.empty:
        raise ValueError("No training pairs have average word lengths on both sides.")
    for feature in CORRELATION_FEATURES:
        scores[f"a_minus_b_{feature}"] = scores[f"a_{feature}"] - scores[f"b_{feature}"]
    # The shared statistic counts positive values. Here a positive value means the winner used shorter words.
    comparison_minus_winner = scores[f"a_minus_b_{FEATURE}"] * (1 - 2 * scores["a_wins"])
    close = scores[f"a_minus_b_{WORD_COUNT}"].abs().le(50)
    probabilities, prediction = paired.evaluate_feature(scores, a, b)
    scores["oof_probability_a_wins"] = probabilities
    opening_ids = pd.unique(scores[["a_comment_id", "b_comment_id"]].to_numpy().ravel())
    return scores, openings.loc[opening_ids], {
        "winner_shorter": paired.paired_higher(scores, comparison_minus_winner),
        "within_50_words": paired.paired_higher(scores.loc[close], comparison_minus_winner.loc[close]),
        "prediction": prediction,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipf-run", type=Path, help="Directory of a saved word-frequency sensitivity run.")
    args = parser.parse_args()
    zipf_run = args.zipf_run
    if zipf_run is None:
        runs = sorted(ZIPF_RUNS.glob("*/results.json"))
        if not runs:
            raise FileNotFoundError(f"No word-frequency sensitivity runs found under {ZIPF_RUNS}.")
        zipf_run = runs[-1].parent
    if json.loads((zipf_run / "results.json").read_text())["split"] != "train":
        raise ValueError("The content-word Zipf source must be a training run.")

    assignments, audit = paired.load_training_pairs()
    openings = score_openings(assignments, measure_opening)
    content_zipf = pd.read_parquet(zipf_run / "opening_scores.parquet", columns=["content_words"])["content_words"]
    scores, analyzed, metrics = analyze(assignments, openings, content_zipf)
    report = {
        "split": "train", "folds": paired.FOLDS, **audit,
        "shared_pairs": len(assignments), "complete_pairs": len(scores),
        "discussions": scores["post_id"].nunique(), "unique_openings": len(analyzed),
        "content_word_zipf_run": zipf_run.name, **metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        scores.to_parquet(staging / "paired_scores.parquet", index=False)
        analyzed.to_parquet(staging / "opening_scores.parquet")
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved average word length run to {destination}")


if __name__ == "__main__":
    main()
