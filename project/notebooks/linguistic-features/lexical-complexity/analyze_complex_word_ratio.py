"""Compare the share of words with three or more syllables in training pairs."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyze_word_frequency import WORDS, minimal_text, paired, score_openings


OUT = paired.DATA / "complex-word-ratio"
FEATURE = "complex_word_ratio"
MIN_SYLLABLES = 3
flesch = paired.import_script(
    "readability_flesch", paired.NOTEBOOKS / "linguistic-features/readability/analyze_flesch.py",
)


def measure_opening(body):
    """Count repeated words and track use of the Flesch fallback separately."""
    text = minimal_text(body or "")
    words = [word.casefold().replace("’", "'") for word in WORDS.findall(text)]
    complex_words = [word for word in words if flesch.syllables(word) >= MIN_SYLLABLES]
    count = len(words)
    return {
        FEATURE: len(complex_words) / count if count else np.nan,
        "measured_tokens": count,
        "complex_tokens": len(complex_words),
        "fallback_tokens": sum(word not in flesch.PRONUNCIATIONS for word in words),
        "complex_examples": ", ".join(list(dict.fromkeys(complex_words))[:8]),
        "opening_preview": " ".join(text.split())[:240],
    }


def analyze(assignments, openings):
    scores = assignments.copy()
    for side in ("a", "b"):
        scores[f"{side}_{FEATURE}"] = scores[f"{side}_comment_id"].map(openings[FEATURE])
    a, b = f"a_{FEATURE}", f"b_{FEATURE}"
    scores = scores.dropna(subset=[a, b]).copy()
    if scores.empty:
        raise ValueError("No training pairs have complex-word ratios on both sides.")
    difference = scores[a] - scores[b]
    scores[f"a_minus_b_{FEATURE}"] = difference
    winner_difference = difference * (2 * scores["a_wins"] - 1)
    close = (scores["a_word_count"] - scores["b_word_count"]).abs().le(50)
    probabilities, prediction = paired.evaluate_feature(scores, a, b)
    scores["oof_probability_a_wins"] = probabilities
    opening_ids = pd.unique(scores[["a_comment_id", "b_comment_id"]].to_numpy().ravel())
    return scores, openings.loc[opening_ids], {
        "winner_higher": paired.paired_higher(scores, winner_difference),
        "within_50_words": paired.paired_higher(scores.loc[close], winner_difference.loc[close]),
        "prediction": prediction,
    }


def main():
    assignments, audit = paired.load_training_pairs()
    scores, openings, metrics = analyze(assignments, score_openings(assignments, measure_opening))
    tokens = openings["measured_tokens"].sum()
    report = {
        "split": "train", "folds": paired.FOLDS, **audit,
        "shared_pairs": len(assignments), "complete_pairs": len(scores),
        "discussions": scores["post_id"].nunique(), "unique_openings": len(openings),
        "min_syllables": MIN_SYLLABLES,
        "cmudict_version": version("cmudict"), "pyphen_version": version("pyphen"),
        "fallback_token_share": float(openings["fallback_tokens"].sum() / tokens),
        "complex_token_share": float(openings["complex_tokens"].sum() / tokens),
        **metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        scores.to_parquet(staging / "paired_scores.parquet", index=False)
        openings.to_parquet(staging / "opening_scores.parquet")
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved complex-word ratio run to {destination}")


if __name__ == "__main__":
    main()
