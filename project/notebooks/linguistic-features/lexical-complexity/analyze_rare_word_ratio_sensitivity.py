"""Compare fixed rarity cutoffs and token filters on published training pairs."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyze_rare_word_ratio import RARE_ZIPF_CUTOFF
from analyze_word_frequency import LANGUAGE, WORDLIST, paired, score_openings
from analyze_word_frequency_sensitivity import CONTENT_TAGS, tagged_word_frequencies


OUT = paired.DATA / "rare-word-ratio-sensitivity"
CUTOFFS = (2, int(RARE_ZIPF_CUTOFF), 4)
FILTERS = ("all_words", "content_words")
BENCHMARK = "content_words_mean_zipf"


def feature_name(token_filter, cutoff):
    return f"{token_filter}_rare_zipf_below_{cutoff}"


def measure_opening(body):
    """Count occurrences and score every cutoff after one POS-tagging pass."""
    values = {token_filter: [] for token_filter in FILTERS}
    for _, tag, frequency in tagged_word_frequencies(body):
        values["all_words"].append(frequency)
        if tag.startswith(CONTENT_TAGS):
            values["content_words"].append(frequency)

    result = {}
    for token_filter, frequencies in values.items():
        listed = [value for value in frequencies if value > 0]
        result[f"{token_filter}_tokens"] = len(frequencies)
        result[f"{token_filter}_listed"] = len(listed)
        for cutoff in CUTOFFS:
            feature = feature_name(token_filter, cutoff)
            count = sum(value < cutoff for value in listed)
            result[f"{feature}_tokens"] = count
            result[feature] = count / len(listed) if listed else np.nan
    # Use the same content-word occurrences for the mean-Zipf benchmark.
    content_listed = [value for value in values["content_words"] if value > 0]
    result[BENCHMARK] = sum(content_listed) / len(content_listed) if content_listed else np.nan
    return result


def analyze(assignments, openings):
    variants = [
        (token_filter, cutoff, feature_name(token_filter, cutoff))
        for token_filter in FILTERS for cutoff in CUTOFFS
    ]
    variants.append(("content_words", None, BENCHMARK))
    scores = assignments.copy()
    columns = []
    for _, _, feature in variants:
        for side in ("a", "b"):
            column = f"{feature}_{side}"
            scores[column] = scores[f"{side}_comment_id"].map(openings[feature])
            columns.append(column)

    # Use the intersection so changes in metrics cannot come from different pairs.
    scores = scores.dropna(subset=columns).copy()
    if scores.empty:
        raise ValueError("No complete training pairs are available for all sensitivity variants.")
    opening_ids = pd.unique(scores[["a_comment_id", "b_comment_id"]].to_numpy().ravel())
    analyzed = openings.loc[opening_ids]
    rows = []
    for token_filter, cutoff, feature in variants:
        a, b = f"{feature}_a", f"{feature}_b"
        winner_difference = (scores[a] - scores[b]) * (2 * scores["a_wins"] - 1)
        probabilities, performance = paired.evaluate_feature(scores, a, b)
        scores[f"{feature}_oof_probability_a_wins"] = probabilities
        tokens = int(analyzed[f"{token_filter}_tokens"].sum())
        listed = int(analyzed[f"{token_filter}_listed"].sum())
        rows.append({
            "feature": feature,
            "token_filter": token_filter,
            "zipf_cutoff_exclusive": cutoff,
            "listed_share": listed / tokens,
            "rare_share_of_listed_tokens": (
                int(analyzed[f"{feature}_tokens"].sum()) / listed if cutoff is not None else None
            ),
            **paired.paired_higher(scores, winner_difference),
            **performance,
        })
    return scores, analyzed, rows


def main():
    assignments, audit = paired.load_training_pairs()
    openings = score_openings(assignments, measure_opening)
    scores, analyzed, metrics = analyze(assignments, openings)
    report = {
        "split": "train", "folds": paired.FOLDS, **audit,
        "shared_pairs": len(assignments), "complete_pairs": len(scores),
        "discussions": scores["post_id"].nunique(), "unique_openings": len(analyzed),
        "wordfreq_version": version("wordfreq"), "nltk_version": version("nltk"),
        "language": LANGUAGE, "wordlist": WORDLIST,
        "content_tags": CONTENT_TAGS, "pos_resource": "averaged_perceptron_tagger_eng",
        "primary_feature": feature_name("all_words", int(RARE_ZIPF_CUTOFF)),
        "metrics": metrics,
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
    print(f"Saved rare-word ratio sensitivity run to {destination}")


if __name__ == "__main__":
    main()
