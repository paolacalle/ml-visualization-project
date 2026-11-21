"""Compare mean English Zipf frequency in the published training pairs."""

from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired
from text_preprocessing import WORDS, minimal_text


OUT = paired.DATA / "word-frequency"
BUILDER = paired.NOTEBOOKS / "nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
LANGUAGE = "en"
WORDLIST = "large"


@lru_cache(maxsize=100_000)
def frequency(word):
    return zipf_frequency(word, LANGUAGE, wordlist=WORDLIST, minimum=0.0)


def measure_opening(body):
    """Average listed token occurrences and track missing lookups separately."""
    text = minimal_text(body or "")
    words = [token.casefold().replace("’", "'") for token in WORDS.findall(text)]
    frequencies = [frequency(word) for word in words]
    listed = [value for value in frequencies if value > 0]
    unlisted = list(dict.fromkeys(word for word, value in zip(words, frequencies) if value == 0))
    return {
        "mean_zipf_frequency": sum(listed) / len(listed) if listed else np.nan,
        "tokens": len(words), "listed_tokens": len(listed),
        "unlisted_tokens": len(words) - len(listed),
        "unlisted_share": 1 - len(listed) / len(words) if words else np.nan,
        "unlisted_examples": ", ".join(unlisted[:6]),
        "opening_preview": " ".join(text.split())[:240],
    }


def score_openings(assignments, measure=measure_opening):
    builder = paired.import_script("branch_builder", BUILDER)
    ids = set(assignments["a_comment_id"]) | set(assignments["b_comment_id"])
    text = builder.manifest(builder.TEXT)
    records = [
        {"comment_id": row["comment_id"], **measure(row["body"])}
        for row in builder.text_rows(text["files"], ids, "train")
    ]
    openings = pd.DataFrame(records).set_index("comment_id")
    if not openings.index.is_unique:
        raise ValueError("Each opening must be scored once.")
    return openings


def analyze(assignments, openings, *, feature="mean_zipf_frequency", difference="a_minus_b_zipf"):
    scored = assignments.copy()
    for side in ("a", "b"):
        ids = scored[f"{side}_comment_id"]
        for column in (feature, "unlisted_share"):
            scored[f"{side}_{column}"] = ids.map(openings[column])
    a, b = f"a_{feature}", f"b_{feature}"
    scored = scored.dropna(subset=[a, b]).copy()
    if scored.empty:
        raise ValueError("No pair has an available frequency score on both sides.")
    scored[difference] = scored[a] - scored[b]
    winner_difference = scored[difference] * (2 * scored["a_wins"] - 1)
    close = (scored["a_word_count"] - scored["b_word_count"]).abs().le(50)
    probabilities, metrics = paired.evaluate_feature(scored, a, b)
    scored["oof_probability_a_wins"] = probabilities

    side_lengths = pd.concat([
        scored[[f"{side}_comment_id", f"{side}_word_count"]].rename(
            columns={f"{side}_comment_id": "comment_id", f"{side}_word_count": "word_count"}
        )
        for side in ("a", "b")
    ]).drop_duplicates().set_index("comment_id")
    if not side_lengths.index.is_unique:
        raise ValueError("Opening word count differs across assignments.")
    analyzed = openings.join(side_lengths, how="inner")
    report = {
        "winner_higher": paired.paired_higher(scored, winner_difference),
        "within_50_words": paired.paired_higher(scored.loc[close], winner_difference.loc[close]),
        "prediction": metrics,
        "spearman_with_word_count": analyzed[feature].corr(
            analyzed["word_count"], method="spearman"
        ),
        "listed_token_share": float(analyzed["listed_tokens"].sum() / analyzed["tokens"].sum()),
    }
    return scored, analyzed, report


def main():
    assignments, audit = paired.load_training_pairs()
    openings = score_openings(assignments)
    scores, analyzed, metrics = analyze(assignments, openings)
    report = {
        "split": "train", "folds": paired.FOLDS, **audit, "shared_pairs": len(assignments),
        "complete_pairs": len(scores), "discussions": scores["post_id"].nunique(),
        "unique_openings": len(analyzed), "wordfreq_version": version("wordfreq"),
        "language": LANGUAGE, "wordlist": WORDLIST,
        "openings_without_listed_tokens": int(openings["listed_tokens"].eq(0).sum()),
        **metrics,
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
    print(f"Saved word-frequency run to {destination}")


if __name__ == "__main__":
    main()
