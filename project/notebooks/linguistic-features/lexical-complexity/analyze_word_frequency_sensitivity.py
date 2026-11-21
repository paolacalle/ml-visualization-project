"""Compare all-token, content-word, and stopword-filtered Zipf frequency."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from nltk import pos_tag
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from analyze_word_frequency import BUILDER, WORDS, frequency, minimal_text, paired


OUT = paired.DATA / "word-frequency-sensitivity"
VARIANTS = ("all_words", "content_words", "no_stopwords")
POS_TOKENS = re.compile(WORDS.pattern + r"|\d+|[^\w\s]")
CONTENT_TAGS = ("NN", "VB", "JJ", "RB")


def tagged_word_frequencies(body):
    """Yield normalized words, POS tags, and Zipf values from minimally cleaned text."""
    text = minimal_text(body or "")
    # Tag original-case tokens with punctuation for context. Only word tokens enter Zipf scoring.
    for token, tag in pos_tag(POS_TOKENS.findall(text), lang="eng"):
        if not WORDS.fullmatch(token):
            continue
        word = token.casefold().replace("’", "'")
        yield word, tag, frequency(word)


def measure_variants(body):
    result = {variant: [] for variant in VARIANTS}
    for word, tag, value in tagged_word_frequencies(body):
        result["all_words"].append(value)
        if tag.startswith(CONTENT_TAGS):
            result["content_words"].append(value)
        if word not in ENGLISH_STOP_WORDS:
            result["no_stopwords"].append(value)

    measured = {}
    for variant, values in result.items():
        listed = [value for value in values if value > 0]
        measured[variant] = sum(listed) / len(listed) if listed else np.nan
        measured[f"{variant}_tokens"] = len(values)
        measured[f"{variant}_listed"] = len(listed)
    return measured


def score_openings(assignments):
    builder = paired.import_script("branch_builder", BUILDER)
    ids = set(assignments["a_comment_id"]) | set(assignments["b_comment_id"])
    text = builder.manifest(builder.TEXT)
    return pd.DataFrame([
        {"comment_id": row["comment_id"], **measure_variants(row["body"])}
        for row in builder.text_rows(text["files"], ids, "train")
    ]).set_index("comment_id")


def analyze(assignments, openings):
    scores = assignments.copy()
    for variant in VARIANTS:
        for side in ("a", "b"):
            scores[f"{variant}_{side}"] = scores[f"{side}_comment_id"].map(openings[variant])
    columns = [f"{variant}_{side}" for variant in VARIANTS for side in ("a", "b")]
    scores = scores.dropna(subset=columns).copy()
    if scores.empty:
        raise ValueError("The three token rules have no complete pairs in common.")
    opening_ids = pd.unique(scores[["a_comment_id", "b_comment_id"]].to_numpy().ravel())
    analyzed = openings.loc[opening_ids]
    rows = []
    for variant in VARIANTS:
        a, b = f"{variant}_a", f"{variant}_b"
        difference = (scores[a] - scores[b]) * (2 * scores["a_wins"] - 1)
        prediction, performance = paired.evaluate_feature(scores, a, b)
        scores[f"{variant}_oof_probability_a_wins"] = prediction
        tokens = int(analyzed[f"{variant}_tokens"].sum())
        rows.append({
            "variant": variant, **paired.paired_higher(scores, difference), **performance,
            "token_share_retained": tokens / int(analyzed["all_words_tokens"].sum()),
            "listed_share": int(analyzed[f"{variant}_listed"].sum()) / tokens,
        })
    return scores, analyzed, rows


def main():
    assignments, audit = paired.load_training_pairs()
    scores, openings, metrics = analyze(assignments, score_openings(assignments))
    report = {
        "split": "train", "folds": paired.FOLDS, **audit,
        "shared_pairs": len(assignments), "complete_pairs": len(scores),
        "nltk_version": version("nltk"), "wordfreq_version": version("wordfreq"),
        "sklearn_version": version("scikit-learn"),
        "pos_resource": "averaged_perceptron_tagger_eng",
        "stopwords": "scikit-learn ENGLISH_STOP_WORDS", "metrics": metrics,
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
    print(f"Saved sensitivity run to {destination}")


if __name__ == "__main__":
    main()
