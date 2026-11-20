"""Compare raw, minimally cleaned, and shorthand-expanded Flesch scores."""

from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import cmudict
import numpy as np
import pandas as pd
import pyphen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired
from text_preprocessing import WORDS, minimal_text


OUT = paired.DATA / "flesch"
BUILDER = paired.NOTEBOOKS / "nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
LENGTH_SCRIPT = paired.NOTEBOOKS / "models-threads/argument-length/train_argument_length.py"
VARIANTS = ("raw", "minimal", "expanded")
SHORTHAND = {
    "afaik": "as far as I know", "btw": "by the way", "fwiw": "for what it is worth",
    "idk": "I do not know", "imho": "in my humble opinion", "imo": "in my opinion",
    "lol": "laughing out loud", "tbh": "to be honest", "tl;dr": "too long did not read",
    "tldr": "too long did not read",
}
EXPANSIONS = re.compile(r"\b(?:" + "|".join(map(re.escape, SHORTHAND)) + r")\b", re.I)
PRONUNCIATIONS = cmudict.dict()
HYPHENATOR = pyphen.Pyphen(lang="en_US")


@lru_cache(maxsize=100_000)
def syllables(word):
    pronunciation = PRONUNCIATIONS.get(word)
    if pronunciation:
        return sum(sound[-1].isdigit() for sound in pronunciation[0])
    return 1 + len(HYPHENATOR.positions(word))


def flesch(text, sentence_break):
    """Apply the same word, sentence, and syllable rules to each cleaning condition."""
    sentences = [WORDS.findall(part) for part in sentence_break.split(text)]
    sentences = [words for words in sentences if words]
    words = [word.casefold().replace("’", "'") for sentence in sentences for word in sentence]
    if not words:
        return np.nan
    return 206.835 - 1.015 * len(words) / len(sentences) - 84.6 * sum(map(syllables, words)) / len(words)


def score_openings(assignments):
    builder = paired.import_script("branch_builder", BUILDER)
    length = paired.import_script("opening_length", LENGTH_SCRIPT)
    ids = set(assignments["a_comment_id"]) | set(assignments["b_comment_id"])
    manifest = builder.manifest(builder.TEXT)
    output = []
    for row in builder.text_rows(manifest["files"], ids, "train"):
        raw = row["body"] or ""
        minimal = minimal_text(raw)
        expanded = EXPANSIONS.sub(lambda found: SHORTHAND[found[0].casefold()], minimal)
        output.append({
            "comment_id": row["comment_id"], "original_text": raw, "minimal_text": minimal,
            "expanded_text": expanded, "word_count": builder.opening_word_count(raw),
            **{name: flesch(body, length.SENTENCE_BREAK)
               for name, body in zip(VARIANTS, (raw, minimal, expanded))},
        })
    scores = pd.DataFrame(output).set_index("comment_id")
    if not scores.index.is_unique:
        raise ValueError("Each opening must be scored once.")
    return scores


def direction(difference):
    return np.sign(difference.where(difference.abs() > paired.TIE_TOLERANCE, 0))


def analyze(assignments, openings):
    scored = assignments.copy()
    for variant in VARIANTS:
        for side in ("a", "b"):
            scored[f"{variant}_{side}"] = scored[f"{side}_comment_id"].map(openings[variant])
    score_columns = [f"{v}_{side}" for v in VARIANTS for side in ("a", "b")]
    scored = scored.dropna(subset=score_columns).copy()
    if scored.empty:
        raise ValueError("No pairs have readable opening text in every condition.")
    common_ids = pd.unique(scored[["a_comment_id", "b_comment_id"]].to_numpy().ravel())
    common = openings.loc[common_ids]
    near_length = (scored["a_word_count"] - scored["b_word_count"]).abs().le(50)
    report, shifts = [], []
    for variant in VARIANTS:
        a, b = f"{variant}_a", f"{variant}_b"
        difference = scored[a] - scored[b]
        winner_difference = difference * (2 * scored["a_wins"] - 1)
        scores, metrics = paired.evaluate_feature(scored, a, b)
        scored[f"{variant}_oof_probability_a_wins"] = scores
        report.append({
            "variant": variant, **paired.paired_higher(scored, winner_difference),
            "near_length": paired.paired_higher(scored.loc[near_length], winner_difference.loc[near_length]),
            "spearman_with_word_count": common[variant].corr(common["word_count"], method="spearman"),
            **metrics,
        })
    for before, after in zip(VARIANTS, VARIANTS[1:]):
        original = direction(scored[f"{before}_a"] - scored[f"{before}_b"])
        revised = direction(scored[f"{after}_a"] - scored[f"{after}_b"])
        score_shift = common[after] - common[before]
        shifts.append({
            "comparison": f"{before} → {after}",
            "median_shift": score_shift.median(), "median_absolute_shift": score_shift.abs().median(),
            "opening_rank_spearman": common[before].corr(common[after], method="spearman"),
            "pair_order_reversals": int((original * revised == -1).sum()),
            "ties_changed": int(((original == 0) ^ (revised == 0)).sum()),
        })
    return scored, common, report, shifts


def blind_review(scored, openings):
    """Sample reversals and stable orderings without exposing scores, labels, or IDs."""
    raw = direction(scored["raw_a"] - scored["raw_b"])
    cleaned = direction(scored["minimal_a"] - scored["minimal_b"])
    candidate = raw.mul(cleaned)
    text_a = scored["a_comment_id"].map(openings["original_text"])
    text_b = scored["b_comment_id"].map(openings["original_text"])
    delta_marker = re.compile(r"∆|Δ|!delta|\bdelta\b", re.I)
    valid = (
        scored["a_word_count"].between(25, 170) & scored["b_word_count"].between(25, 170)
        & ~text_a.str.contains(delta_marker) & ~text_b.str.contains(delta_marker)
    )
    rng = np.random.default_rng(paired.SEED)
    chosen = []
    for product in (-1, 1):
        sample = scored.index[valid & candidate.eq(product)].to_numpy()
        chosen.extend(rng.choice(sample, size=min(3, len(sample)), replace=False))
    rng.shuffle(chosen)
    review = pd.DataFrame({"case": range(1, len(chosen) + 1),
                           "opening_A": text_a.loc[chosen].to_list(),
                           "opening_B": text_b.loc[chosen].to_list()})
    key = scored.loc[chosen, ["pair_id", "a_wins", "raw_a", "raw_b", "minimal_a", "minimal_b",
                              "expanded_a", "expanded_b"]].reset_index(drop=True)
    key.insert(0, "case", review["case"])
    return review, key


def main():
    assignments, audit = paired.load_training_pairs()
    scores, openings, metrics, shifts = analyze(assignments, score_openings(assignments))
    review, key = blind_review(scores, openings)
    report = {"split": "train", "folds": paired.FOLDS, **audit, "shared_pairs": len(assignments),
              "complete_pairs": len(scores), "discussions": scores["post_id"].nunique(),
              "cmudict_version": version("cmudict"), "pyphen_version": version("pyphen"),
              "metrics": metrics, "cleaning_shifts": shifts}
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        scores.to_parquet(staging / "paired_scores.parquet", index=False)
        openings.drop(columns=["original_text"]).to_parquet(staging / "opening_scores.parquet")
        review.to_parquet(staging / "blind_review.parquet", index=False)
        key.to_parquet(staging / "review_key.parquet", index=False)
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved Flesch run to {destination}")


if __name__ == "__main__":
    main()
