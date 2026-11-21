"""Measure visible Markdown structure in the published training openings."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired
from text_preprocessing import LINK, URL, WORDS

OUT = paired.DATA / "reply-structure"
BUILDER = paired.NOTEBOOKS / "nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
FEATURES = ("paragraph_count", "has_question", "has_list")
LIST_MARKER = re.compile(r"^\s*(?:[-+*]|\d{1,9}[.)])\s+")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
INLINE_CODE = re.compile(r"(`+)[^`\n]+\1")
QUESTION = re.compile(r"\?+")


def prose_lines(lines):
    """Skip fenced code and preserve blank lines and unescaped Markdown markers."""
    fence = ""
    for line in lines:
        marker = FENCE.match(line)
        if fence:
            if (marker and marker[1].startswith(fence) and not marker[2].strip()):
                fence = ""
        elif marker:
            fence = marker[1]
            yield ""
        else:
            yield line


def measure_opening(lines):
    paragraphs = list_items = 0
    in_paragraph = False
    question_example = ""
    list_example = ""
    for raw in prose_lines(lines):
        text = INLINE_CODE.sub("", raw)
        text = URL.sub("", LINK.sub(r"\1", text))
        if not raw.strip():
            in_paragraph = False
        if not WORDS.search(text):
            continue
        if not in_paragraph:
            paragraphs += 1
            in_paragraph = True
        if LIST_MARKER.match(text):
            list_items += 1
            if not list_example:
                list_example = text.strip()[:240]
        if not question_example and (question := QUESTION.search(text)):
            start = max(0, question.start() - 170)
            question_example = text[start:question.end() + 60].strip()

    return {
        "paragraph_count": paragraphs,
        "has_question": int(bool(question_example)),
        "has_list": int(bool(list_items)),
        "list_item_count": list_items,
        "question_example": question_example,
        "list_example": list_example,
    }


def analyze(assignments, openings):
    scores = assignments.copy()
    for side in ("a", "b"):
        for feature in FEATURES:
            scores[f"{side}_{feature}"] = scores[f"{side}_comment_id"].map(openings[feature])
    if scores[[f"{side}_{feature}" for side in ("a", "b") for feature in FEATURES]].isna().any().any():
        raise ValueError("Some training openings could not be scored from Parquet.")
    close = (scores["a_word_count"] - scores["b_word_count"]).abs().le(50)
    results = {}
    for feature in FEATURES:
        a, b = f"a_{feature}", f"b_{feature}"
        difference = scores[a] - scores[b]
        scores[f"a_minus_b_{feature}"] = difference
        winner_difference = difference * (2 * scores["a_wins"] - 1)
        _, performance = paired.evaluate_feature(scores, a, b)
        results[feature] = {
            "all_pairs": paired.paired_higher(scores, winner_difference),
            "within_50_words": paired.paired_higher(scores.loc[close], winner_difference.loc[close]),
            **performance,
        }
    return scores, results


def main():
    assignments, audit = paired.load_training_pairs()
    builder = paired.import_script("branch_builder", BUILDER)
    ids = set(assignments["a_comment_id"]) | set(assignments["b_comment_id"])
    manifest = builder.manifest(builder.TEXT)
    openings = pd.DataFrame([
        {"comment_id": row["comment_id"], **measure_opening(builder.opening_text_lines(row["body"]))}
        for row in builder.text_rows(manifest["files"], ids, "train")
    ]).set_index("comment_id")
    if not openings.index.is_unique:
        raise ValueError("Each training opening must be scored once.")
    scores, metrics = analyze(assignments, openings)
    report = {
        "split": "train", "folds": paired.FOLDS, **audit, "pairs": len(scores),
        "discussions": scores["post_id"].nunique(), "unique_openings": len(openings),
        "archive_sha256": manifest["raw_archive"]["sha256"], "metrics": metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        scores.to_parquet(staging / "paired_scores.parquet", index=False)
        openings.to_parquet(staging / "opening_scores.parquet")
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(f"Saved reply structure run to {destination}")


if __name__ == "__main__":
    main()
