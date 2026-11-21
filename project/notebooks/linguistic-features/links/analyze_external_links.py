"""Measure outbound HTTP(S) links in published CMV training openings."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired


OUT = paired.DATA / "external-links"
FEATURE = "has_external_link"
DIFFERENCE = f"a_minus_b_{FEATURE}"
URL = re.compile(r"""\b(?:https?://|www\.)[^\s<>\[\]()\x60\x22]+""", re.IGNORECASE)
MARKDOWN_LABEL = re.compile(r"!?\[[^\]\n]*\](?=\()")
REDDIT_DOMAINS = ("reddit.com", "redd.it", "redditmedia.com", "redditstatic.com")
structure = paired.import_script(
    "reply_structure", Path(__file__).resolve().parents[1] / "structure/analyze_reply_structure.py",
)


def external_host(url):
    if url.lower().startswith("www."):
        url = "https://" + url
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if "." not in host or any(host == domain or host.endswith("." + domain) for domain in REDDIT_DOMAINS):
        return None
    return host


def measure_opening(lines):
    link_count = 0
    first_host = first_example = ""
    for line in structure.prose_lines(lines):
        text = structure.INLINE_CODE.sub("", line)
        labels = [match.span() for match in MARKDOWN_LABEL.finditer(text)]
        for match in URL.finditer(text):
            if any(start <= match.start() < end for start, end in labels):
                continue
            host = external_host(match[0].rstrip(".,;?!)]}"))
            if host is None:
                continue
            link_count += 1
            if not first_host:
                first_host = host
                first_example = text[max(0, match.start() - 90):match.end() + 90].strip()
    return {
        FEATURE: int(link_count > 0), "external_link_count": link_count,
        "first_external_host": first_host, "first_external_link_context": first_example,
    }


def analyze(assignments, openings):
    scores = assignments.copy()
    for side in ("a", "b"):
        scores[f"{side}_{FEATURE}"] = scores[f"{side}_comment_id"].map(openings[FEATURE])
    a, b = f"a_{FEATURE}", f"b_{FEATURE}"
    if scores[[a, b]].isna().any().any():
        raise ValueError("Some training openings are missing from the text Parquet files.")
    difference = scores[a] - scores[b]
    scores[DIFFERENCE] = difference
    winner_difference = difference * (2 * scores["a_wins"] - 1)
    close = (scores["a_word_count"] - scores["b_word_count"]).abs().le(50)
    _, prediction = paired.evaluate_feature(scores, a, b)
    return scores, {
        "all_pairs": paired.paired_higher(scores, winner_difference),
        "within_50_words": paired.paired_higher(scores.loc[close], winner_difference.loc[close]),
        "both_link": int((scores[a].eq(1) & scores[b].eq(1)).sum()),
        "neither_links": int((scores[a].eq(0) & scores[b].eq(0)).sum()),
        "prediction": prediction,
    }


def main():
    assignments, audit = paired.load_training_pairs()
    builder = paired.import_script("branch_builder", structure.BUILDER)
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
        "openings_with_external_link": int(openings[FEATURE].sum()),
        "archive_sha256": manifest["raw_archive"]["sha256"], **metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        scores.to_parquet(staging / "paired_scores.parquet", index=False)
        openings.to_parquet(staging / "opening_scores.parquet")
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(f"Saved external-link analysis to {destination}")


if __name__ == "__main__":
    main()
