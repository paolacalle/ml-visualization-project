"""Compare the share of listed words with Zipf below 3 in published training pairs."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from analyze_word_frequency import (
    LANGUAGE, WORDLIST, WORDS, analyze, frequency, minimal_text, paired, score_openings,
)


OUT = paired.DATA / "rare-word-ratio"
RARE_ZIPF_CUTOFF = 3.0
FEATURE = "rare_word_ratio"


def measure_rare_word_ratio(body):
    """Count rare token occurrences among recognized tokens, separately from unlisted tokens."""
    text = minimal_text(body or "")
    words = [token.casefold().replace("’", "'") for token in WORDS.findall(text)]
    frequencies = [frequency(word) for word in words]
    listed = sum(value > 0 for value in frequencies)
    rare = [word for word, value in zip(words, frequencies) if 0 < value < RARE_ZIPF_CUTOFF]
    return {
        FEATURE: len(rare) / listed if listed else np.nan,
        "tokens": len(words), "listed_tokens": listed, "rare_tokens": len(rare),
        "unlisted_tokens": len(words) - listed,
        "unlisted_share": (len(words) - listed) / len(words) if words else np.nan,
        "rare_examples": ", ".join(list(dict.fromkeys(rare))[:8]),
        "opening_preview": " ".join(text.split())[:240],
    }


def main():
    assignments, audit = paired.load_training_pairs()
    openings = score_openings(assignments, measure_rare_word_ratio)
    scores, analyzed, metrics = analyze(
        assignments, openings, feature=FEATURE, difference="a_minus_b_rare_word_ratio"
    )
    report = {
        "split": "train", "folds": paired.FOLDS, **audit, "shared_pairs": len(assignments),
        "complete_pairs": len(scores), "discussions": scores["post_id"].nunique(),
        "unique_openings": len(analyzed), "wordfreq_version": version("wordfreq"),
        "language": LANGUAGE, "wordlist": WORDLIST, "rare_zipf_cutoff_exclusive": RARE_ZIPF_CUTOFF,
        "openings_without_listed_tokens": int(openings["listed_tokens"].eq(0).sum()),
        "rare_share_among_listed_tokens": float(analyzed["rare_tokens"].sum() / analyzed["listed_tokens"].sum()),
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
    print(f"Saved rare-word ratio run to {destination}")


if __name__ == "__main__":
    main()
