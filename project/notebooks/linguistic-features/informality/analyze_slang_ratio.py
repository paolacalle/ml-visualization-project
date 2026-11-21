"""Compare a fixed slang dictionary with a contextual word tagger on CMV training pairs."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired
from text_preprocessing import WORDS, minimal_text


OUT = paired.DATA / "slang-ratio"
BIO_CSV = paired.NOTEBOOKS.parent / "data/raw/slang/bio_tagged_dataset_final.csv"
BIO_SOURCE = "https://github.com/Sohaila-Abdulsattar/Slang-Detection-and-Identification/blob/main/datasets/bio_tagged_dataset_final.csv"
DICTIONARY = Path(__file__).with_name("slang_dictionary.tsv")
BUILDER = paired.NOTEBOOKS / "nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
METHODS = ("dictionary", "contextual")
THRESHOLD = 0.5
# Preserve punctuation for neighbors and offsets. Count only matches to the shared word tokenizer.
CONTEXT_TOKENS = re.compile(WORDS.pattern + r"|[^\s]")
DICTIONARY_PATTERN_TEMPLATE = r"(?<![\w'’])(?:{})(?![\w'’])"


def normalize(token):
    return token.casefold().replace("’", "'")


def dictionary_pattern():
    terms = pd.read_csv(DICTIONARY, sep="\t")["term"]
    alternatives = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(DICTIONARY_PATTERN_TEMPLATE.format(alternatives), re.IGNORECASE)


def dictionary_hits(text, words, pattern):
    # Map a dictionary span back to word indexes. For example, tl;dr covers tl and dr.
    spans = iter(pattern.finditer(text.replace("’", "'")))
    span = next(spans, None)
    hits = np.zeros(len(words), dtype=bool)
    for index, word in enumerate(words):
        while span is not None and span.end() <= word.start():
            span = next(spans, None)
        hits[index] = span is not None and span.start() < word.end() and span.end() > word.start()
    return hits


def token_features(tokens, index):
    token = tokens[index]
    word = normalize(token)
    previous = normalize(tokens[index - 1]) if index else "<START>"
    following = normalize(tokens[index + 1]) if index + 1 < len(tokens) else "<END>"
    features = {
        "word": word, "previous": previous, "following": following,
        "left_bigram": f"{previous} {word}", "right_bigram": f"{word} {following}",
        "all_caps": token.isupper(), "title_case": token.istitle(),
        "letters": len(word) / 10,
    }
    for size in (2, 3, 4):
        for start in range(len(word) - size + 1):
            features[f"char={word[start:start + size]}"] = True
        if len(word) >= size:
            features[f"prefix={word[:size]}"] = True
            features[f"suffix={word[-size:]}"] = True
    return features


def load_tagged_sentences(path):
    rows = pd.read_csv(path)
    candidates = defaultdict(set)
    rejected = 0
    for sentence, tags in rows[["sentence", "word_labels"]].itertuples(index=False, name=None):
        tokens = tuple(sentence.split())
        labels = tuple(tag.strip() for tag in tags.split(","))
        if len(tokens) != len(labels) or not set(labels) <= {"B", "I", "O"}:
            rejected += 1
            continue
        candidates[tokens].add(labels)
    clean = [(tokens, next(iter(labels))) for tokens, labels in candidates.items() if len(labels) == 1]
    audit = {
        "source_rows": len(rows), "malformed_rows": rejected,
        "conflicting_sentences": sum(len(labels) > 1 for labels in candidates.values()),
        "usable_distinct_sentences": len(clean),
    }
    return clean, audit


def training_tokens(sentences):
    features, labels = [], []
    for tokens, bio in sentences:
        for index, (token, tag) in enumerate(zip(tokens, bio)):
            if WORDS.fullmatch(token):
                features.append(token_features(tokens, index))
                labels.append(tag != "O")
    return features, np.asarray(labels)


def fit_tagger(path):
    sentences, audit = load_tagged_sentences(path)
    has_slang = [any(tag != "O" and WORDS.fullmatch(word) for word, tag in zip(tokens, tags))
                 for tokens, tags in sentences]
    train, validation = train_test_split(
        sentences, test_size=0.2, random_state=paired.SEED, stratify=has_slang,
    )
    x_train, y_train = training_tokens(train)
    x_validation, y_validation = training_tokens(validation)
    tagger = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000))
    tagger.fit(x_train, y_train)
    probabilities = tagger.predict_proba(x_validation)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_validation, probabilities >= THRESHOLD, average="binary", zero_division=0,
    )
    # Refit on the entire external corpus after the diagnostic, using the same settings.
    tagger.fit(x_train + x_validation, np.concatenate((y_train, y_validation)))
    return tagger, {
        **audit, "validation_sentences": len(validation), "validation_tokens": len(y_validation),
        "validation_prevalence": float(y_validation.mean()),
        "validation_average_precision": average_precision_score(y_validation, probabilities),
        "validation_precision": precision, "validation_recall": recall, "validation_f1": f1,
    }


def detect_words(text, tagger, pattern):
    """Return shared word offsets and the two detectors’ flags."""
    tokens = list(CONTEXT_TOKENS.finditer(text))
    surfaces = [token[0] for token in tokens]
    word_indexes = [index for index, token in enumerate(surfaces) if WORDS.fullmatch(token)]
    words = [tokens[index] for index in word_indexes]
    dictionary = dictionary_hits(text, words, pattern)
    contextual = (
        tagger.predict_proba([token_features(surfaces, index) for index in word_indexes])[:, 1] >= THRESHOLD
        if words else np.zeros(0, dtype=bool)
    )
    return words, dictionary, contextual


def measure_opening(body, tagger, pattern):
    text = minimal_text(body or "")
    words, dictionary, contextual = detect_words(text, tagger, pattern)
    result = {"measured_tokens": len(words), "both_tokens": int((dictionary & contextual).sum())}
    for method, hits in zip(METHODS, (dictionary, contextual)):
        count = int(hits.sum())
        result[f"{method}_tokens"] = count
        result[f"{method}_slang_ratio"] = count / len(words) if words else np.nan
        result[f"{method}_matches"] = [normalize(word[0]) for word, hit in zip(words, hits) if hit]
        if count:
            first = words[int(np.argmax(hits))]
            before = text[max(0, first.start() - 55):first.start()]
            after = text[first.end():first.end() + 55]
            snippet = f"{before}【{first[0]}】{after}"
            result[f"{method}_context"] = " ".join(snippet.split())
        else:
            result[f"{method}_context"] = ""
    return result


def analyze(assignments, openings):
    scores = assignments.copy()
    for method in METHODS:
        for side in ("a", "b"):
            scores[f"{method}_{side}"] = scores[f"{side}_comment_id"].map(openings[f"{method}_slang_ratio"])
    scores = scores.dropna(subset=[f"{method}_{side}" for method in METHODS for side in ("a", "b")]).copy()
    if scores.empty:
        raise ValueError("No training pairs have both slang ratios.")
    ids = pd.unique(scores[["a_comment_id", "b_comment_id"]].to_numpy().ravel())
    openings = openings.loc[ids]
    close = (scores["a_word_count"] - scores["b_word_count"]).abs().le(50)
    rows = []
    for method in METHODS:
        a, b = f"{method}_a", f"{method}_b"
        scores[f"{method}_a_minus_b"] = scores[a] - scores[b]
        winner_difference = scores[f"{method}_a_minus_b"] * (2 * scores["a_wins"] - 1)
        probabilities, metrics = paired.evaluate_feature(scores, a, b)
        scores[f"{method}_oof_probability_a_wins"] = probabilities
        rows.append({
            "method": method,
            "winner_higher": paired.paired_higher(scores, winner_difference),
            "within_50_words": paired.paired_higher(scores.loc[close], winner_difference.loc[close]),
            "openings_with_hits": int(openings[f"{method}_tokens"].gt(0).sum()),
            "token_share": float(openings[f"{method}_tokens"].sum() / openings["measured_tokens"].sum()),
            **metrics,
        })
    return scores, openings, rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bio-csv", type=Path, default=BIO_CSV, help="Local external slang annotation CSV.")
    args = parser.parse_args()
    tagger, external = fit_tagger(args.bio_csv)
    pattern = dictionary_pattern()
    assignments, audit = paired.load_training_pairs()
    ids = set(assignments["a_comment_id"]) | set(assignments["b_comment_id"])
    builder = paired.import_script("branch_builder", BUILDER)
    manifest = builder.manifest(builder.TEXT)
    openings = pd.DataFrame([
        {"comment_id": row["comment_id"], **measure_opening(row["body"], tagger, pattern)}
        for row in builder.text_rows(manifest["files"], ids, "train")
    ]).set_index("comment_id")
    if not openings.index.is_unique:
        raise ValueError("Each training opening must be scored once.")
    scores, analyzed, metrics = analyze(assignments, openings)
    report = {
        "split": "train", "folds": paired.FOLDS, **audit,
        "shared_pairs": len(assignments), "complete_pairs": len(scores),
        "discussions": scores["post_id"].nunique(), "unique_openings": len(analyzed),
        "bio_source": BIO_SOURCE, "bio_sha256": sha256(args.bio_csv),
        "dictionary_sha256": sha256(DICTIONARY), "scikit_learn_version": version("scikit-learn"),
        "tagger_threshold": THRESHOLD, "external_validation": external, "metrics": metrics,
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
    print(f"Saved slang ratio run to {destination}")


if __name__ == "__main__":
    main()
