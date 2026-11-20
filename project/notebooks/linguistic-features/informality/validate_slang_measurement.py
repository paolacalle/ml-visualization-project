"""Prepare blinded CMV annotations and evaluate completed labels."""

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyze_slang_ratio import (
    BIO_CSV, BUILDER, DICTIONARY, OUT as DETECTOR_RUNS, detect_words,
    dictionary_pattern, fit_tagger, minimal_text, paired, sha256,
)


OUT = paired.DATA / "slang-measurement-validation"
ALLOCATIONS = {"neither": 100, "model_only": 75, "dictionary_only": 75, "both": 50}
OPENING_SAMPLE_SIZE = 30
LABELS = ["slang", "colloquial", "internet_shorthand", "profanity", "neutral"]
DETECTORS = ["dictionary_detected", "model_detected"]
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_context(text, start, end, breaks):
    index = bisect_right(breaks, start)
    left = breaks[index - 1] if index else 0
    right = breaks[index] if index < len(breaks) else len(text)
    return f"{text[left:start]}【{text[start:end]}】{text[end:right]}".strip()


def audit_openings(assignments, archived, tagger, pattern):
    # Opening IDs are unique in CMV. Verify their OP grouping before using it for inference.
    post = pd.concat([
        assignments.set_index(f"{side}_comment_id")["post_id"]
        for side in ("a", "b")
    ])
    if not post.groupby(level=0).nunique().eq(1).all():
        raise ValueError("Some openings have more than one OP discussion.")
    posts = post.groupby(level=0).first()
    if not posts.index.difference(archived.index).empty:
        raise ValueError("The source slang run is missing training openings.")

    rng = np.random.default_rng(paired.SEED)
    opening_ids = set(rng.choice(sorted(posts.index), OPENING_SAMPLE_SIZE, replace=False))
    reservoirs = {name: [] for name in ALLOCATIONS}
    population = dict.fromkeys(ALLOCATIONS, 0)
    opening_sample = []
    seen = set()
    builder = paired.import_script("branch_builder", BUILDER)
    manifest = builder.manifest(builder.TEXT)
    for row in builder.text_rows(manifest["files"], set(posts.index), "train"):
        comment_id = row["comment_id"]
        if comment_id in seen:
            raise ValueError(f"Duplicate opening {comment_id}.")
        seen.add(comment_id)
        text = minimal_text(row["body"] or "")
        words, dictionary, model = detect_words(text, tagger, pattern)
        n = len(words)
        saved = archived.loc[comment_id]
        counts = (n, dictionary.sum(), model.sum(), (dictionary & model).sum())
        if tuple(saved[["measured_tokens", "dictionary_tokens", "contextual_tokens", "both_tokens"]]) != counts:
            raise ValueError(f"Detector output differs from the archived slang run for {comment_id}.")
        post_id = posts[comment_id]
        if comment_id in opening_ids:
            opening_sample.append({
                "comment_id": comment_id, "post_id": post_id, "opening": text, "word_count": n,
                "dictionary_ratio": int(dictionary.sum()) / n, "model_ratio": int(model.sum()) / n,
                # Display the exact denominator tokenization for manual word counts.
                "numbered_words": " | ".join(f"{i}: {m[0]}" for i, m in enumerate(words, start=1)),
            })

        # Keeping the smallest independent random priorities gives a uniform sample without replacement.
        priorities = rng.random(n)
        strata = 2 * dictionary.astype(int) + model.astype(int)
        spans = list(pattern.finditer(text.replace("’", "'")))
        span_ends = [match.end() for match in spans]
        breaks = [match.end() for match in SENTENCE_BREAK.finditer(text)]
        for code, (name, capacity) in enumerate(ALLOCATIONS.items()):
            indices = np.flatnonzero(strata == code)
            population[name] += len(indices)
            pool = reservoirs[name]
            if len(indices) > capacity:
                indices = indices[np.argpartition(priorities[indices], capacity - 1)[:capacity]]
            for index in indices:
                priority = float(priorities[index])
                if len(pool) == capacity and priority >= -pool[0][0]:
                    continue
                word = words[index]
                start, end = word.span()
                match = spans[bisect_right(span_ends, start)] if dictionary[index] else None
                item = {
                    "comment_id": comment_id, "post_id": post_id, "span_start": start, "span_end": end,
                    "expression": word[0], "sentence": sentence_context(text, start, end, breaks),
                    "dictionary_detected": bool(dictionary[index]), "model_detected": bool(model[index]),
                    "dictionary_span_start": match.start() if match else None,
                    "dictionary_span_end": match.end() if match else None,
                    "dictionary_expression": text[match.start():match.end()] if match else "",
                    "sampling_stratum": name,
                }
                entry = (-priority, comment_id, start, item)
                if len(pool) == capacity:
                    heapq.heapreplace(pool, entry)
                else:
                    heapq.heappush(pool, entry)

    if seen != set(posts.index):
        raise ValueError("Could not recover all archived training openings.")
    strata_report = {
        name: {"N_h": population[name], "n_h": len(reservoirs[name])} for name in ALLOCATIONS
    }
    rows = [entry[-1] for pool in reservoirs.values() for entry in pool]
    tokens = pd.DataFrame(rows).sample(frac=1, random_state=paired.SEED, ignore_index=True)
    tokens.insert(0, "item_id", [f"T{i:03d}" for i in range(1, len(tokens) + 1)])
    for column in ("N_h", "n_h"):
        tokens[column] = tokens["sampling_stratum"].map({name: values[column] for name, values in strata_report.items()})
    tokens["weight"] = tokens["N_h"] / tokens["n_h"]
    documents = pd.DataFrame(opening_sample).sample(frac=1, random_state=paired.SEED, ignore_index=True)
    documents.insert(0, "item_id", [f"O{i:03d}" for i in range(1, len(documents) + 1)])
    return tokens, documents, strata_report


def prepare(bio_csv):
    latest = sorted(DETECTOR_RUNS.glob("*/results.json"))[-1]
    source = json.loads(latest.read_text())
    if (source["split"] != "train" or sha256(DICTIONARY) != source["dictionary_sha256"]
            or sha256(bio_csv) != source["bio_sha256"]):
        raise ValueError("Use the same dictionary and annotation CSV as the archived training run.")
    assignments, _ = paired.load_training_pairs()
    archived = pd.read_parquet(latest.parent / "opening_scores.parquet")
    tagger, _ = fit_tagger(bio_csv)
    tokens, documents, strata = audit_openings(assignments, archived, tagger, dictionary_pattern())
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        tokens[["item_id", "expression", "sentence"]].assign(**dict.fromkeys(LABELS, "")).to_csv(
            staging / "annotate_tokens.csv", index=False,
        )
        documents[["item_id", "opening", "word_count", "numbered_words"]].assign(target_word_count="").to_csv(
            staging / "annotate_openings.csv", index=False,
        )
        keys = staging / "scoring_keys"
        keys.mkdir()
        tokens.drop(columns="sentence").to_parquet(keys / "tokens.parquet", index=False)
        documents.drop(columns=["opening", "numbered_words"]).to_parquet(keys / "openings.parquet", index=False)
        (staging / "design.json").write_text(json.dumps({
            "seed": paired.SEED, "split": "train", "source_run": latest.parent.name,
            "dictionary_sha256": source["dictionary_sha256"], "bio_sha256": source["bio_sha256"],
            "strata": strata, "opening_sample_size": len(documents),
        }, indent=2) + "\n")
        staging.rename(destination)
    print(f"Saved annotation files to {destination}")
    print(pd.DataFrame(strata).T.to_string())
    return destination


def read_completed_labels(path, key, columns):
    labels = pd.read_csv(path, usecols=["item_id", *columns])
    if labels["item_id"].duplicated().any() or set(labels["item_id"]) != set(key["item_id"]):
        raise ValueError(f"{path.name} must contain exactly one row for every sampled item.")
    if labels[columns].isna().any().any():
        raise ValueError(f"Complete all annotation fields in {path.name} before scoring.")
    return key.merge(labels, on="item_id", validate="one_to_one")


def bootstrap_token_weights(tokens, draws):
    # Share one random multiplier across every token from the same OP discussion.
    rng = np.random.default_rng(paired.SEED)
    groups, posts = pd.factorize(tokens["post_id"])
    weights = np.ones((draws, len(tokens)))
    multipliers = rng.exponential(size=(draws, len(posts)))[:, groups]
    for _, values in tokens.groupby("sampling_stratum"):
        indices = values.index.to_numpy()
        population, sample = values[["N_h", "n_h"]].iloc[0]
        block = multipliers[:, indices]
        calibrated = block / block.mean(axis=1, keepdims=True)
        # Keep each replicate's weighted stratum total at its known population size.
        correction = np.sqrt(1 - sample / population)
        weights[:, indices] = (population / sample) * (1 + correction * (calibrated - 1))
    return weights


def ratio_of(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


def weighted_metrics(tokens, draws=1000):
    labels = tokens[LABELS].astype(bool)
    labels["overall"] = labels[LABELS[:-1]].any(axis=1)
    base = tokens["weight"].to_numpy()
    replicates = bootstrap_token_weights(tokens, draws)
    rows = []
    for detector in DETECTORS:
        detected = tokens[detector].to_numpy(dtype=bool)
        for label in [*LABELS[:-1], "overall"]:
            truth = labels[label].to_numpy()
            hit = detected & truth
            weighted_detected, weighted_truth = base @ detected, base @ truth
            tp = base @ hit
            fp, fn = weighted_detected - tp, weighted_truth - tp
            tn = base.sum() - tp - fp - fn
            precision = ratio_of(replicates @ hit, replicates @ detected)
            recall = ratio_of(replicates @ hit, replicates @ truth)

            def interval(values, numerator, denominator):
                # Degenerate resamples cannot estimate uncertainty for unseen outcomes.
                finite = values[np.isfinite(values)]
                if not len(finite) or numerator <= 0 or numerator >= denominator:
                    return [np.nan, np.nan]
                return np.quantile(finite, [0.025, 0.975])

            precision_low, precision_high = interval(precision, tp, weighted_detected)
            recall_low, recall_high = interval(recall, tp, weighted_truth)
            rows.append({
                "detector": detector.removesuffix("_detected"), "target": label,
                "sample_true_positives": int(hit.sum()),
                "sample_false_negatives": int((~detected & truth).sum()),
                "weighted_tp": tp, "weighted_fp": fp, "weighted_fn": fn, "weighted_tn": tn,
                "precision": tp / weighted_detected if weighted_detected else np.nan,
                "precision_ci_low": precision_low, "precision_ci_high": precision_high,
                "recall": tp / weighted_truth if weighted_truth else np.nan,
                "recall_ci_low": recall_low, "recall_ci_high": recall_high,
            })
    return pd.DataFrame(rows)


def opening_errors(openings):
    reference = openings["target_word_count"] / openings["word_count"]
    rows = []
    for detector in ("dictionary", "model"):
        error = openings[f"{detector}_ratio"] - reference
        rows.append({
            "detector": detector, "openings": len(openings),
            "mean_absolute_error_pp": 100 * error.abs().mean(),
            "mean_signed_error_pp": 100 * error.mean(),
        })
    return pd.DataFrame(rows)


def evaluate(run, token_labels=None, opening_labels=None):
    keys = run / "scoring_keys"
    tokens = read_completed_labels(
        token_labels or run / "annotate_tokens.csv", pd.read_parquet(keys / "tokens.parquet"), LABELS,
    )
    if not tokens[LABELS].isin([0, 1]).all().all():
        raise ValueError("Token labels must be 0 or 1.")
    if not tokens["neutral"].eq(~tokens[LABELS[:-1]].any(axis=1)).all():
        raise ValueError("Set neutral to 1 exactly when all four target labels are 0.")
    token_results = weighted_metrics(tokens)
    document_results = None
    if opening_labels is not None:
        openings = read_completed_labels(
            opening_labels, pd.read_parquet(keys / "openings.parquet"), ["target_word_count"],
        )
        counts = pd.to_numeric(openings["target_word_count"], errors="coerce")
        if not (counts.mod(1).eq(0) & counts.ge(0) & counts.le(openings["word_count"])).all():
            raise ValueError("Enter an integer target count between 0 and word_count on each opening.")
        document_results = opening_errors(openings.assign(target_word_count=counts))
    return token_results, document_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_args = commands.add_parser("prepare")
    prepare_args.add_argument("--bio-csv", type=Path, default=BIO_CSV)
    evaluate_args = commands.add_parser("evaluate")
    evaluate_args.add_argument("run", type=Path)
    evaluate_args.add_argument("--token-labels", type=Path)
    evaluate_args.add_argument("--opening-labels", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.bio_csv)
        return
    tokens, openings = evaluate(args.run, args.token_labels, args.opening_labels)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.run / f"evaluation_{stamp}"
    with TemporaryDirectory(dir=args.run) as temporary:
        staging = Path(temporary)
        tokens.to_csv(staging / "weighted_token_metrics.csv", index=False)
        if openings is not None:
            openings.to_csv(staging / "opening_errors.csv", index=False)
        staging.rename(destination)
    print(tokens.to_string(index=False))
    if openings is not None:
        print(openings.to_string(index=False))
    print(f"Saved evaluation to {destination}")


if __name__ == "__main__":
    main()
