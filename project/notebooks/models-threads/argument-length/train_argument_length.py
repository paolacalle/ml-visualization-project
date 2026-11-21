"""Predict later OP deltas using three length features from the opening reply."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits


REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "project/data/processed"
BASELINE = DATA / "m0-baseline/1h"
OUTPUT = DATA / "argument-length/1h"
BUILDER = REPO / "project/notebooks/nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
TARGET = "delta_after_1h"
KEYS = ["post_id", "branch_id", TARGET]
FEATURES = ["word_count", "sentence_count", "average_sentence_length"]
SENTENCE_BREAK = re.compile(r"""[.!?]+["'”’)\]]*(?=\s|$)|\n[ \t]*\n""")
HAS_WORD = re.compile(r"[^\W_]")
SEED = 42
BOOTSTRAP_DRAWS = 500


def length_features(body, clean_lines):
    """Count whitespace-delimited words and approximate sentences in archived text."""
    clean = "\n".join(clean_lines(body))
    words = len(clean.split())
    sentences = sum(bool(HAS_WORD.search(span)) for span in SENTENCE_BREAK.split(clean))
    if not words or not sentences:
        return None
    return words, sentences, words / sentences


def load_cohort():
    audit = json.loads((BASELINE / "train/audit.json").read_text())
    if (audit["split"], audit["observation_hours"], audit["followup_days_after_cutoff"]) != ("train", 1, 7):
        raise ValueError("Expected the one-hour training cohort with a seven-day follow-up.")
    branches = pd.read_parquet(BASELINE / "train/branches.parquet", columns=[*KEYS, "eligible", "opening_reply_length"])
    eligible = branches.loc[branches["eligible"], [*KEYS, "opening_reply_length"]].sort_values("branch_id").reset_index(drop=True)
    eligible[TARGET] = eligible[TARGET].astype(int)
    folds = pd.read_parquet(BASELINE / "training_oof_predictions.parquet", columns=[*KEYS, "fold"])
    folds = folds.sort_values("branch_id").reset_index(drop=True)
    if not eligible["branch_id"].is_unique or not eligible[KEYS].equals(folds[KEYS]):
        raise ValueError("Saved validation folds do not match the training cohort.")
    if set(folds["fold"]) != {1, 2, 3} or not folds.groupby("post_id")["fold"].nunique().eq(1).all():
        raise ValueError("Expected three folds grouped by post_id.")
    eligible["fold"] = folds["fold"]

    # Reuse the branch builder's Parquet reader and opening-text cleaning.
    spec = importlib.util.spec_from_file_location("branch_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    text = builder.manifest(builder.TEXT)
    if text["raw_archive"]["sha256"] != audit["raw_archive_sha256"]:
        raise ValueError("Branch metadata and opening text come from different archives.")
    records = []
    for row in builder.text_rows(text["files"], set(eligible["branch_id"]), "train"):
        features = length_features(row["body"], builder.opening_text_lines)
        if features is not None:
            records.append((row["comment_id"], *features))
    lengths = pd.DataFrame(records, columns=["branch_id", *FEATURES])
    cohort = eligible.merge(lengths, on="branch_id", how="inner", validate="one_to_one")
    if not cohort["opening_reply_length"].eq(cohort["word_count"]).all():
        raise ValueError("Recomputed opening word counts disagree with the baseline input.")
    return eligible, cohort.drop(columns="opening_reply_length")


def make_classifier():
    """Use fixed L2 regularization to keep the three-fold comparison simple."""
    return make_pipeline(
        FunctionTransformer(np.log1p), StandardScaler(),
        LogisticRegression(C=1, solver="lbfgs", max_iter=1000),
    )


def scores(y, probability):
    return {
        "branches": len(y),
        "later_deltas": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
    }


def fit_classifier(cohort):
    fold_scores = []
    probability = np.full(len(cohort), np.nan)
    for fold in (1, 2, 3):
        test = cohort["fold"].eq(fold)
        fitting, validation = cohort.loc[~test], cohort.loc[test]
        model = make_classifier().fit(fitting[FEATURES], fitting[TARGET])
        fold_probability = model.predict_proba(validation[FEATURES])[:, 1]
        probability[test.to_numpy()] = fold_probability
        fold_scores.append({"fold": fold, **scores(validation[TARGET], fold_probability)})
    predictions = cohort.assign(argument_length_probability=probability)
    final = make_classifier().fit(cohort[FEATURES], cohort[TARGET])
    return predictions, fold_scores, final


def length_ranges(cohort):
    """Estimate observed rates by word-count quintile with discussion-bootstrap intervals."""
    bins = pd.qcut(cohort["word_count"], 5, labels=False, duplicates="drop")
    grouped = cohort.assign(length_bin=bins).groupby(["post_id", "length_bin"])[TARGET]
    totals = grouped.size().unstack(fill_value=0)
    positives = grouped.sum().unstack(fill_value=0)
    rng = np.random.default_rng(SEED)
    n_posts = len(totals)
    post_probabilities = np.full(n_posts, 1 / n_posts)
    replicates = []
    for _ in range(BOOTSTRAP_DRAWS // 50):
        weights = rng.multinomial(n_posts, post_probabilities, size=50)
        denominator = weights @ totals.to_numpy()
        numerator = weights @ positives.to_numpy()
        replicates.append(np.divide(
            numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0,
        ))
    lower, upper = np.nanquantile(np.concatenate(replicates), [0.025, 0.975], axis=0)
    result = (
        cohort.assign(length_bin=bins).groupby("length_bin")
        .agg(min_words=("word_count", "min"), max_words=("word_count", "max"),
             branches=(TARGET, "size"), later_deltas=(TARGET, "sum"))
        .reset_index(drop=True)
    )
    return result.assign(
        later_delta_rate=result["later_deltas"] / result["branches"], ci_lower=lower, ci_upper=upper,
    ).to_dict(orient="records")


def save_run(predictions, model, report):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUTPUT) as tmp:
        staging = Path(tmp)
        predictions.to_parquet(staging / "training_oof_predictions.parquet", index=False)
        joblib.dump(model, staging / "argument_length.joblib")
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    return destination


def main():
    eligible, cohort = load_cohort()
    with threadpool_limits(limits=1):
        predictions, folds, model = fit_classifier(cohort)
        ranges = length_ranges(cohort)
    report = {
        "model": "argument_length", "features": FEATURES, "split": "train",
        "observation_hours": 1, "followup_days": 7, "k_folds": 3, "C": 1,
        "eligible_branches": len(eligible), "eligible_later_deltas": int(eligible[TARGET].sum()),
        "excluded_unusable_openings": len(eligible) - len(cohort),
        "pooled_out_of_fold": scores(predictions[TARGET], predictions["argument_length_probability"]),
        "folds": folds, "word_count_ranges": ranges, "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    destination = save_run(predictions, model, report)
    print(json.dumps(report, indent=2))
    print(f"Saved argument-length run to {destination}")


if __name__ == "__main__":
    main()
