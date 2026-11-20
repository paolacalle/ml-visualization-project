"""Fit M1a (M0a plus binary first-hour OP engagement) on the training split."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits


REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "project/data/processed"
BASELINE_DIR = DATA / "m0-baseline/1h"
OUTPUT_DIR = DATA / "m1a-engagement/1h"
BASELINE_PREDICTIONS = BASELINE_DIR / "training_oof_predictions.parquet"
BRANCHES = BASELINE_DIR / "train/branches.parquet"
TARGET = "delta_after_1h"
KEYS = ["post_id", "branch_id", TARGET]
FEATURES = ["non_op_comments_1h", "opening_delay_minutes", "opening_reply_length", "op_engaged_1h"]
SEED = 42
CANDIDATE_C = [0.01, 0.1, 1, 10]
BOOTSTRAP_DRAWS = 500


def load_training():
    audit = json.loads((BASELINE_DIR / "train/audit.json").read_text())
    settings = (audit["split"], audit["observation_hours"], audit["followup_days_after_cutoff"])
    if settings != ("train", 1, 7):
        raise ValueError(f"Expected the one-hour training cohort with seven-day follow-up, got {settings}.")

    branches = pd.read_parquet(BRANCHES, columns=[*KEYS, *FEATURES, "eligible"])
    train = branches.loc[branches["eligible"]].sort_values("branch_id").reset_index(drop=True)
    train[TARGET] = train[TARGET].astype(int)
    # Keep the engagement feature binary for all eligible branches.
    if train["op_engaged_1h"].isna().any():
        raise ValueError("The eligible cohort includes branches with unknown early OP engagement.")
    train["op_engaged_1h"] = train["op_engaged_1h"].astype(int)

    predictions = pd.read_parquet(BASELINE_PREDICTIONS, columns=[*KEYS, "fold", "M0a"])
    predictions = predictions.sort_values("branch_id").reset_index(drop=True)
    if not train["branch_id"].is_unique or not train[KEYS].equals(predictions[KEYS]):
        raise ValueError("M0a predictions and M1a training examples must have identical branch IDs and outcomes.")
    if (set(predictions["fold"]) != {1, 2, 3}
            or not predictions.groupby("post_id")["fold"].nunique().eq(1).all()
            or not predictions["M0a"].between(0, 1).all()):
        raise ValueError("Expected three valid M0a folds, with each discussion assigned to one fold.")
    return train, predictions


def make_search(seed):
    """Use the M0a preprocessing and regularization grid with the added feature."""
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(solver="lbfgs", max_iter=1000)),
    ])
    return GridSearchCV(
        model, {"logistic__C": CANDIDATE_C}, scoring="average_precision",
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed),
        refit=True, error_score="raise",
    )


def fit_m1a(train, predictions):
    fold_metrics = []
    scores = np.full(len(train), np.nan)
    # Score exactly the branches M0a scored in each outer fold. Tune inside its fitting rows.
    for fold in (1, 2, 3):
        evaluate = predictions["fold"].eq(fold)
        fitting, validation = train.loc[~evaluate], train.loc[evaluate]
        search = make_search(SEED + fold)
        search.fit(fitting[FEATURES], fitting[TARGET], groups=fitting["post_id"])
        probability = search.predict_proba(validation[FEATURES])[:, 1]
        scores[evaluate.to_numpy()] = probability
        fold_metrics.append({
            "fold": fold, "branches": len(validation), "later_deltas": int(validation[TARGET].sum()),
            "C": search.best_params_["logistic__C"],
            "M0a_ap": average_precision_score(validation[TARGET], predictions.loc[evaluate, "M0a"]),
            "M1a_ap": average_precision_score(validation[TARGET], probability),
        })
    if not np.isfinite(scores).all():
        raise ValueError("Some branches did not receive M1a validation predictions.")

    final_search = make_search(SEED)
    final_search.fit(train[FEATURES], train[TARGET], groups=train["post_id"])
    return predictions.assign(M1a=scores), fold_metrics, final_search


def evaluate(y, probability):
    top = np.argsort(-probability, kind="stable")[:100]
    return {
        "average_precision": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "later_deltas_in_top_100": int(y[top].sum()),
    }


def prepare_weighted_ap(y, probability, post_codes):
    """Sort once and group score ties before resampling whole discussions."""
    order = np.argsort(-probability, kind="stable")
    thresholds = np.r_[np.flatnonzero(np.diff(probability[order])), len(y) - 1]
    return y[order], post_codes[order], thresholds


def weighted_ap(prepared, post_weights):
    y, post_codes, thresholds = prepared
    weights = post_weights[post_codes]
    true_positives = np.cumsum(weights * y)[thresholds]
    predicted_positives = np.cumsum(weights)[thresholds]
    precision = np.divide(
        true_positives, predicted_positives, out=np.zeros(len(thresholds)), where=predicted_positives > 0,
    )
    return np.dot(np.diff(true_positives, prepend=0), precision) / true_positives[-1]


def paired_ap_interval(predictions):
    """Calculate a percentile interval for M1a AP minus M0a AP with paired post draws."""
    post_codes, posts = pd.factorize(predictions["post_id"])
    y = predictions[TARGET].to_numpy()
    baseline, m1a = [
        prepare_weighted_ap(y, predictions[name].to_numpy(), post_codes) for name in ("M0a", "M1a")
    ]
    positives_by_post = np.bincount(post_codes, weights=y, minlength=len(posts))
    rng = np.random.default_rng(SEED)
    probabilities = np.full(len(posts), 1 / len(posts))
    differences = []
    for _ in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(len(posts), probabilities)
        if np.dot(positives_by_post, weights) > 0:
            differences.append(weighted_ap(m1a, weights) - weighted_ap(baseline, weights))
    return np.quantile(differences, [0.025, 0.975]).tolist(), len(differences)


def save_run(predictions, final_search, report):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = OUTPUT_DIR / name
    with TemporaryDirectory(dir=OUTPUT_DIR) as temporary:
        staging = Path(temporary)
        predictions.to_parquet(staging / "training_oof_predictions.parquet", index=False)
        joblib.dump(final_search.best_estimator_, staging / "M1a.joblib")
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        # A completed run becomes visible as a unit. An existing nonempty run cannot be replaced.
        staging.rename(destination)
    return destination


def main():
    train, baseline = load_training()
    with threadpool_limits(limits=1):
        predictions, fold_metrics, final_search = fit_m1a(train, baseline)
        delta_interval, draws = paired_ap_interval(predictions)
    y = predictions[TARGET].to_numpy()
    metrics = {name: evaluate(y, predictions[name].to_numpy()) for name in ("M0a", "M1a")}
    with BASELINE_PREDICTIONS.open("rb") as source:
        baseline_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "model": "M1a", "baseline": "M0a", "split": "train",
        "observation_hours": 1, "followup_days": 7, "features": FEATURES,
        "seed": SEED, "C_grid": CANDIDATE_C, "final_C": final_search.best_params_["logistic__C"],
        "scikit_learn_version": sklearn.__version__, "M0a_oof_sha256": baseline_sha256,
        "branches": len(train), "later_deltas": int(y.sum()), "prevalence": y.mean(),
        "out_of_fold": metrics, "folds": fold_metrics,
        "M1a_minus_M0a_ap": metrics["M1a"]["average_precision"] - metrics["M0a"]["average_precision"],
        "paired_post_bootstrap_95_ci": delta_interval, "bootstrap_draws": draws,
    }
    destination = save_run(predictions, final_search, report)
    print(json.dumps(report, indent=2))
    print(f"Saved M1a run to {destination}")


if __name__ == "__main__":
    main()
