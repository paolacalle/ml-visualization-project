"""Shared training-pair assignments, statistics, and grouped feature evaluation."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NOTEBOOKS = Path(__file__).resolve().parents[1]
DATA = NOTEBOOKS.parent / "data/processed/linguistic-features"
PAIR_SCRIPT = NOTEBOOKS / "paired-argument-length/compare_paired_length.py"
SEED = 42
FOLDS = 3
TIE_TOLERANCE = 1e-9


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assign_pair_sides(pairs):
    """Assign winners to A or B with the frozen seed and pair ordering."""
    pairs = pairs.sort_values("pair_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    a_wins = rng.integers(0, 2, len(pairs))
    assignments = pairs[["pair_id", "post_id"]].assign(a_wins=a_wins)
    for side, choose_winner in (("a", a_wins == 1), ("b", a_wins == 0)):
        for column in ("comment_id", "word_count"):
            assignments[f"{side}_{column}"] = np.where(
                choose_winner, pairs[f"winner_{column}"], pairs[f"comparison_{column}"]
            )
    return assignments


def load_training_pairs():
    """Pin A/B orientation and grouped folds once for all feature experiments."""
    pair_analysis = import_script("published_pairs", PAIR_SCRIPT)
    published, exclusions, pairs = pair_analysis.load_pairs()
    assignments = assign_pair_sides(pairs)
    assignments["fold"] = 0
    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, validation) in enumerate(cv.split(
        assignments, assignments["a_wins"], groups=assignments["post_id"]
    ), start=1):
        assignments.loc[validation, "fold"] = fold
    if (not assignments.groupby("post_id")["fold"].nunique().eq(1).all()
            or set(assignments["fold"]) != set(range(1, FOLDS + 1))):
        raise ValueError("Pair folds must stay grouped by OP discussion.")

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "training_pair_assignments.parquet"
    if path.exists():
        # Check rather than silently replacing the shared split artifact.
        pd.testing.assert_frame_equal(assignments, pd.read_parquet(path))
    else:
        with TemporaryDirectory(dir=DATA) as directory:
            temporary = Path(directory) / path.name
            assignments.to_parquet(temporary, index=False)
            temporary.replace(path)
    return assignments, {"published_pairs": published, "opening_exclusions": exclusions}


def paired_higher(frame, difference):
    """Report winner-higher share among unequal scores and a discussion bootstrap CI."""
    high = difference.gt(TIE_TOLERANCE)
    low = difference.lt(-TIE_TOLERANCE)
    non_ties = frame.loc[high | low, ["post_id"]]
    counts = {
        "pairs": len(frame), "winner_higher": int(high.sum()), "winner_lower": int(low.sum()),
        "ties": int((~(high | low)).sum()),
    }
    if non_ties.empty:
        return {**counts, "winner_higher_rate": None, "ci_95": [None, None]}
    pair_analysis = import_script("paired_length_bootstrap", PAIR_SCRIPT)
    interval = pair_analysis.rate_intervals(
        non_ties.assign(bin="all"), "bin", high.loc[non_ties.index],
    )["all"]
    return {**counts, "winner_higher_rate": float(high.sum() / len(non_ties)), "ci_95": interval}


def evaluate_feature(frame, a, b):
    """Predict whether A wins from a single signed difference on the shared folds."""
    differences = (frame[a] - frame[b]).to_frame("difference")
    return evaluate_differences(frame, differences)


def make_paired_classifier():
    """Keep scaling inside the fit for each fold."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))


def evaluate_differences(frame, differences):
    """Fit on the supplied signed differences using the shared discussion folds."""
    y = frame["a_wins"]
    scores = np.full(len(frame), np.nan)
    for fold in range(1, FOLDS + 1):
        validation = frame["fold"].eq(fold)
        train = ~validation
        model = make_paired_classifier()
        model.fit(differences.loc[train], y.loc[train])
        scores[validation.to_numpy()] = model.predict_proba(differences.loc[validation])[:, 1]
    return scores, {
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "a_win_prevalence": float(y.mean()),
    }
