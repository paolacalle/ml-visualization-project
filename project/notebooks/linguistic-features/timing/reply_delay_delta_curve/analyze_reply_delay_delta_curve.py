"""Summarize fixed-window delta rates in the training branch population."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paired_evaluation as paired

DATA = paired.DATA.parent
OUT = DATA / "reply-delay-delta-curve"
HOUR = 3600
FOLLOWUP = 7 * 24 * HOUR
BINS = [0, 1, 3, 6, 12, 24, 48, 72, 7 * 24, np.inf]
LABELS = ["0–1 h", ">1–3 h", ">3–6 h", ">6–12 h", ">12–24 h", ">1–2 d", ">2–3 d", ">3–7 d", ">7 d"]
BOOTSTRAP_DRAWS = 1000
KEYS = ["post_id", "branch_id"]


def window(frame, boundary, observation_hours):
    """Apply the original delta timing notebook's eligibility rules for one window."""
    cutoff = frame["branch_created_utc"] + observation_hours * HOUR
    end = cutoff + FOLLOWUP
    award = frame["first_verified_delta_utc"].fillna(np.inf)
    confirmation = frame["first_confirmation_utc"].fillna(np.inf)
    within = frame["branch_static_valid"] & end.le(boundary)
    clock_disagreement = award.le(end) & (
        confirmation.gt(end) | (award.le(cutoff) & confirmation.gt(cutoff))
    )
    eligible = within & ~clock_disagreement & award.gt(cutoff)
    for column in ("earliest_unlinked_utc", "earliest_unresolved_award_utc"):
        eligible &= frame[column].fillna(np.inf).gt(end)
    return eligible, eligible & award.le(end)


def rates(frame):
    by_thread = frame.groupby(["post_id", "delay_bin"], observed=True)["delta"].agg(["size", "sum"])
    counts = by_thread["size"].unstack(fill_value=0).reindex(columns=LABELS, fill_value=0).to_numpy(dtype=float)
    events = by_thread["sum"].unstack(fill_value=0).reindex(columns=LABELS, fill_value=0).to_numpy(dtype=float)
    rng = np.random.default_rng(paired.SEED)
    p = np.full(len(counts), 1 / len(counts))
    draws = []
    for _ in range(BOOTSTRAP_DRAWS // 25):
        weights = rng.multinomial(len(p), p, size=25).astype(float)
        totals, awards = weights @ counts, weights @ events
        draws.append(np.divide(awards, totals, out=np.full_like(totals, np.nan), where=totals > 0))
    lo, hi = np.nanquantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    total, awards = counts.sum(axis=0), events.sum(axis=0)
    return pd.DataFrame({
        "delay_bin": LABELS, "replies": total.astype(int), "deltas": awards.astype(int),
        "delta_rate": np.divide(awards, total, out=np.full_like(total, np.nan), where=total > 0),
        "ci_lower": lo, "ci_upper": hi,
    })


def compare_delay_encodings(frame):
    """Compare continuous delay with binary thresholds on exactly the same cohort."""
    hours = frame["opening_delay_minutes"] / 60
    features = {"Continuous log1p(minutes)": np.log1p(frame["opening_delay_minutes"]).to_numpy()}
    features.update({f"Delay > {hour} h": hours.gt(hour).to_numpy(dtype=float) for hour in (6, 24, 72)})
    y = frame["delta"].to_numpy()
    groups = frame["post_id"]
    cv = StratifiedGroupKFold(n_splits=paired.FOLDS, shuffle=True, random_state=paired.SEED)
    folds = list(cv.split(frame, y, groups))
    fit_groups = [set(groups.iloc[train]).isdisjoint(groups.iloc[valid]) for train, valid in folds]
    if not all(fit_groups):
        raise ValueError("OP discussions overlap between fitting and validation.")
    rows = []
    for label, values in features.items():
        x = values.reshape(-1, 1)
        predictions = np.full(len(frame), np.nan)
        for train, valid in folds:
            model = paired.make_paired_classifier().fit(x[train], y[train])
            predictions[valid] = model.predict_proba(x[valid])[:, 1]
        rows.append({
            "encoding": label, "roc_auc": roc_auc_score(y, predictions),
            "average_precision": average_precision_score(y, predictions),
            "log_loss": log_loss(y, predictions),
        })
    return pd.DataFrame(rows)


def main():
    timing = DATA / "delta-timing-analysis/train"
    landmark = DATA / "m0-baseline/1h/train"
    audit = json.loads((timing / "audit.json").read_text())
    landmark_audit = json.loads((landmark / "audit.json").read_text())
    boundary = audit["followup_boundary_utc"]
    if (audit["split"] != "train" or landmark_audit["split"] != "train"
            or landmark_audit["followup_boundary_utc"] != boundary):
        raise ValueError("Expected the same training follow-up boundary in both branch files.")
    frame = pd.read_parquet(timing / "branches.parquet")
    reference = pd.read_parquet(landmark / "branches.parquet", columns=[
        *KEYS, "opening_delay_minutes", "branch_created_utc", "eligible", "delta_after_1h",
    ])
    frame = frame.merge(reference, on=KEYS, validate="one_to_one", suffixes=("", "_1h"))
    if len(frame) != audit["counts"]["top_level_branches"] or len(frame) != len(reference):
        raise ValueError("Branch snapshots must contain the same openings.")
    pd.testing.assert_series_equal(frame["branch_created_utc"], frame["branch_created_utc_1h"], check_names=False)

    profiles, counts = [], {}
    for hours in (0, 1, 24):
        eligible, positive = window(frame, boundary, hours)
        if hours in (1, 24):
            suffix = "_1h" if hours == 1 else ""
            np.testing.assert_array_equal(eligible, frame[f"eligible{suffix}"])
            np.testing.assert_array_equal(positive, frame[f"delta_after_{hours}h"].eq(1))
        if hours == 24:
            continue
        usable = eligible & frame["opening_delay_minutes"].ge(0) & np.isfinite(frame["opening_delay_minutes"])
        cohort = frame.loc[usable, [*KEYS, "opening_delay_minutes"]].assign(delta=positive.loc[usable].astype(int))
        cohort["delay_bin"] = pd.cut(
            cohort["opening_delay_minutes"] / 60, BINS, labels=LABELS, include_lowest=True,
        )
        profiles.append(rates(cohort).assign(observation_hours=hours))
        counts[str(hours)] = {
            "eligible_before_delay_check": int(eligible.sum()), "missing_delay": int((eligible & ~usable).sum()),
            "replies": len(cohort), "discussions": cohort["post_id"].nunique(), "deltas": int(cohort["delta"].sum()),
        }
        if hours == 0:
            comparisons = compare_delay_encodings(cohort)

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        pd.concat(profiles, ignore_index=True).to_parquet(staging / "delay_curve.parquet", index=False)
        comparisons.to_csv(staging / "delay_encoding_cv.csv", index=False)
        (staging / "results.json").write_text(json.dumps({
            "split": "train", "branches": len(frame), "followup_days": 7, "cohorts": counts,
            "bootstrap_draws": BOOTSTRAP_DRAWS, "folds": paired.FOLDS, "seed": paired.SEED,
        }, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(counts, indent=2))
    print(comparisons.to_string(index=False))
    print(f"Saved reply delay curves to {destination}")


if __name__ == "__main__":
    main()
