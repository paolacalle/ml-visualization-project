"""Audit retrospective engagement associations under two counting rules."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analyze_branch_engagement import (
    BUILDER_PATH, DEPTH, FEATURES, VOLUME, bin_volume, cluster_rates, draw_rates, fit_clustered_logit,
    load_module, stratified_rates,
)

ORIGINAL_RUN = "20261121T195352867471Z"
ORIGINAL_POSITIONS = "Award events omitted · original positions"
FULL = "All observed"
OMITTED = "Award events omitted"
RULES = {FULL: "", OMITTED: "_without_award"}
DEPTH_TERM = "log2_max_depth"
MEAN_TERM = "log2_mean_depth"
OP_TERM = "op_participated"


def binned_comparisons(frame):
    tables, stratified = [], []
    for label, suffix in RULES.items():
        volume = bin_volume(frame[f"total_replies{suffix}"])
        max_depth = frame[f"max_depth{suffix}"]
        depth = pd.cut(max_depth, [-1, 0, 1, 2, 4, np.inf], labels=DEPTH).astype(str)
        tables.extend([
            cluster_rates(frame, volume, VOLUME).assign(feature="Total descendants", counting_rule=label),
            cluster_rates(frame, depth, DEPTH).assign(feature="Maximum depth", counting_rule=label),
        ])
        stratified.append(stratified_rates(frame, volume, max_depth.ge(3), "Depth 3+").assign(counting_rule=label))
    return pd.concat(tables, ignore_index=True), pd.concat(stratified, ignore_index=True)


def correlations(frame):
    rows = []
    for label, suffix in RULES.items():
        values = frame[[f"{feature}{suffix}" for feature in FEATURES]].set_axis(FEATURES, axis=1)
        for subset, group in (
            ("All eligible branches", values),
            ("At least one descendant", values.loc[values["total_replies"].ge(1)]),
            ("At least two descendants", values.loc[values["total_replies"].ge(2)]),
        ):
            pairs = group.corr(method="spearman").rename_axis(index="feature_1", columns="feature_2").stack()
            rows.append(pairs.rename("spearman").reset_index().assign(
                counting_rule=label, subset=subset, branches=len(group),
            ))
    return pd.concat(rows, ignore_index=True)


def fit(frame, suffix, terms, specification, cohort):
    volume = bin_volume(frame[f"total_replies{suffix}"])
    order = [group for group in VOLUME if group in set(volume)]
    counts = pd.crosstab(volume, frame["verified_op_delta"])
    if 0 not in counts or 1 not in counts or not counts.gt(0).all().all():
        raise ValueError(f"At least one volume bin has no events or no nonevents in {specification}.")
    bins = pd.get_dummies(pd.Categorical(volume, categories=order), prefix="volume", drop_first=True, dtype=float)
    covariates = pd.DataFrame({
        OP_TERM: frame[f"op_replies{suffix}"].gt(0).to_numpy(dtype=float),
        DEPTH_TERM: np.log2(1 + frame[f"max_depth{suffix}"].to_numpy()),
        MEAN_TERM: np.log2(1 + frame[f"mean_depth{suffix}"].to_numpy()),
    })
    design = bins.join(covariates[list(terms)])
    # Remove the intercept by centering only for diagnosing collinearity.
    centered = design.to_numpy() - design.mean().to_numpy()
    standardized = centered / centered.std(axis=0)
    correlation = standardized.T @ standardized / len(standardized)
    vif = np.diag(np.linalg.inv(correlation))
    report = fit_clustered_logit(frame, design)
    report["vif"] = np.r_[np.nan, vif]
    return report.assign(
        specification=specification, cohort=cohort, volume_reference=order[0], branches=len(frame),
        events=int(frame["verified_op_delta"].sum()),
    )


def model_audit(frame):
    suffix = RULES[OMITTED]
    rows = []
    for name, terms in [
        ("Volume only", ()), ("Volume + OP", (OP_TERM,)), ("Volume + max depth", (DEPTH_TERM,)),
        ("Volume + OP + max depth", (OP_TERM, DEPTH_TERM)),
        ("Volume + OP + mean depth", (OP_TERM, MEAN_TERM)),
    ]:
        rows.append(fit(frame, suffix, terms, name, "All eligible · award events omitted"))

    # Full-count bins at low volume can lack deltas after restricting the sample.
    # Using at least three award-omitted descendants gives both outcomes in each
    # included volume bin under both counting rules. Omit OP in these
    # fits because every full-count delta includes an OP reply.
    common = frame.loc[frame["total_replies_without_award"].ge(3)].copy()
    for label, suffix in {**RULES, ORIGINAL_POSITIONS: "_original_positions"}.items():
        for name, terms in [
            ("Volume + max depth", (DEPTH_TERM,)), ("Volume + mean depth", (MEAN_TERM,)),
        ]:
            rows.append(fit(common, suffix, terms, name, f"Common sample · {label}"))
    return pd.concat(rows, ignore_index=True)


def plot_rates(rates):
    fig, axes = plt.subplots(2, 2, sharey="col", figsize=(11.1, 6.5), layout="constrained")
    for row, (label, color) in enumerate([(FULL, "#75828d"), (OMITTED, "#176d83")]):
        for ax, feature in zip(axes[row], ("Total descendants", "Maximum depth")):
            data = rates.loc[rates["counting_rule"].eq(label) & rates["feature"].eq(feature)]
            draw_rates(ax, data, color)
            ax.set_xticks(range(len(data)), [f"{r.group}\n{r.events:,}\n{r.branches:,}" for r in data.itertuples()],
                          fontsize=8)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_title(f"{feature} · {label}")
    fig.supxlabel("Feature bin · deltas · eligible branches (top to bottom)")
    fig.suptitle("Seven-day delta rates before and after omitting award events", fontsize=15, weight="bold")
    return fig


def audit_depth(frame, original):
    keys = ["post_id", "branch_id"]
    legacy = original.set_index(keys).loc[frame.set_index(keys).index]
    for feature in FEATURES:
        np.testing.assert_array_equal(frame[f"{feature}_original_positions"], legacy[f"{feature}_without_award"])
    counts = frame["total_replies_without_award"]
    corrected = frame["max_depth_without_award"]
    before = frame["max_depth_original_positions"]
    if not (corrected.le(counts) & corrected.le(before)).all():
        raise ValueError("The retained tree must be connected and no deeper than the original.")
    flagged = frame.loc[counts.between(1, 2) & before.ge(3), [
        *keys, "verified_op_delta", "total_replies", "total_replies_without_award",
        "max_depth", "max_depth_original_positions", "max_depth_without_award",
        "direct_replies_original_positions", "direct_replies_without_award",
    ]].copy()
    return flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", help="A branch engagement run with contracted depths.")
    args = parser.parse_args()
    builder = load_module("engagement_audit_builder", BUILDER_PATH)
    base = builder.DATA / "branch-engagement-analysis"
    if args.source_run:
        source = base / args.source_run
    else:
        sources = [
            path.parent for path in sorted(base.glob("*/results.json"))
            if json.loads(path.read_text()).get("award_omitted_depth_rule") == "contract_to_nearest_retained_ancestor"
        ]
        if not sources:
            raise FileNotFoundError("Run analyze_branch_engagement.py for the corrected depth table first.")
        source = sources[-1]
    metadata = json.loads((source / "results.json").read_text())
    if (metadata["split"] != "train" or metadata["followup_days"] != 7
            or metadata.get("award_omitted_depth_rule") != "contract_to_nearest_retained_ancestor"):
        raise ValueError("Expected the archived seven-day training cohort.")
    frame = pd.read_parquet(source / "engagement_features.parquet")
    original = pd.read_parquet(base / ORIGINAL_RUN / "engagement_features.parquet")
    flagged = audit_depth(frame, original)
    with threadpool_limits(limits=1):
        rates, stratified = binned_comparisons(frame)
        spearman = correlations(frame)
        models = model_audit(frame)
    fig = plot_rates(rates)
    out = base / "sensitivity"
    out.mkdir(exist_ok=True)
    destination = out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        flagged.to_csv(stage / "flagged_branch_depth_audit.csv", index=False)
        rates.to_csv(stage / "volume_depth_rates.csv", index=False)
        stratified.to_csv(stage / "depth_by_volume.csv", index=False)
        spearman.to_csv(stage / "engagement_spearman_sensitivity.csv", index=False)
        models.to_csv(stage / "regression_audit.csv", index=False)
        fig.savefig(stage / "award_omission_volume_depth.png", dpi=170, facecolor="white")
        (stage / "results.json").write_text(json.dumps({
            "source_run": source.name, "original_source_run": ORIGINAL_RUN,
            "split": "train", "branches": len(frame),
            "flagged_branches": len(flagged), "flagged_deltas": int(flagged["verified_op_delta"].sum()),
            "flagged_by_retained_count": flagged["total_replies_without_award"].value_counts().sort_index().to_dict(),
            "remaining_impossible_depths": int(frame["max_depth_without_award"].gt(frame["total_replies_without_award"]).sum()),
            "events": int(frame["verified_op_delta"].sum()), "common_sample_branches": int(frame["total_replies_without_award"].ge(3).sum()),
        }, indent=2) + "\n")
        stage.rename(destination)
    plt.close(fig)
    print(rates.round(4).to_string(index=False))
    print(models.loc[models["term"].isin([DEPTH_TERM, MEAN_TERM, OP_TERM])].round(4).to_string(index=False))
    print(f"Saved engagement audit to {destination}")


if __name__ == "__main__":
    main()
