"""Describe observed engagement and verified OP awards in a common seven-day window."""

from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parents[1]
BUILDER_PATH = HERE.parent / "op_engagement_study/build_branch_dataset.py"
WINDOW_PATH = NOTEBOOKS / "linguistic-features/timing/reply_delay_delta_curve/analyze_reply_delay_delta_curve.py"
SEED = 42
DRAWS = 1000
FEATURES = ("op_replies", "direct_replies", "total_replies", "max_depth", "mean_depth", "max_op_depth")
LABELS = ("OP replies", "Direct replies", "Total descendants", "Maximum depth", "Mean depth", "Maximum OP depth")
VOLUME = ["0", "1", "2", "3–5", "6–10", "11–20", "21+"]
DEPTH = ["0", "1", "2", "3–4", "5+"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort(builder, windows):
    reference = builder.DATA / "delta-timing-analysis/train"
    report = json.loads((reference / "audit.json").read_text())
    metadata = builder.manifest(builder.METADATA)
    if report["split"] != "train" or report["raw_archive_sha256"] != metadata["raw_archive"]["sha256"]:
        raise ValueError("The existing branch snapshot and metadata must use the same training archive.")
    branches = pd.read_parquet(reference / "branches.parquet")
    boundary = report["followup_boundary_utc"]
    eligible, positive = windows.window(branches, boundary, 0)
    saved_eligible, saved_positive = windows.window(branches, boundary, 24)
    np.testing.assert_array_equal(saved_eligible, branches["eligible"])
    np.testing.assert_array_equal(saved_positive, branches["delta_after_24h"].eq(1))
    start = branches["branch_created_utc"]
    end = start + windows.FOLLOWUP
    award = branches["first_verified_delta_utc"].fillna(np.inf)
    confirmation = branches["first_confirmation_utc"].fillna(np.inf)
    reasons = np.select([
        ~branches["branch_static_valid"], end.gt(boundary),
        award.le(end) & (confirmation.gt(end) | (award.le(start) & confirmation.gt(start))),
        award.le(start), branches["earliest_unlinked_utc"].fillna(np.inf).le(end),
        branches["earliest_unresolved_award_utc"].fillna(np.inf).le(end),
    ], [
        "Outside valid challenger/bot-period/timestamp cohort", "Insufficient seven-day follow-up",
        "Award and confirmation cross a window boundary", "Award by branch opening",
        "Incomplete parent path during follow-up", "Uncertain award attribution during follow-up",
    ], default="Eligible")
    np.testing.assert_array_equal(reasons == "Eligible", eligible)
    selected = branches.loc[eligible, [
        "post_id", "branch_id", "branch_created_utc", "first_verified_delta_utc",
    ]].assign(followup_end_utc=end[eligible], verified_op_delta=positive[eligible].astype(int))
    events = pd.read_parquet(reference / "verified_awards.parquet")
    omitted = set(events["award_comment_id"].dropna()) | set(events["confirmation_comment_id"].dropna())
    return selected, metadata, omitted, pd.Series(reasons).value_counts().to_dict()


def summarize_tree(post_id, op, comments, openings, omit):
    roots, root_of, depth = openings
    retained_depth = dict.fromkeys(roots, 0)
    # Count from the same parent-first traversal as the original branch builder.
    counts, without_awards, original_positions = defaultdict(list), defaultdict(list), defaultdict(list)
    skipped = 0
    for cid, root in root_of.items():
        if cid == root or root not in roots:
            continue
        comment = comments[cid]
        start, end = roots[root]
        time = builder_timestamp(comment["time"])
        if time is None or not start <= time <= end:
            continue
        parent = comment["parent"]
        if parent not in retained_depth:
            skipped += 1
            continue
        # An omitted node passes its nearest retained ancestor's depth to its children.
        # A retained node then adds one edge in the contracted tree.
        keep = cid not in omit
        retained_depth[cid] = retained_depth[parent] + int(keep)
        value = (depth[cid], comment["author"] == op)
        counts[root].append(value)
        if keep:
            without_awards[root].append((retained_depth[cid], value[1]))
            original_positions[root].append(value)
    rows = []
    for root in roots:
        base = depth_features(counts[root])
        sensitivity = depth_features(without_awards[root])
        rows.append((post_id, root, *base, *sensitivity, *depth_features(original_positions[root])))
    return rows, skipped


def builder_timestamp(value):
    return value if value is not None and np.isfinite(value) else None


def depth_features(records):
    if not records:
        return (0, 0, 0, 0, 0.0, 0)
    depths, is_op = zip(*records)
    count = len(depths)
    return (sum(is_op), depths.count(1), count, max(depths), sum(depths) / count,
            max((depth for depth, op in records if op), default=0))


def extract(builder, branches, metadata, omit):
    columns = ["source_id", "task", "original_split"]
    full = next(s["source_id"] for s in builder.rows_for(metadata, "sources", columns, split="train")
                if s["task"] == "all" and s["original_split"] == "train")
    groups = {
        post_id: dict(zip(group["branch_id"], zip(group["branch_created_utc"], group["followup_end_utc"])))
        for post_id, group in branches.groupby("post_id", sort=False)
    }
    posts = {
        p["post_id"]: builder.speaker(p["author"])
        for p in builder.rows_for(metadata, "posts", ["post_id", "author", "source_id"], split="train")
        if p["source_id"] == full and p["post_id"] in groups
    }
    comments = defaultdict(dict)
    columns = ["post_id", "comment_id", "author", "parent_comment_id", "original_parent_id",
               "created_utc", "source_id", "content_available"]
    for row in builder.rows_for(metadata, "comments", columns, split="train"):
        pid = row["post_id"]
        if pid in groups and row["source_id"] == full and row["content_available"]:
            comments[pid][row["comment_id"]] = {
                "author": builder.speaker(row["author"]),
                "parent": row["parent_comment_id"] or row["original_parent_id"], "time": row["created_utc"],
            }
    output, skipped = [], 0
    for pid, roots in groups.items():
        _, root_of, depth = builder.branch_roots(pid, comments[pid])
        if any(root_of.get(root) != root or comments[pid][root]["time"] != start
               for root, (start, _) in roots.items()):
            raise ValueError(f"Roots or opening times in {pid} differ from the archived cohort.")
        rows, missing_path = summarize_tree(pid, posts[pid], comments[pid], (roots, root_of, depth), omit)
        output.extend(rows)
        skipped += missing_path
    features = pd.DataFrame(output, columns=["post_id", "branch_id", *FEATURES,
                                           *(f"{f}_without_award" for f in FEATURES),
                                           *(f"{f}_original_positions" for f in FEATURES)])
    cleaned = branches.merge(features, on=["post_id", "branch_id"], validate="one_to_one")
    if len(cleaned) != len(branches):
        raise ValueError("Some eligible branches were not reconstructed.")
    return cleaned, skipped


def bin_volume(counts):
    return pd.cut(counts, [-1, 0, 1, 2, 5, 10, 20, np.inf], labels=VOLUME).astype(str)


def cluster_rates(frame, category, order):
    aggregated = frame.assign(group=category).groupby(["post_id", "group"], observed=True)["verified_op_delta"].agg(["size", "sum"])
    n = aggregated["size"].unstack(fill_value=0).reindex(columns=order, fill_value=0).to_numpy(dtype=float)
    k = aggregated["sum"].unstack(fill_value=0).reindex(columns=order, fill_value=0).to_numpy(dtype=float)
    total = n.sum(axis=0)
    events = k.sum(axis=0)
    rng = np.random.default_rng(SEED)
    probabilities = np.full(len(n), 1 / len(n))
    draws = []
    for _ in range(DRAWS // 25):
        weights = rng.multinomial(len(n), probabilities, size=25).astype(float)
        sample_n = weights @ n
        draws.append(np.divide(weights @ k, sample_n, out=np.full_like(sample_n, np.nan), where=sample_n > 0))
    replicates = np.concatenate(draws)
    limits = np.full((2, len(order)), np.nan)
    informative = (events > 0) & (events < total)
    limits[:, informative] = np.nanquantile(replicates[:, informative], [0.025, 0.975], axis=0)
    return pd.DataFrame({
        "group": order, "branches": total.astype(int), "events": events.astype(int),
        "discussions": (n > 0).sum(axis=0),
        "rate": np.divide(events, total, out=np.full_like(events, np.nan), where=total > 0),
        "ci_low": limits[0], "ci_high": limits[1],
    })


def stratified_rates(frame, volume, indicator, kind):
    flag = indicator.astype(int).astype(str)
    keys = [f"{group}|{side}" for group in VOLUME for side in ("0", "1")]
    result = cluster_rates(frame, volume + "|" + flag, keys)
    result[["volume", "indicator"]] = result["group"].str.split("|", regex=False, expand=True)
    result["comparison"] = kind
    return result.drop(columns="group")


def adjusted_model(frame):
    suffix = "_without_award"
    volume = pd.Categorical(bin_volume(frame[f"total_replies{suffix}"]), categories=VOLUME)
    bins = pd.get_dummies(volume, prefix="volume", drop_first=True, dtype=float)
    explanatory = bins.assign(
        op_participated=frame[f"op_replies{suffix}"].gt(0).astype(float),
        log2_max_depth=np.log2(1 + frame[f"max_depth{suffix}"]),
    )
    return fit_clustered_logit(frame, explanatory).assign(counts_exclude_award_and_confirmation=True)


def fit_clustered_logit(frame, explanatory):
    """Fit an unpenalized logit and compute OP-clustered sandwich intervals."""
    y = frame["verified_op_delta"].to_numpy()
    x = explanatory.to_numpy()
    fit = LogisticRegression(C=np.inf, max_iter=1000, tol=1e-9).fit(x, y)
    predictor = np.column_stack((np.ones(len(x)), x))
    probability = fit.predict_proba(x)[:, 1]
    bread = np.linalg.inv((predictor.T * (probability * (1 - probability))) @ predictor)
    cluster_scores = pd.DataFrame(predictor * (y - probability)[:, None]).groupby(frame["post_id"].to_numpy()).sum()
    count, terms = predictor.shape
    groups = len(cluster_scores)
    correction = groups / (groups - 1) * (count - 1) / (count - terms)
    covariance = correction * bread @ (cluster_scores.to_numpy().T @ cluster_scores.to_numpy()) @ bread
    coefficient = np.r_[fit.intercept_, fit.coef_[0]]
    se = np.sqrt(np.diag(covariance))
    return pd.DataFrame({
        "term": ["intercept", *explanatory.columns], "odds_ratio": np.exp(coefficient),
        "ci_low": np.exp(coefficient - 1.96 * se), "ci_high": np.exp(coefficient + 1.96 * se),
    })


def distributions(frame):
    rows = []
    for outcome, group in frame.groupby("verified_op_delta"):
        for name in FEATURES:
            values = group[name]
            rows.append({
                "verified_op_delta": outcome, "feature": name, "branches": len(group),
                "mean": values.mean(), "median": values.median(), "zero_share": values.eq(0).mean(),
                "p90": values.quantile(0.9), "p99": values.quantile(0.99),
            })
    return pd.DataFrame(rows)


def distribution_plot(frame):
    fig, axs = plt.subplots(2, 3, figsize=(11.2, 6.1), sharey=True, layout="constrained")
    palette = [(0, "No verified delta", "#75828d"), (1, "Verified OP delta", "#176d83")]
    groups = {outcome: group for outcome, group in frame.groupby("verified_op_delta")}
    for ax, field, label in zip(axs.flat, FEATURES, LABELS):
        for result, outcome_label, color in palette:
            values = groups[result][field]
            x = np.sort(values.unique())
            survival = 1 - (values.value_counts(sort=False).reindex(x).cumsum() / len(values)).to_numpy()
            ax.step(x, survival, where="post", color=color, label=outcome_label, lw=1.7)
        ax.set(title=label, xscale="symlog", ylim=(-0.02, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axs[0, 0].legend(frameon=False, fontsize=8)
    axs[0, 0].set_ylabel("Branches above x")
    axs[1, 0].set_ylabel("Branches above x")
    fig.supxlabel("Feature value (log-spaced above 2)")
    fig.suptitle("Engagement distributions by verified OP delta", fontsize=15, weight="bold")
    return fig


def draw_rates(ax, table, color, label=None, offset=0, marker="o"):
    x = np.arange(len(table)) + offset
    valid = table["branches"].gt(0)
    ax.vlines(x, table["ci_low"], table["ci_high"], color=color, lw=1.8)
    ax.scatter(x[valid], table.loc[valid, "rate"], marker=marker, s=30, color=color, label=label, zorder=3)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylabel("Verified OP delta rate")
    ax.set_axisbelow(True)
    ax.margins(y=0.15)


def marginal_plot(volume, depth):
    fig, axs = plt.subplots(1, 2, figsize=(10.9, 4.2), layout="constrained")
    for ax, table, title in zip(axs, (volume, depth), ("Total descendants", "Maximum thread depth")):
        draw_rates(ax, table, "#176d83")
        ax.set_xticks(range(len(table)), [f"{r.group}\n{r.events:,}/{r.branches:,}" for r in table.itertuples()],
                      fontsize=8)
        ax.set(title=title, xlabel="Feature bin · deltas / branches")
    fig.suptitle("Seven-day delta rates by volume and depth", fontsize=15, weight="bold")
    return fig


def stratified_plot(stratified):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False, layout="constrained")
    groups = [
        ("op_participation_without_award", "OP participation, with award events omitted", ("No OP replies", "OP replied")),
        ("deep_branch", "Depth, with all observed comments", ("Depth 0–2", "Depth 3+")),
    ]
    for ax, (name, title, labels) in zip(axs, groups):
        subset = stratified.loc[stratified["comparison"].eq(name)]
        series = [subset.loc[subset["indicator"].eq(str(i))] for i in (0, 1)]
        draw_rates(ax, series[0], "#75828d", labels[0], offset=-0.10, marker="s")
        draw_rates(ax, series[1], "#176d83", labels[1], offset=0.10)
        counts = zip(series[0]["branches"], series[1]["branches"])
        ax.set_xticks(range(len(VOLUME)), [
            f"{volume}\n{left:,}\n{right:,}" for volume, (left, right) in zip(VOLUME, counts)
        ], fontsize=8)
        ax.set(title=title, xlabel="Total descendants · group sizes in legend order")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Engagement patterns within reply-count groups", fontsize=15, weight="bold")
    return fig


def main():
    builder = load_module("engagement_branch_builder", BUILDER_PATH)
    windows = load_module("engagement_fixed_window", WINDOW_PATH)
    branches, metadata, events, eligibility = cohort(builder, windows)
    frame, skipped = extract(builder, branches, metadata, events)
    count, maximum = frame["total_replies_without_award"], frame["max_depth_without_award"]
    if not (maximum.le(count) & maximum.eq(0).eq(count.eq(0))).all():
        raise ValueError("Retained-tree depth must agree with retained descendant counts.")
    volume = bin_volume(frame["total_replies"])
    depth = pd.cut(frame["max_depth"], [-1, 0, 1, 2, 4, np.inf], labels=DEPTH).astype(str)
    volume_without_awards = bin_volume(frame["total_replies_without_award"])
    with threadpool_limits(limits=1):
        marginal_volume = cluster_rates(frame, volume, VOLUME)
        marginal_depth = cluster_rates(frame, depth, DEPTH)
        marginal = [marginal_volume.assign(feature="total_replies"), marginal_depth.assign(feature="max_depth")]
        for feature in ("op_replies", "direct_replies"):
            marginal.append(cluster_rates(frame, bin_volume(frame[feature]), VOLUME).assign(feature=feature))
        for feature in ("mean_depth", "max_op_depth"):
            levels = ["0", ">0–1", ">1–2", ">2–4", ">4"]
            category = pd.cut(frame[feature], [-1, 0, 1, 2, 4, np.inf], labels=levels).astype(str)
            marginal.append(cluster_rates(frame, category, levels).assign(feature=feature))
        stratified = pd.concat([
            stratified_rates(frame, volume, frame["op_replies"].gt(0), "op_participation"),
            stratified_rates(frame, volume_without_awards, frame["op_replies_without_award"].gt(0),
                             "op_participation_without_award"),
            stratified_rates(frame, volume, frame["max_depth"].ge(3), "deep_branch"),
        ], ignore_index=True)
        # The original award comment can define both OP participation and the outcome.
        adjusted = adjusted_model(frame)
    winners = frame.loc[frame["verified_op_delta"].eq(1)]
    report = {
        "split": "train", "followup_days": 7, "seed": SEED, "bootstrap_draws": DRAWS,
        "award_omitted_depth_rule": "contract_to_nearest_retained_ancestor",
        "eligibility": eligibility, "branches": len(frame), "discussions": frame["post_id"].nunique(),
        "deltas": len(winners), "window_descendants_without_in_window_parent": skipped,
        "positive_branches_with_no_op_reply": int(winners["op_replies"].eq(0).sum()),
        "positive_branches_with_no_op_reply_when_omitting_awards": int(winners["op_replies_without_award"].eq(0).sum()),
        "raw_archive_sha256": metadata["raw_archive"]["sha256"],
    }
    output = builder.DATA / "branch-engagement-analysis"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        frame.to_parquet(stage / "engagement_features.parquet", index=False)
        distributions(frame).to_csv(stage / "engagement_distributions.csv", index=False)
        frame[list(FEATURES)].corr(method="spearman").to_csv(stage / "spearman_correlations.csv")
        pd.concat(marginal, ignore_index=True).to_csv(stage / "marginal_rates.csv", index=False)
        stratified.to_csv(stage / "stratified_rates.csv", index=False)
        adjusted.to_csv(stage / "adjusted_associations.csv", index=False)
        for fig, filename in [
            (distribution_plot(frame), "engagement_distributions.png"),
            (marginal_plot(marginal_volume, marginal_depth), "delta_rate_by_volume_and_depth.png"),
            (stratified_plot(stratified), "op_participation_and_depth_by_volume.png"),
        ]:
            fig.savefig(stage / filename, dpi=170, facecolor="white")
            plt.close(fig)
        (stage / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(destination)
    print(json.dumps(report, indent=2))
    print(adjusted.round(3).to_string(index=False))
    print(f"Saved engagement analysis to {destination}")


if __name__ == "__main__":
    main()
