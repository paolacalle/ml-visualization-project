"""Measure Phase 3A argument- and pair-level stability and save the figures."""

import argparse
import json
from pathlib import Path
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_boundary_experiment import PHASE2, RESULTS, SIZES, TRIALS
from run_chunk_size_experiment import sha256
from run_pilot import DATA

TIE_TOLERANCE = 1e-9
LENGTH_EDGES = [256, 384, 510, 768, 1020, float("inf")]
LENGTH_LABELS = ["257–384", "385–510", "511–768", "769–1,020", ">1,020"]
DISPLACEMENT_EDGES = [0, 16, 32, 64, 128, 256, float("inf")]
DISPLACEMENT_LABELS = [">0–16", ">16–32", ">32–64", ">64–128", ">128–256", ">256"]
COLORS = {256: "#19727a", 510: "#8c4b83"}


def spearman(a, b):
    a_rank, b_rank = a.rank(), b.rank()
    return a_rank.corr(b_rank) if a_rank.nunique() > 1 and b_rank.nunique() > 1 else np.nan


def argument_results(design, trials, baseline):
    fixed = baseline[["politeness_score", "predicted_class"]].rename(
        columns={"politeness_score": "fixed_score", "predicted_class": "fixed_class"},
    )
    trials = trials.join(fixed, on="opening_id", validate="many_to_one")
    if trials[["fixed_score", "fixed_class"]].isna().any().any():
        raise ValueError("A randomized opening is missing its Phase 2 reference.")
    trials["signed_change"] = trials["politeness_score"] - trials["fixed_score"]
    trials["absolute_change"] = trials["signed_change"].abs()
    trials["class_agrees"] = trials["predicted_class"].eq(trials["fixed_class"])
    trials["displacement_band"] = pd.cut(
        trials["mean_boundary_displacement"], DISPLACEMENT_EDGES, labels=DISPLACEMENT_LABELS,
    )
    trials["fraction_displaced_band"] = pd.cut(
        trials["fraction_of_available_displacement"], [0, .25, .5, .75, 1],
    )
    groups = trials.groupby("opening_id")
    opening = design.join(groups.agg(
        mean_absolute_change=("absolute_change", "mean"),
        mean_signed_change=("signed_change", "mean"),
        trial_score_sd=("politeness_score", "std"),
        class_agreement=("class_agrees", "mean"),
        mean_displacement=("mean_boundary_displacement", "mean"),
        mean_fraction_displaced=("fraction_of_available_displacement", "mean"),
        mean_randomized_score=("politeness_score", "mean"),
    ), on="argument_id", validate="one_to_one")
    opening = opening.join(fixed, on="argument_id", validate="one_to_one")
    opening["length_band"] = pd.cut(opening["n_tokens"], LENGTH_EDGES, labels=LENGTH_LABELS)
    if not opening.loc[opening["distinct_changed_layouts"].eq(0), "mean_absolute_change"].isna().all():
        raise ValueError("Unperturbable openings must be excluded from change statistics.")
    return opening, trials


def pair_results(pairs, baseline, trials, size, cohort):
    fixed = baseline["politeness_score"]
    base_pair = pairs.assign(
        fixed_a=pairs["a_comment_id"].map(fixed), fixed_b=pairs["b_comment_id"].map(fixed),
    )
    if base_pair[["fixed_a", "fixed_b"]].isna().any().any():
        raise ValueError("Every archived pair must have two fixed scores.")
    base_pair["fixed_difference"] = base_pair["fixed_a"] - base_pair["fixed_b"]
    repeated = []
    for trial in range(1, TRIALS + 1):
        sampled = trials.loc[trials["trial"].eq(trial)].set_index("opening_id")["politeness_score"]
        a = base_pair["a_comment_id"].map(sampled)
        b = base_pair["b_comment_id"].map(sampled)
        changed = a.fillna(base_pair["fixed_a"]) - b.fillna(base_pair["fixed_b"])
        repeated.append(base_pair.assign(
            trial=trial, content_limit=size, cohort=cohort, a_randomized=a.notna(), b_randomized=b.notna(),
            randomized_openings=a.notna().astype(int) + b.notna().astype(int),
            randomized_difference=changed,
        ))
    result = pd.concat(repeated, ignore_index=True)
    fixed_sign = np.sign(result["fixed_difference"].mask(result["fixed_difference"].abs().le(TIE_TOLERANCE), 0))
    random_sign = np.sign(result["randomized_difference"].mask(result["randomized_difference"].abs().le(TIE_TOLERANCE), 0))
    return result.assign(
        absolute_difference_change=(result["fixed_difference"] - result["randomized_difference"]).abs(),
        ranking_agrees=fixed_sign.eq(random_sign), rank_reversal=fixed_sign.mul(random_sign).eq(-1),
        into_tie=fixed_sign.ne(0) & random_sign.eq(0), out_of_tie=fixed_sign.eq(0) & random_sign.ne(0),
    )


def summarize_openings(group):
    valid = group.dropna(subset=["mean_absolute_change"])
    return {
        "n_openings": len(group), "n_perturbable": len(valid),
        "n_fewer_than_five_layouts": int(group["distinct_changed_layouts"].lt(TRIALS).sum()),
        "at_most_one_token": int(group["at_most_one_token_of_flexibility"].sum()),
        "mean_absolute_change": valid["mean_absolute_change"].mean(),
        "mean_signed_change": valid["mean_signed_change"].mean(),
        "mean_trial_score_sd": valid["trial_score_sd"].mean(),
        "class_agreement": valid["class_agreement"].mean(),
        "spearman_against_mean": spearman(valid["fixed_score"], valid["mean_randomized_score"]),
        "mean_displacement": valid["mean_displacement"].mean(),
        "mean_available_displacement": valid["max_boundary_displacement"].mean(),
        "mean_fraction_displaced": valid["mean_fraction_displaced"].mean(),
    }


def common_comparison(openings, trials):
    common = openings.pivot(index="argument_id", columns="content_limit", values="mean_absolute_change")
    common = common.loc[openings.loc[openings["content_limit"].eq(510), "argument_id"]]
    valid = common.dropna().index
    selected = openings.loc[openings["argument_id"].isin(valid)]
    raw = [{"content_limit": size, **summarize_openings(group)}
           for size, group in selected.groupby("content_limit")]
    # Pair the configurations within each opening and realized displacement band.
    comparable = trials.loc[trials["opening_id"].isin(valid)]
    adjusted = []
    for column in ("displacement_band", "fraction_displaced_band"):
        by_band = comparable.groupby(
            ["opening_id", column, "content_limit"], observed=True,
        )["absolute_change"].mean().unstack("content_limit").dropna()
        by_opening = by_band.groupby(level="opening_id").mean()
        adjusted.append({
            "matching": column, "n_common_openings": len(common), "n_perturbable_both": len(valid),
            "n_with_shared_displacement_band": len(by_opening), "n_matched_opening_bands": len(by_band),
            "mean_change_256": by_opening[256].mean(), "mean_change_510": by_opening[510].mean(),
            "mean_change_510_minus_256": (by_opening[510] - by_opening[256]).mean(),
        })
    return pd.DataFrame(raw), pd.DataFrame(adjusted)


def save_figures(openings, trials, pairs, destination):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    sizes = sorted(openings["content_limit"].unique())

    def save(fig, name):
        fig.savefig(destination / f"{name}.png", dpi=145)
        plt.close(fig)

    valid = openings.dropna(subset=["mean_absolute_change"]).groupby("argument_id")["content_limit"].nunique()
    common = openings.loc[openings["argument_id"].isin(valid.index[valid.eq(len(sizes))])]
    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    for size, group in common.groupby("content_limit"):
        data = group["mean_absolute_change"].dropna()
        ax.hist(data, bins=np.linspace(0, 3, 46), weights=np.full(len(data), 100 / len(data)),
                histtype="step", linewidth=2, color=COLORS[size], label=f"{size} tokens")
    ax.set(xlabel="Mean absolute score change per opening", ylabel="Openings per bin (%)",
           title="Boundary sensitivity on common long openings" if len(sizes) > 1 else "Boundary sensitivity")
    ax.legend(frameon=False)
    save(fig, "absolute_score_changes")

    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    for size, group in openings.groupby("content_limit"):
        means = group.groupby("length_band", observed=True)["mean_absolute_change"].mean().reindex(LENGTH_LABELS)
        ax.plot(LENGTH_LABELS, means, "o-", color=COLORS[size], label=f"{size} tokens")
    ax.set(xlabel="Opening length (content tokens)", ylabel="Mean absolute score change",
           title="Boundary sensitivity by opening length")
    ax.legend(frameon=False)
    save(fig, "sensitivity_by_length")

    fig, axes = plt.subplots(1, len(sizes), figsize=(4.5 * len(sizes), 4), squeeze=False,
                             layout="constrained", sharex=True, sharey=True)
    axes = axes[0]
    for (size, group), ax in zip(trials.groupby("content_limit"), axes, strict=True):
        ax.scatter(group["mean_boundary_displacement"], group["absolute_change"], s=4, alpha=.20,
                   color=COLORS[size], edgecolors="none")
        ax.set(xlabel="Mean boundary displacement (tokens)", title=f"{size} content tokens")
    axes[0].set_ylabel("Absolute score change")
    save(fig, "score_change_vs_displacement")

    fig, axes = plt.subplots(1, len(sizes), figsize=(4.5 * len(sizes), 4), squeeze=False,
                             layout="constrained", sharex=True, sharey=True)
    axes = axes[0]
    for (size, group), ax in zip(trials.groupby("content_limit"), axes, strict=True):
        ax.scatter(group["fixed_score"], group["politeness_score"], s=4, alpha=.17,
                   color=COLORS[size], edgecolors="none")
        ax.plot([0, 3], [0, 3], "--", linewidth=1, color="#555555")
        ax.set(xlabel="Fixed-boundary score", title=f"{size} content tokens", xlim=(0, 3), ylim=(0, 3))
        ax.set_aspect("equal")
    axes[0].set_ylabel("Randomized score")
    save(fig, "fixed_vs_randomized")

    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    offsets = (-.18, .18) if len(sizes) == 2 else (0,)
    for offset, size in zip(offsets, sizes, strict=True):
        rows = pairs.loc[pairs["content_limit"].eq(size) & pairs["cohort"].eq("full_sample")]
        rows = rows.set_index("randomized_openings").reindex([1, 2])
        ax.bar(np.arange(2) + offset, 100 * rows["rank_reversal_rate"], width=.33,
               color=COLORS[size], label=f"{size} tokens")
    ax.set(xticks=[0, 1], xticklabels=["One opening randomized", "Both openings randomized"],
           ylabel="Reversal rate among pair-trials (%)", title="Within-pair politeness ranking reversals")
    ax.legend(frameon=False)
    save(fig, "paired_ranking_reversals")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or sorted(RESULTS.glob("*/experiment_metadata.json"))[-1].parent
    metadata = json.loads((run / "experiment_metadata.json").read_text())
    phase2 = PHASE2 / metadata["phase2_run"]
    if (metadata["phase"], metadata["split"]) != (3, "train"):
        raise ValueError("Expected Phase 3 training results.")
    sizes = tuple(configuration["content_limit"] for configuration in metadata["configurations"])
    if not sizes or len(set(sizes)) != len(sizes) or not set(sizes).issubset(SIZES):
        raise ValueError("The run must have one result for each recorded Phase 3 chunk limit.")
    compare_sizes = len(sizes) == len(SIZES)
    fixed_path = phase2 / "chunk_size_scores.parquet"
    if sha256(fixed_path) != metadata["phase2_scores_sha256"]:
        raise ValueError("The fixed Phase 2 scores have changed.")
    baseline = pd.read_parquet(fixed_path)
    source = DATA / "paired-linguistic-strategies" / metadata["source_run"]
    pair_path = source / "pair_strategy_scores.parquet"
    pairs = pd.read_parquet(pair_path, columns=["pair_id", "post_id", "a_comment_id", "b_comment_id"])
    if not pairs["pair_id"].is_unique:
        raise ValueError("Expected one row per archived matched pair.")
    opening_frames, trial_frames, pair_frames = [], [], []
    for size in sizes:
        fixed = baseline.loc[baseline["content_limit"].eq(size)].set_index("argument_id")
        design = pd.read_parquet(run / f"boundary_flexibility_{size}.parquet")
        trials = pd.read_parquet(run / f"randomized_scores_{size}.parquet")
        opening, trials = argument_results(design, trials, fixed)
        opening_frames.append(opening)
        trial_frames.append(trials)
        pair_frames.append(pair_results(pairs, fixed, trials, size, "full_sample"))
        if compare_sizes:
            long_ids = design.loc[design["n_tokens"].gt(510), "argument_id"]
            long_trials = trials.loc[trials["opening_id"].isin(long_ids)]
            pair_frames.append(pair_results(pairs, fixed, long_trials, size, "common_long_openings"))

    openings = pd.concat(opening_frames, ignore_index=True)
    trials = pd.concat(trial_frames, ignore_index=True)
    pair_trials = pd.concat(pair_frames, ignore_index=True)
    overview = pd.DataFrame([
        {"content_limit": size, **summarize_openings(group)} for size, group in openings.groupby("content_limit")
    ])
    by_length = pd.DataFrame([
        {"content_limit": size, "length_band": band, **summarize_openings(group)}
        for (size, band), group in openings.groupby(["content_limit", "length_band"], observed=True)
    ])
    per_trial = pd.DataFrame([
        {"content_limit": size, "trial": trial, "n_openings": len(group),
         "spearman": spearman(group["fixed_score"], group["politeness_score"])}
        for (size, trial), group in trials.groupby(["content_limit", "trial"])
    ])
    pair_summary = pair_trials.groupby(["cohort", "content_limit", "randomized_openings"]).agg(
        n_pair_trials=("pair_id", "size"), n_unique_pairs=("pair_id", "nunique"),
        mean_absolute_difference_change=("absolute_difference_change", "mean"),
        ranking_agreement=("ranking_agrees", "mean"), rank_reversal_rate=("rank_reversal", "mean"),
        into_tie_rate=("into_tie", "mean"), out_of_tie_rate=("out_of_tie", "mean"),
    ).reset_index()
    tables = {
        "argument_summary": overview, "argument_by_length": by_length, "rank_by_trial": per_trial,
        "pair_summary": pair_summary,
    }
    if compare_sizes:
        tables["common_cohort"], tables["common_displacement"] = common_comparison(openings, trials)
    analysis = run / "analysis"
    analysis.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(analysis / f"{name}.csv", index=False)
    openings.to_parquet(analysis / "argument_robustness.parquet", index=False)
    pair_trials.to_parquet(analysis / "paired_trials.parquet", index=False)
    (analysis / "analysis_metadata.json").write_text(json.dumps({
        "pair_context_sha256": sha256(pair_path), "pairs": len(pairs), "tie_tolerance": TIE_TOLERANCE,
        "content_limits": sizes,
        "length_bands": LENGTH_LABELS, "displacement_bands": DISPLACEMENT_LABELS,
        "fraction_displaced_band_edges": [0, .25, .5, .75, 1],
    }, indent=2) + "\n")
    save_figures(openings, trials, pair_summary, analysis)
    for table in tables.values():
        print(table.round(4).to_string(index=False), end="\n\n")
    print(f"Saved Phase 3A analysis to {analysis}")


if __name__ == "__main__":
    main()
