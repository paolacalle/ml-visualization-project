"""Summarize Phase 2 agreement and plot sensitivity to chunk size."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_chunk_size_experiment import RESULTS, SIZES, sha256
from run_pilot import DATA

LENGTH_GROUPS = ("≤128", "129–256", "257–510", ">510")
COMPARISONS = tuple(combinations(SIZES, 2))
COLORS = ("#19727a", "#8c4b83", "#b55325")


def opening_context(source):
    file = source / "pair_strategy_scores.parquet"
    pairs = pd.read_parquet(file, columns=["a_comment_id", "b_comment_id", "a_wins", "broad_topic"])
    if not pairs["a_wins"].isin([0, 1]).all() or pairs["broad_topic"].isna().any():
        raise ValueError("Expected a binary outcome and a topic for every archived pair.")
    a = pairs.rename(columns={"a_comment_id": "argument_id"}).assign(won=pairs["a_wins"].eq(1))
    b = pairs.rename(columns={"b_comment_id": "argument_id"}).assign(won=pairs["a_wins"].eq(0))
    columns = ["argument_id", "broad_topic", "won"]
    context = pd.concat([a[columns], b[columns]], ignore_index=True).drop_duplicates().set_index("argument_id")
    if not context.index.is_unique:
        raise ValueError("An opening has conflicting archived topics or outcomes.")
    return context, sha256(file)


def load_comparisons(run, source):
    metadata = json.loads((run / "experiment_metadata.json").read_text())
    if (metadata["phase"], metadata["split"], metadata["source_run"]) != (2, "train", source.name):
        raise ValueError("Expected Phase 2 scores and their matching training context.")
    if metadata["input_sha256"] != sha256(source / "opening_strategy_scores.parquet"):
        raise ValueError("The archived openings differ from the scored input.")
    data = pd.read_parquet(run / "chunk_size_scores.parquet")
    context, context_hash = opening_context(source)
    if set(data["content_limit"]) != set(SIZES) or set(data["argument_id"]) != set(context.index):
        raise ValueError("The scores must cover all archived openings at the three chunk sizes.")
    wide = data.pivot(index="argument_id", columns="content_limit", values=["n_tokens", "politeness_score", "predicted_class"])
    if wide.isna().any().any() or len(wide) != metadata["arguments"]:
        raise ValueError("Each opening must have one complete prediction per size.")
    n_tokens = wide["n_tokens"].astype(int)
    if not n_tokens.eq(n_tokens[SIZES[0]], axis=0).all().all():
        raise ValueError("Token counts must match across chunk sizes.")
    lengths = n_tokens[SIZES[0]]
    context = context.reindex(wide.index).assign(
        n_tokens=lengths,
        length_group=pd.cut(lengths, [0, *SIZES, float("inf")], labels=LENGTH_GROUPS),
        outcome=context["won"].map({True: "Winner", False: "Comparison"}),
    )
    scores = wide["politeness_score"].astype(float)
    classes = wide["predicted_class"]
    differences = {}
    for small, large in COMPARISONS:
        difference = scores[small] - scores[large]
        agrees = classes[small].eq(classes[large])
        same_input = lengths.le(small)
        if not difference[same_input].eq(0).all() or not agrees[same_input].all():
            raise ValueError("Identical unchunked inputs should have identical cached predictions.")
        differences[small, large] = context.assign(
            score_small=scores[small], score_large=scores[large],
            score_difference=difference, absolute_difference=difference.abs(), class_agrees=agrees,
        )
    return differences, scores, metadata, context_hash


def metrics(group):
    first = group["score_small"].rank()
    second = group["score_large"].rank()
    correlation = first.corr(second) if first.nunique() > 1 and second.nunique() > 1 else np.nan
    return {
        "n_openings": len(group),
        "spearman": correlation,
        "mean_absolute_difference": group["absolute_difference"].mean(),
        "mean_signed_difference": group["score_difference"].mean(),
        "class_agreement": group["class_agrees"].mean(),
    }


def summarize(differences):
    overall, subgroups = [], []
    for (small, large), rows in differences.items():
        keys = {"smaller_tokens": small, "larger_tokens": large}
        cohorts = {
            "all": rows,
            "different_segmentation": rows.loc[rows["n_tokens"].gt(small)],
        }
        if large == max(SIZES):
            cohorts["unchunked_reference"] = rows.loc[rows["n_tokens"].le(large)]
            cohorts["unchunked_reference_different_segmentation"] = rows.loc[rows["n_tokens"].between(small + 1, large)]
        overall.extend({**keys, "cohort": label, **metrics(group)} for label, group in cohorts.items())

        dimensions = [
            ("length", ["length_group"]),
            ("topic", ["broad_topic"]),
            ("outcome", ["outcome"]),
            ("topic_within_length", ["broad_topic", "length_group"]),
            ("outcome_within_length", ["outcome", "length_group"]),
        ]
        for dimension, fields in dimensions:
            for values, group in rows.groupby(fields, observed=True, sort=True):
                labels = dict(zip(fields, values, strict=True))
                subgroups.append({**keys, "dimension": dimension, **labels, **metrics(group)})
    return pd.DataFrame(overall), pd.DataFrame(subgroups)


def save_figures(differences, scores, subgroups, destination):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    ranks = scores.corr(method="spearman").loc[list(SIZES), list(SIZES)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    image = ax.imshow(ranks, vmin=0, vmax=1, cmap="Blues")
    for i, row in enumerate(ranks.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > .65 else "black")
    ax.set(xticks=range(len(SIZES)), yticks=range(len(SIZES)), xticklabels=SIZES, yticklabels=SIZES,
           xlabel="Content tokens per chunk", ylabel="Content tokens per chunk", title="Rank agreement across chunk sizes")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.savefig(destination / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    by_length = subgroups.loc[subgroups["dimension"].eq("length")]
    for (small, large), color in zip(COMPARISONS, COLORS, strict=True):
        values = by_length.loc[by_length["smaller_tokens"].eq(small) & by_length["larger_tokens"].eq(large)]
        means = values.set_index("length_group")["mean_absolute_difference"].reindex(LENGTH_GROUPS)
        ax.plot(LENGTH_GROUPS, means, marker="o", label=f"{small} vs {large}", color=color)
    ax.set(xlabel="Opening length in content tokens", ylabel="Mean absolute score difference",
           title="Chunk-size sensitivity by opening length")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(destination / "sensitivity_by_length.png", dpi=150)
    plt.close(fig)

    reference = differences[128, 510].loc[lambda frame: frame["n_tokens"].between(129, 510)]
    fig, ax = plt.subplots(figsize=(5.8, 5.2), layout="constrained")
    density = ax.hexbin(reference["score_large"], reference["score_small"], extent=(0, 3, 0, 3),
                        gridsize=30, mincnt=1, bins="log", cmap="viridis")
    ax.plot([0, 3], [0, 3], "--", color="#60666c", linewidth=1)
    ax.set(xlabel="Unchunked score", ylabel="128-token score", xlim=(0, 3), ylim=(0, 3),
           title=f"Score agreement with unchunked inputs (n={len(reference):,})")
    ax.set_aspect("equal")
    fig.colorbar(density, ax=ax, label="Openings per bin")
    fig.savefig(destination / "score_agreement.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    bins = np.linspace(-3, 3, 61)
    for (small, large), color in zip(COMPARISONS, COLORS, strict=True):
        rows = differences[small, large]
        change = rows.loc[rows["n_tokens"].gt(small), "score_difference"]
        weights = np.full(len(change), 100 / len(change))
        ax.hist(change, bins=bins, histtype="step", weights=weights, linewidth=1.8,
                label=f"{small} vs {large} (n={len(change):,})", color=color)
    ax.axvline(0, color="#60666c", linestyle="--", linewidth=1)
    ax.set(xlabel="Smaller-chunk score minus larger-chunk score", ylabel="Openings per bin (%)",
           title="Score changes when chunk boundaries differ")
    ax.legend(frameon=False)
    fig.savefig(destination / "score_difference_distribution.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    run = args.run
    if run is None:
        manifests = sorted(RESULTS.glob("*/experiment_metadata.json"))
        if not manifests:
            parser.error("Run the chunk-size experiment first or provide --run.")
        run = manifests[-1].parent
    manifest = json.loads((run / "experiment_metadata.json").read_text())
    source = args.source or DATA / "paired-linguistic-strategies" / manifest["source_run"]
    differences, scores, _, context_hash = load_comparisons(run, source)
    overall, subgroups = summarize(differences)
    destination = run / "analysis"
    destination.mkdir(exist_ok=True)
    overall.to_csv(destination / "agreement.csv", index=False)
    subgroups.to_csv(destination / "subgroup_agreement.csv", index=False)
    (destination / "analysis_metadata.json").write_text(json.dumps({
        "pair_context_sha256": context_hash, "length_bins": LENGTH_GROUPS,
        "reference": "510-token configuration restricted to openings with at most 510 content tokens",
        "units": "unique openings from the archived matched training pairs",
    }, indent=2) + "\n")
    save_figures(differences, scores, subgroups, destination)
    print(overall.round(4).to_string(index=False))
    print(f"Saved Phase 2 analysis to {destination}")


if __name__ == "__main__":
    main()
