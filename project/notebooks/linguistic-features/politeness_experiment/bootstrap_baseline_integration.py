"""Estimate paired, discussion-bootstrap intervals for baseline integration."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from compare_baseline_integration import (
    BASELINES, METRICS, MODEL, OUT, REPORTED_METRICS, compare_models, load_predictions, scores,
)
from paired_model_comparison import input_fingerprint, paired
from train_p5_topic_interactions import discussion_weights


def bootstrap_intervals(predictions, names):
    """Hold all predictions fixed and use each resample for every model."""
    outcome = predictions["a_wins"].to_numpy()
    probabilities = predictions[names].to_numpy().T
    metric_indices = [METRICS.index(metric) for metric in REPORTED_METRICS]
    _, changes = compare_models(predictions, names)
    draws = np.array([
        [scores(outcome, probability, weights)[metric_indices] for probability in probabilities]
        for weights in discussion_weights(predictions["post_id"])
        if weights[outcome == 0].any() and weights[outcome == 1].any()
    ])
    if not len(draws):
        raise ValueError("The bootstrap needs draws containing both outcomes.")
    positions = {name: index for index, name in enumerate(names)}
    rows = []
    for (name, reference), point in changes.set_index(["model", "reference"]).iterrows():
        differences = draws[:, positions[name]] - draws[:, positions[reference]]
        lower, upper = np.quantile(differences, [.025, .975], axis=0)
        rows.extend({
            "model": name, "reference": reference, "metric": metric,
            "change": point[metric], "ci_low": lo, "ci_high": hi,
        } for metric, lo, hi in zip(REPORTED_METRICS, lower, upper, strict=True))
    return pd.DataFrame(rows), len(draws)


def plot_intervals(intervals, path):
    labels = ("Δ ROC-AUC", "Δ accuracy (pp)", "Δ log loss")
    fig, axes = plt.subplots(len(BASELINES), len(REPORTED_METRICS), figsize=(10.5, 5.7),
                             sharex="col", sharey="row", layout="constrained")
    for row, reference in enumerate(BASELINES):
        for column, (metric, label) in enumerate(zip(REPORTED_METRICS, labels, strict=True)):
            ax = axes[row, column]
            panel = intervals.loc[intervals["reference"].eq(reference) & intervals["metric"].eq(metric)]
            factor = 100 if metric == "Pairwise accuracy" else 1
            y = np.arange(len(panel))
            ax.axvline(0, color="#6e777d", linestyle="--", linewidth=1)
            ax.hlines(y, factor * panel["ci_low"], factor * panel["ci_high"], color="#19727a", linewidth=1.7)
            ax.scatter(factor * panel["change"], y, color="#125158", s=24, zorder=3)
            ax.set_yticks(y, panel["model"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x", alpha=.15)
            if column == 0:
                ax.set_ylabel(f"Against {reference}")
                ax.invert_yaxis()
            if row == 0:
                ax.set_title(label)
    fig.suptitle("Changes in out-of-fold performance with 95% paired bootstrap intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or sorted((OUT / MODEL).glob("*/results.json"))[-1].parent
    predictions, names = load_predictions(run)
    with threadpool_limits(limits=1):
        intervals, n_draws = bootstrap_intervals(predictions, names)
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    intervals.to_csv(output / "bootstrap_intervals.csv", index=False)
    (output / "bootstrap_metadata.json").write_text(json.dumps({
        "resampling_unit": "OP discussion", "seed": paired.SEED, "draws": n_draws,
        "confidence_level": .95, "method": "paired percentile intervals with fixed out-of-fold predictions",
        "discussions": predictions["post_id"].nunique(),
        "predictions_sha256": input_fingerprint(run / "oof_predictions.parquet"),
    }, indent=2) + "\n")
    plot_intervals(intervals, output / "incremental_confidence_intervals.png")
    print(intervals.round(5).to_string(index=False))
    print(f"Saved intervals from {n_draws:,} paired discussion bootstrap draws to {output}")


if __name__ == "__main__":
    main()
