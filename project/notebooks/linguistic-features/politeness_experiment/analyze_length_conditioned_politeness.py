"""Compare the 256-token Polite Guard extensions by relative reply length."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_baseline_integration import MODEL, OUT, load_predictions
from paired_model_comparison import IDENTIFIERS, input_fingerprint, paired
from train_p5_topic_interactions import discussion_weights


LENGTH_EDGES = (0, .10, .25, .50, 1)
LENGTH_LABELS = ("0–10%", ">10–25%", ">25–50%", ">50–100%")
COMPARISONS = (("P6a", "P6c"), ("P6", "P6e"))


def load_length_bands(predictions):
    path = paired.DATA / "training_pair_assignments.parquet"
    assignments = pd.read_parquet(path)
    if not assignments["pair_id"].is_unique:
        raise ValueError("Expected a unique archived assignment for each pair.")
    matched = assignments.set_index("pair_id").loc[predictions["pair_id"]].reset_index()
    pd.testing.assert_frame_equal(predictions[IDENTIFIERS], matched[IDENTIFIERS])
    words = matched[["a_word_count", "b_word_count"]].to_numpy(dtype=float)
    if not np.isfinite(words).all() or (words <= 0).any():
        raise ValueError("Archived reply lengths must be finite and positive.")
    relative_gap = np.abs(words[:, 0] - words[:, 1]) / words.max(axis=1)
    groups = pd.cut(relative_gap, LENGTH_EDGES, labels=LENGTH_LABELS, include_lowest=True)
    if (groups.codes < 0).any():
        raise ValueError("Every relative length gap must fall in a specified band.")
    return groups, input_fingerprint(path)


def length_accuracy(predictions, groups):
    names = [model for comparison in COMPARISONS for model in comparison]
    outcomes = predictions["a_wins"].to_numpy()
    correct = (predictions[names].ge(.5).to_numpy() == outcomes[:, None]).T
    codes = groups.codes
    n_groups = len(LENGTH_LABELS)

    def group_accuracy(weights):
        mass = np.bincount(codes, weights=weights, minlength=n_groups)
        hits = np.stack([
            np.bincount(codes, weights=weights * model_correct, minlength=n_groups) for model_correct in correct
        ])
        return np.divide(hits, mass, out=np.full_like(hits, np.nan), where=mass > 0)

    points = group_accuracy(np.ones(len(predictions)))
    samples = np.stack([group_accuracy(weights) for weights in discussion_weights(predictions["post_id"])])
    counts = np.bincount(codes, minlength=n_groups)
    discussions = predictions.assign(length_band=groups).groupby("length_band", observed=True)["post_id"].nunique()
    if (counts == 0).any():
        raise ValueError("Each predeclared length band needs at least one pair.")

    summaries, contrasts = [], []
    for (baseline, extended), base_index in zip(COMPARISONS, (0, 2), strict=True):
        added_index = base_index + 1
        delta = points[added_index] - points[base_index]
        sampled_delta = samples[:, added_index] - samples[:, base_index]
        lower, upper = np.nanquantile(sampled_delta, [.025, .975], axis=0)
        for index, label in enumerate(LENGTH_LABELS):
            summaries.append({
                "baseline": baseline, "extended": extended, "length_band": label,
                "pairs": int(counts[index]), "discussions": int(discussions.loc[label]),
                "baseline_accuracy": points[base_index, index], "extended_accuracy": points[added_index, index],
                "change": delta[index], "ci_low": lower[index], "ci_high": upper[index],
            })
        contrast = sampled_delta[:, 0] - sampled_delta[:, -1]
        low, high = np.nanquantile(contrast, [.025, .975])
        contrasts.append({
            "baseline": baseline, "extended": extended,
            "first_band": LENGTH_LABELS[0], "second_band": LENGTH_LABELS[-1],
            "difference_in_gains": delta[0] - delta[-1], "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), len(samples)


def plot_length_accuracy(summary, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), sharex=True, sharey="col", layout="constrained")
    positions = np.arange(len(LENGTH_LABELS))
    for row, (baseline, extended) in enumerate(COMPARISONS):
        table = summary.loc[summary["baseline"].eq(baseline)]
        levels, changes = axes[row]
        levels.plot(positions, 100 * table["baseline_accuracy"], marker="o", label=baseline, color="#747a80")
        levels.plot(positions, 100 * table["extended_accuracy"], marker="o", label=extended, color="#19727a")
        levels.set(ylabel="Accuracy (%)")
        levels.legend(frameon=False)
        changes.axhline(0, color="#6e777d", linestyle="--", linewidth=1)
        changes.vlines(positions, 100 * table["ci_low"], 100 * table["ci_high"], color="#19727a", linewidth=1.7)
        changes.scatter(positions, 100 * table["change"], color="#125158", s=26, zorder=3)
        changes.set(ylabel="Accuracy change (pp)")
        for ax in (levels, changes):
            ax.grid(axis="y", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set(xticks=positions, xticklabels=LENGTH_LABELS, xlabel="Relative word-count difference")
    fig.suptitle("Predictive value of contextual politeness by reply length difference")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or sorted((OUT / MODEL).glob("*/results.json"))[-1].parent
    predictions, _ = load_predictions(run)
    groups, assignment_hash = load_length_bands(predictions)
    manifest = json.loads((run / "results.json").read_text())
    if assignment_hash != manifest["pair_assignments_sha256"]:
        raise ValueError("The archived length assignments changed since baseline integration.")
    summary, contrasts, draws = length_accuracy(predictions, groups)
    output = run / "analysis" / "length_conditioned"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "length_accuracy.csv", index=False)
    contrasts.to_csv(output / "length_contrasts.csv", index=False)
    (output / "length_analysis_metadata.json").write_text(json.dumps({
        "split": "train", "predictions_sha256": input_fingerprint(run / "oof_predictions.parquet"),
        "pair_assignments_sha256": assignment_hash, "seed": paired.SEED, "draws": draws,
        "length_edges": LENGTH_EDGES, "length_labels": LENGTH_LABELS, "resampling_unit": "OP discussion",
        "interval": "paired 95% percentile intervals with fixed out-of-fold predictions",
    }, indent=2) + "\n")
    plot_length_accuracy(summary, output / "length_conditioned_performance.png")
    print(summary.round(5).to_string(index=False))
    print(contrasts.round(5).to_string(index=False))
    print(f"Saved length-conditioned analysis to {output}")


if __name__ == "__main__":
    main()
