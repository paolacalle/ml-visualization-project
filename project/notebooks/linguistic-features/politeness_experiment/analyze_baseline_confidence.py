"""Measure where contextual politeness changes the baseline model's picks."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from analyze_length_conditioned_politeness import COMPARISONS
from compare_baseline_integration import MODEL, OUT, load_predictions
from paired_model_comparison import input_fingerprint, paired
from train_p5_topic_interactions import discussion_weights


CONFIDENCE_EDGES = (.50, .60, .80, 1)
CONFIDENCE_LABELS = ("50–60%", ">60–80%", ">80–100%")
TRANSITIONS = ("both_correct", "new_error", "corrected", "both_wrong")


def summarize_confidence(predictions):
    outcome = predictions["a_wins"].to_numpy()
    summaries, contrasts = [], []
    weights = list(discussion_weights(predictions["post_id"]))
    for baseline, extended in COMPARISONS:
        p = predictions[baseline].to_numpy()
        base_correct = (p >= .5) == outcome
        extended_correct = (predictions[extended].to_numpy() >= .5) == outcome
        confidence = np.maximum(p, 1 - p)
        bands = pd.cut(confidence, CONFIDENCE_EDGES, labels=CONFIDENCE_LABELS, include_lowest=True)
        counts = np.bincount(bands.codes, minlength=len(CONFIDENCE_LABELS))
        if (counts == 0).any():
            raise ValueError("Each confidence band needs at least one pair.")
        patterns = (
            base_correct & extended_correct, base_correct & ~extended_correct,
            ~base_correct & extended_correct, ~base_correct & ~extended_correct,
        )
        change = patterns[2].astype(int) - patterns[1].astype(int)

        def sampled_change(sample_weights):
            masses = np.bincount(bands.codes, weights=sample_weights, minlength=len(CONFIDENCE_LABELS))
            gains = np.bincount(bands.codes, weights=sample_weights * change, minlength=len(CONFIDENCE_LABELS))
            return np.divide(gains, masses, out=np.full_like(gains, np.nan), where=masses > 0)

        draws = np.stack([sampled_change(sample_weights) for sample_weights in weights])
        lower, upper = np.nanquantile(draws, [.025, .975], axis=0)
        point = []
        for index, label in enumerate(CONFIDENCE_LABELS):
            mask = bands == label
            transitions = {name: int(pattern[mask].sum()) for name, pattern in zip(TRANSITIONS, patterns, strict=True)}
            n_pairs = int(counts[index])
            base_hits = transitions["both_correct"] + transitions["new_error"]
            base_misses = n_pairs - base_hits
            gain = float(change[mask].mean())
            point.append(gain)
            summaries.append({
                "baseline": baseline, "extended": extended, "confidence_band": label,
                "pairs": n_pairs, "discussions": predictions.loc[mask, "post_id"].nunique(), **transitions,
                "baseline_accuracy": base_hits / n_pairs,
                "extended_accuracy": (transitions["both_correct"] + transitions["corrected"]) / n_pairs,
                "corrected_share_of_baseline_errors": transitions["corrected"] / base_misses if base_misses else np.nan,
                "new_error_share_of_baseline_correct": transitions["new_error"] / base_hits if base_hits else np.nan,
                "accuracy_change": gain, "ci_low": lower[index], "ci_high": upper[index],
            })
        low, high = np.nanquantile(draws[:, 0] - draws[:, -1], [.025, .975])
        contrasts.append({
            "baseline": baseline, "extended": extended, "low_minus_high_accuracy_change": point[0] - point[-1],
            "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), len(weights)


def plot_disagreements(summary, path):
    palette = ListedColormap(["#dbe9df", "#f4dfd4", "#d1ecea", "#e7e9eb"])
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8), layout="constrained")
    for row, (baseline, extended) in enumerate(COMPARISONS):
        for column, label in enumerate(CONFIDENCE_LABELS):
            ax = axes[row, column]
            cell = summary.loc[summary["baseline"].eq(baseline) & summary["confidence_band"].eq(label)].iloc[0]
            ax.imshow(np.arange(4).reshape(2, 2), cmap=palette, vmin=0, vmax=3)
            labels = ("Both correct", "New error", "Corrected", "Both wrong")
            for index, (name, title) in enumerate(zip(TRANSITIONS, labels, strict=True)):
                y, x = divmod(index, 2)
                count = int(cell[name])
                ax.text(x, y, f"{title}\n{count:,} · {count / cell['pairs']:.1%}",
                        ha="center", va="center", color="#223238", fontsize=10)
            ax.set(xticks=[0, 1], xticklabels=["Correct", "Wrong"], yticks=[0, 1], yticklabels=["Correct", "Wrong"])
            ax.set_xlabel(f"{extended} pick")
            ax.set_ylabel(f"{baseline} pick")
            ax.set_title(f"{label} confidence · {int(cell['pairs']):,} pairs", fontsize=11)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("Matched-pair prediction agreement by baseline confidence")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or sorted((OUT / MODEL).glob("*/results.json"))[-1].parent
    predictions, _ = load_predictions(run)
    summary, contrasts, draws = summarize_confidence(predictions)
    output = run / "analysis" / "baseline_confidence"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "confidence_accuracy.csv", index=False)
    contrasts.to_csv(output / "confidence_contrasts.csv", index=False)
    (output / "confidence_metadata.json").write_text(json.dumps({
        "split": "train", "confidence_edges": CONFIDENCE_EDGES, "confidence_labels": CONFIDENCE_LABELS,
        "seed": paired.SEED, "draws": draws, "resampling_unit": "OP discussion",
        "interval": "paired 95% percentile intervals with fixed out-of-fold predictions",
        "predictions_sha256": input_fingerprint(run / "oof_predictions.parquet"),
    }, indent=2) + "\n")
    plot_disagreements(summary, output / "model_disagreement_matrix.png")
    print(summary.round(5).to_string(index=False))
    print(contrasts.round(5).to_string(index=False))
    print(f"Saved confidence analysis to {output}")


if __name__ == "__main__":
    main()
