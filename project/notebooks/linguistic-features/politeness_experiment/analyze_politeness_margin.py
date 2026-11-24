"""Compare winner rates and predictive gains by 256-token politeness margin."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from analyze_length_conditioned_politeness import COMPARISONS
from compare_baseline_integration import MODEL, OUT, REPORTED_METRICS, load_predictions
from paired_model_comparison import IDENTIFIERS, input_fingerprint, paired
from train_baseline_integration import CONTEXTUAL_FEATURES, PHASE2, pair_features
from train_p5_topic_interactions import discussion_weights


MARGIN_EDGES = (0, .25, .50, 1, 3)
MARGIN_LABELS = ("≤0.25", ">0.25–0.50", ">0.50–1.00", ">1.00–3.00")
CONTENT_LIMIT = 256


def margin_summary(predictions, difference):
    outcome = predictions["a_wins"].to_numpy()
    margin = np.abs(difference)
    tied = margin <= paired.TIE_TOLERANCE
    bands = pd.cut(margin, MARGIN_EDGES, labels=MARGIN_LABELS)
    if (bands.codes[~tied] < 0).any():
        raise ValueError("Every non-tied politeness margin must be on the 0–3 score scale.")
    higher_won = (difference > 0) == outcome
    names = [model for pair in COMPARISONS for model in pair]
    probabilities = predictions[names].to_numpy()
    correct = (probabilities >= .5) == outcome[:, None]
    safe = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    losses = -np.where(outcome[:, None] == 1, np.log(safe), np.log1p(-safe))

    masks = [(bands == label) & ~tied for label in MARGIN_LABELS]
    if any(not mask.any() for mask in masks):
        raise ValueError("Each non-tied politeness band needs at least one pair.")

    def statistics(weights):
        association, comparisons = [], []
        for mask in masks:
            weight = weights[mask]
            if not weight.any():
                association.append(np.nan)
                comparisons.append(np.full((len(names), len(REPORTED_METRICS)), np.nan))
                continue
            target = outcome[mask]
            group_probabilities = probabilities[mask]
            enough_labels = weight[target == 1].any() and weight[target == 0].any()
            aucs = (
                [roc_auc_score(target, p, sample_weight=weight) for p in group_probabilities.T]
                if enough_labels else np.full(len(names), np.nan)
            )
            association.append(np.average(higher_won[mask], weights=weight))
            comparisons.append(np.column_stack([
                aucs, np.average(correct[mask], axis=0, weights=weight), np.average(losses[mask], axis=0, weights=weight),
            ]))
        return np.asarray(association), np.asarray(comparisons)

    point_rate, point_metrics = statistics(np.ones(len(predictions)))
    bootstrap = [statistics(weights) for weights in discussion_weights(predictions["post_id"])]
    rates, metrics = (np.stack(values) for values in zip(*bootstrap, strict=True))
    win_lower, win_upper = np.nanquantile(rates, [.025, .975], axis=0)
    association = pd.DataFrame([
        {"margin": label, "pairs": int(mask.sum()), "discussions": predictions.loc[mask, "post_id"].nunique(),
         "higher_score_wins": int(higher_won[mask].sum()), "win_rate": point_rate[index],
         "ci_low": win_lower[index], "ci_high": win_upper[index]}
        for index, (label, mask) in enumerate(zip(MARGIN_LABELS, masks, strict=True))
    ])
    predictive_rows = []
    differences = []
    for index, (baseline, extended) in enumerate(COMPARISONS):
        base_index, extended_index = 2 * index, 2 * index + 1
        points = point_metrics[:, extended_index] - point_metrics[:, base_index]
        draws = metrics[:, :, extended_index] - metrics[:, :, base_index]
        lower, upper = np.nanquantile(draws, [.025, .975], axis=0)
        for band, label in enumerate(MARGIN_LABELS):
            for metric, name in enumerate(REPORTED_METRICS):
                predictive_rows.append({
                    "baseline": baseline, "extended": extended, "margin": label, "metric": name,
                    "baseline_value": point_metrics[band, base_index, metric],
                    "extended_value": point_metrics[band, extended_index, metric],
                    "change": points[band, metric], "ci_low": lower[band, metric], "ci_high": upper[band, metric],
                })
        accuracy_index = REPORTED_METRICS.index("Pairwise accuracy")
        low, high = np.nanquantile(draws[:, -1, accuracy_index] - draws[:, 0, accuracy_index], [.025, .975])
        differences.append({
            "contrast": f"{extended} vs {baseline} accuracy gain",
            "large_minus_nearly_equal": points[-1, accuracy_index] - points[0, accuracy_index],
            "ci_low": low, "ci_high": high,
        })
    low, high = np.nanquantile(rates[:, -1] - rates[:, 0], [.025, .975])
    differences.insert(0, {
        "contrast": "Higher-score win rate", "large_minus_nearly_equal": point_rate[-1] - point_rate[0],
        "ci_low": low, "ci_high": high,
    })
    tie_summary = pd.DataFrame({
        "model": names, "pairs": int(tied.sum()), "correct_pairs": correct[tied].sum(axis=0),
        "discussions": predictions.loc[tied, "post_id"].nunique(),
    })
    return association, pd.DataFrame(predictive_rows), pd.DataFrame(differences), tie_summary, len(bootstrap)


def plot_margins(association, performance, path):
    fig, (wins, accuracy) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    positions = np.arange(len(MARGIN_LABELS))
    wins.axhline(50, color="#6e777d", linestyle="--", linewidth=1)
    wins.vlines(positions, 100 * association["ci_low"], 100 * association["ci_high"], color="#19727a", linewidth=1.7)
    wins.scatter(positions, 100 * association["win_rate"], color="#125158", s=27, zorder=3)
    wins.set(ylabel="Higher-scoring reply won (%)", title="Association with winning")
    accuracy.axhline(0, color="#6e777d", linestyle="--", linewidth=1)
    accuracy_rows = performance.loc[performance["metric"].eq("Pairwise accuracy")]
    for offset, (baseline, extended), color in zip((-.09, .09), COMPARISONS, ("#19727a", "#8c4b83"), strict=True):
        rows = accuracy_rows.loc[accuracy_rows["baseline"].eq(baseline)]
        accuracy.vlines(positions + offset, 100 * rows["ci_low"], 100 * rows["ci_high"], color=color, linewidth=1.7)
        accuracy.scatter(positions + offset, 100 * rows["change"], color=color, s=24,
                         label=f"{extended} vs {baseline}", zorder=3)
    accuracy.set(ylabel="Accuracy change (pp)", title="Incremental predictive accuracy")
    accuracy.legend(frameon=False)
    for ax in (wins, accuracy):
        ax.set(xticks=positions, xticklabels=MARGIN_LABELS, xlabel="Absolute Polite Guard score difference")
        ax.grid(axis="y", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or sorted((OUT / MODEL).glob("*/results.json"))[-1].parent
    predictions, _ = load_predictions(run)
    manifest = json.loads((run / "results.json").read_text())
    pairs, features, _, provenance = pair_features(PHASE2 / manifest["phase2_run"])
    if any(manifest.get(key) != value for key, value in provenance.items()):
        raise ValueError("The margin inputs differ from the baseline-integration training run.")
    pd.testing.assert_frame_equal(predictions[IDENTIFIERS], pairs[IDENTIFIERS])
    difference = features[CONTEXTUAL_FEATURES[CONTENT_LIMIT]].to_numpy()
    with threadpool_limits(limits=1):
        association, performance, contrast, ties, draws = margin_summary(predictions, difference)
    output = run / "analysis" / "politeness_margin"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "higher_score_win_rate": association, "incremental_performance": performance,
        "margin_contrasts": contrast, "ties": ties,
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "margin_metadata.json").write_text(json.dumps({
        "split": "train", "content_limit": CONTENT_LIMIT, "draws": draws, "seed": paired.SEED,
        "margin_edges": MARGIN_EDGES, "margin_labels": MARGIN_LABELS, "tie_tolerance": paired.TIE_TOLERANCE,
        "resampling_unit": "OP discussion", "interval": "paired 95% percentile intervals with fixed predictions",
        "predictions_sha256": input_fingerprint(run / "oof_predictions.parquet"), **provenance,
    }, indent=2) + "\n")
    plot_margins(association, performance, output / "politeness_margin.png")
    print(association.round(5).to_string(index=False))
    print(performance.loc[performance["metric"].eq("Pairwise accuracy")].round(5).to_string(index=False))
    print(contrast.round(5).to_string(index=False))
    print(ties.to_string(index=False))
    print(f"Saved margin analysis to {output}")


if __name__ == "__main__":
    main()
