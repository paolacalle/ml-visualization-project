"""Refit Phase 4 specifications to summarize standardized feature coefficients."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from train_baseline_integration import CONTEXTUAL_FEATURES, MODEL, OUT, PHASE2, pair_features, paired
from train_p6_adjusted_linguistic_strategies import DRAWS, adjusted_coefficients


def coefficient_table(pairs, features, specifications):
    frame = pairs.join(features)
    tables = []
    for name, columns in specifications.items():
        table = adjusted_coefficients(frame, features=columns)
        tables.append(table.assign(model=name))
    combined = pd.concat(tables, ignore_index=True)
    scales = combined.pivot(index="feature", columns="model", values="training_feature_sd")
    np.testing.assert_allclose(scales.min(axis=1), scales.max(axis=1), atol=1e-12)
    return combined.set_index(["model", "feature"]).reset_index()


def plot_contextual_coefficients(rows, path):
    y = np.arange(len(rows))
    labels = rows["model"] + " · " + rows["content_limit"].astype(str) + " tokens"
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    ax.axvline(0, color="#6e777d", linestyle="--", linewidth=1)
    ax.hlines(y, rows["ci_low"], rows["ci_high"], color="#19727a", linewidth=2)
    ax.scatter(rows["coefficient"], y, color="#125158", s=30, zorder=3)
    ax.set(
        yticks=y, yticklabels=labels, title="Contextual politeness coefficients with 95% intervals",
        xlabel="Change in log odds per SD of the A-minus-B politeness score",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.invert_yaxis()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or sorted((OUT / MODEL).glob("*/results.json"))[-1].parent
    manifest = json.loads((run / "results.json").read_text())
    if (manifest["experiment"], manifest["split"]) != (MODEL, "train"):
        raise ValueError("Expected a baseline-integration training run.")
    pairs, features, _, provenance = pair_features(PHASE2 / manifest["phase2_run"])
    if any(manifest.get(name) != digest for name, digest in provenance.items()):
        raise ValueError("The coefficient inputs differ from the baseline-integration training run.")
    with threadpool_limits(limits=1):
        coefficients = coefficient_table(pairs, features, manifest["specifications"])

    size_by_feature = {feature: size for size, feature in CONTEXTUAL_FEATURES.items()}
    contextual = coefficients.loc[coefficients["feature"].isin(size_by_feature)].copy()
    contextual["content_limit"] = contextual["feature"].map(size_by_feature)
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    coefficients.to_csv(output / "standardized_coefficients.csv", index=False)
    contextual.to_csv(output / "contextual_politeness_coefficients.csv", index=False)
    (output / "coefficient_metadata.json").write_text(json.dumps({
        "split": "train", "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
        "seed": paired.SEED, "draws": DRAWS, "confidence_level": .95, "resampling_unit": "OP discussion",
        "method": "penalized logistic refits on bootstrap discussions with the full-training scale held fixed",
        **provenance,
    }, indent=2) + "\n")
    plot_contextual_coefficients(contextual, output / "contextual_politeness_coefficients.png")
    fields = ["model", "content_limit", "coefficient", "ci_low", "ci_high", "informative_pairs"]
    print(contextual[fields].round(5).to_string(index=False))
    print(f"Saved standardized coefficients and plot to {output}")


if __name__ == "__main__":
    main()
