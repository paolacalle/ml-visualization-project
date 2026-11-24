"""Compare the six saved baseline-integration models on the shared training folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_baseline_integration import MODEL, OUT
from train_p5_topic_interactions import METRICS, scores


BASELINES = ("P6", "P6a")
REPORTED_METRICS = ("ROC-AUC", "Pairwise accuracy", "Log loss")


def load_predictions(run):
    manifest = json.loads((run / "results.json").read_text())
    if (manifest["experiment"], manifest["split"]) != (MODEL, "train"):
        raise ValueError("Expected a baseline-integration training run.")
    names = list(manifest["specifications"])
    predictions = pd.read_parquet(run / "oof_predictions.parquet")
    if (
        len(predictions) != manifest["pairs"]
        or not predictions["pair_id"].is_unique
        or predictions["post_id"].nunique() != manifest["discussions"]
        or not predictions.groupby("post_id")["fold"].nunique().eq(1).all()
        or set(predictions["fold"]) != set(range(1, manifest["folds"] + 1))
        or set(predictions["a_wins"]) != {0, 1}
        or not predictions[names].apply(lambda probabilities: probabilities.between(0, 1).all()).all()
    ):
        raise ValueError("Expected one valid prediction per training pair and model, with grouped folds.")
    return predictions, names


def compare_models(predictions, names):
    outcome = predictions["a_wins"].to_numpy()
    values = [scores(outcome, predictions[name].to_numpy()) for name in names]
    performance = pd.DataFrame(values, index=pd.Index(names, name="model"), columns=METRICS)
    performance = performance.loc[:, REPORTED_METRICS].copy()
    performance["Correct pairs"] = (predictions[names].ge(.5).to_numpy() == outcome[:, None]).sum(axis=0)
    np.testing.assert_allclose(performance["Pairwise accuracy"], performance["Correct pairs"] / len(outcome))

    changes = []
    for reference in BASELINES:
        alternatives = performance.drop(index=reference)
        delta = alternatives.subtract(performance.loc[reference], axis="columns")
        delta["Correct pairs"] = delta["Correct pairs"].astype(int)
        changes.append(delta.reset_index().assign(reference=reference))
    incremental = pd.concat(changes, ignore_index=True)
    return performance.reset_index(), incremental[["model", "reference", *REPORTED_METRICS, "Correct pairs"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    run = args.run or sorted((OUT / MODEL).glob("*/results.json"))[-1].parent
    predictions, names = load_predictions(run)
    performance, incremental = compare_models(predictions, names)
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    performance.to_csv(output / "model_performance.csv", index=False)
    incremental.to_csv(output / "incremental_performance.csv", index=False)
    print(performance.round(5).to_string(index=False))
    print(incremental.round(5).to_string(index=False))
    print(f"Saved point estimates to {output}")


if __name__ == "__main__":
    main()
