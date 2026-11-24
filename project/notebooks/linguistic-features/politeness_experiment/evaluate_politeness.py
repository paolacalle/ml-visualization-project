"""Evaluate matched-pair politeness predictions and prepare a blind review."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from politeness_extractor import POLITENESS_WEIGHTS
from run_chunk_size_experiment import RESULTS as PHASE2, SIZES, sha256
from run_pilot import DATA, SEED

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from paired_model_comparison import IDENTIFIERS, paired
from train_p5_topic_interactions import DRAWS, METRICS, discussion_weights, scores

RESULTS = DATA / "politeness-experiment/phase-4"
LENGTH = "Word count"
COURTESY = "Courtesy count"
LENGTH_COURTESY = "Word count + courtesy"


def load_inputs(phase2):
    manifest = json.loads((phase2 / "experiment_metadata.json").read_text())
    if (manifest["phase"], manifest["split"]) != (2, "train"):
        raise ValueError("Expected the Phase 2 training-only scores.")
    source = DATA / "paired-linguistic-strategies" / manifest["source_run"]
    opening_path = source / "opening_strategy_scores.parquet"
    if sha256(opening_path) != manifest["input_sha256"]:
        raise ValueError("The archived opening text differs from the Phase 2 input.")
    openings = pd.read_parquet(opening_path, columns=["opening_text", "analyzed_word_count", "courtesy_count"])
    pair_path = source / "pair_strategy_scores.parquet"
    pairs = pd.read_parquet(pair_path, columns=[*IDENTIFIERS, "a_word_count", "b_word_count", "broad_topic"])
    assignments_path = DATA / "training_pair_assignments.parquet"
    assignments = pd.read_parquet(assignments_path)
    pd.testing.assert_frame_equal(pairs[assignments.columns], assignments)
    if not pairs.groupby("post_id")["fold"].nunique().eq(1).all():
        raise ValueError("All pairs from a discussion must stay in the same fold.")
    score_path = phase2 / "chunk_size_scores.parquet"
    scores = pd.read_parquet(score_path)
    if set(scores["content_limit"]) != set(SIZES) or set(scores["argument_id"]) != set(openings.index):
        raise ValueError("Expected each Phase 2 configuration to score all archived training openings.")
    wide = scores.pivot(index="argument_id", columns="content_limit", values=["politeness_score", "predicted_class"])
    if wide.isna().any().any():
        raise ValueError("Expected complete contextual scores and class predictions.")
    provenance = {name: sha256(path) for name, path in {
        "opening_scores_sha256": opening_path, "pair_scores_sha256": pair_path,
        "phase2_scores_sha256": score_path, "pair_assignments_sha256": assignments_path,
    }.items()}
    return pairs, openings, wide, provenance


def features_for_pairs(pairs, openings, wide):
    values = pd.DataFrame({LENGTH: pairs["a_word_count"] - pairs["b_word_count"]})
    per_opening = openings[["courtesy_count"]].rename(columns={"courtesy_count": COURTESY})
    for size in SIZES:
        per_opening[f"Polite Guard {size}"] = wide["politeness_score"][size].astype(float)
    for field in per_opening:
        values[field] = pairs["a_comment_id"].map(per_opening[field]) - pairs["b_comment_id"].map(per_opening[field])
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("The evaluation requires complete paired feature differences.")
    return values


def extra_precision(y, probabilities, weights):
    predicted = probabilities >= .5
    positive_mass = weights[predicted].sum()
    return weights[predicted & (y == 1)].sum() / positive_mass if positive_mass else np.nan


def evaluate(pairs, differences):
    specifications = {
        LENGTH: ([LENGTH], None),
        COURTESY: ([COURTESY], LENGTH),
        LENGTH_COURTESY: ([LENGTH, COURTESY], LENGTH),
    }
    for size in SIZES:
        tone = f"Polite Guard {size}"
        specifications[tone] = ([tone], LENGTH)
        specifications[f"Word count + {tone}"] = ([LENGTH, tone], LENGTH)
        specifications[f"Word count + courtesy + {tone}"] = ([LENGTH, COURTESY, tone], LENGTH_COURTESY)
    predictions = {
        label: paired.evaluate_differences(pairs, differences[columns])[0]
        for label, (columns, _) in specifications.items()
    }
    y = pairs["a_wins"].to_numpy()
    metrics = (*METRICS, "Precision at 0.5")

    def calculate(weights):
        return np.array([
            [*scores(y, probabilities, weights), extra_precision(y, probabilities, weights)]
            for probabilities in predictions.values()
        ])

    point = calculate(np.ones(len(pairs)))
    draws = np.stack([calculate(weights) for weights in discussion_weights(pairs["post_id"])])
    rows = []
    for position, (label, (_, reference)) in enumerate(specifications.items()):
        if reference is None:
            change, lower, upper = (np.full(len(metrics), np.nan) for _ in range(3))
        else:
            baseline = list(specifications).index(reference)
            change = point[position] - point[baseline]
            lower, upper = np.nanquantile(draws[:, position] - draws[:, baseline], [.025, .975], axis=0)
        rows.extend({
            "model": label, "metric": metric, "score": point[position, column],
            "reference": reference, "difference": change[column],
            "difference_ci_low": lower[column], "difference_ci_high": upper[column],
        } for column, metric in enumerate(metrics))
    return pd.DataFrame(rows), pairs[IDENTIFIERS].assign(**predictions)


def review_sample(pairs, openings, wide):
    context = pd.concat([
        pairs[[f"{side}_comment_id", "post_id", "broad_topic"]].rename(columns={f"{side}_comment_id": "comment_id"})
        for side in ("a", "b")
    ]).drop_duplicates().set_index("comment_id")
    if not context.index.is_unique or set(context.index) != set(openings.index):
        raise ValueError("Expected each opening to map to one discussion and topic.")
    rng = np.random.default_rng(SEED)
    pool = openings.join(context).sort_index()
    pool["score_class"] = wide["predicted_class"][256]
    pool["length_band"] = pd.qcut(pool["analyzed_word_count"], 3, labels=["Short", "Medium", "Long"])
    pool["random_order"] = rng.permutation(len(pool))
    buckets = list(pool.groupby(["score_class", "length_band"], observed=True))
    if len(buckets) != len(POLITENESS_WEIGHTS) * 3:
        raise ValueError("Expected representation for all tone and length strata.")

    topic_counts = dict.fromkeys(pool["broad_topic"].unique(), 0)
    chosen_ids, discussions = [], set()
    for (label, length), group in sorted(buckets, key=lambda entry: (len(entry[1]), entry[0])):
        quota = 3 if label == "impolite" and length == "Medium" else 2
        for _ in range(quota):
            candidates = group.loc[~group["post_id"].isin(discussions)].copy()
            if candidates.empty:
                raise ValueError(f"Too few unique discussions to sample {label}, {length}.")
            candidates["topic_count"] = candidates["broad_topic"].map(topic_counts)
            picked = candidates.sort_values(["topic_count", "random_order"]).iloc[0]
            chosen_ids.append(picked.name)
            discussions.add(picked["post_id"])
            topic_counts[picked["broad_topic"]] += 1
    chosen = pool.loc[rng.permutation(chosen_ids)].copy()
    chosen.insert(0, "review_id", [f"R{index:02d}" for index in range(1, len(chosen) + 1)])
    blind = chosen[["review_id", "opening_text"]].reset_index(drop=True).assign(manual_label="", notes="")
    # A text prefix prevents a spreadsheet from evaluating any opening as a formula.
    risky = blind["opening_text"].str.lstrip().str.startswith(("=", "+", "-", "@"))
    blind.loc[risky, "opening_text"] = "'" + blind.loc[risky, "opening_text"]
    key = chosen[["review_id", "post_id", "broad_topic", "length_band", "courtesy_count"]].copy()
    for size in SIZES:
        key[f"score_{size}"] = wide["politeness_score"][size].astype(float)
        key[f"class_{size}"] = wide["predicted_class"][size]
    return blind, key.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run", type=Path)
    args = parser.parse_args()
    phase2 = args.phase2_run or sorted(PHASE2.glob("*/experiment_metadata.json"))[-1].parent
    pairs, openings, wide, provenance = load_inputs(phase2)
    differences = features_for_pairs(pairs, openings, wide)
    with threadpool_limits(limits=1):
        comparison, predictions = evaluate(pairs, differences)
    for label, probabilities in predictions.drop(columns=IDENTIFIERS).items():
        expected = comparison.loc[comparison["model"].eq(label) & comparison["metric"].eq("Log loss"), "score"].item()
        np.testing.assert_allclose(expected, log_loss(pairs["a_wins"], probabilities), atol=1e-12)
    blind, key = review_sample(pairs, openings, wide)
    assert len(blind) == 25 and blind["review_id"].is_unique and key["post_id"].is_unique

    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=RESULTS) as temporary:
        stage = Path(temporary)
        comparison.to_csv(stage / "predictive_comparison.csv", index=False)
        predictions.to_parquet(stage / "oof_predictions.parquet", index=False)
        blind.to_csv(stage / "blind_politeness_review.csv", index=False)
        key.to_parquet(stage / "review_key.parquet", index=False)
        (stage / "experiment_metadata.json").write_text(json.dumps({
            "split": "train", "phase2_run": phase2.name, "seed": SEED, "folds": paired.FOLDS,
            "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "a_win_prevalence": float(pairs["a_wins"].mean()), "bootstrap_draws": DRAWS,
            "review_openings": len(blind), "review_topics": key["broad_topic"].nunique(),
            "polite_guard_model": "Intel/polite-guard", **provenance,
        }, indent=2) + "\n")
        stage.rename(destination)
    print(comparison.pivot(index="model", columns="metric", values="score").round(4).to_string())
    print(f"Saved predictive results and blind review to {destination}")


if __name__ == "__main__":
    main()
