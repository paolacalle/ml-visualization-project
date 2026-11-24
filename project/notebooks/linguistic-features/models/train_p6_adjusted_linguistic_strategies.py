"""Test a fixed set of linguistic strategies as an extension to P4."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from paired_model_comparison import DATA, IDENTIFIERS, OUT, input_fingerprint, paired
from train_p2_plus_reply_timing_and_external_link import FEATURES as P4_FEATURES, MODEL as P4_MODEL
from train_p5_topic_interactions import DRAWS, METRICS, discussion_weights, scores as prediction_metrics

P4_RUN = OUT / P4_MODEL / "20261120T220731301773Z"
STRATEGY_RUN = DATA / "paired-linguistic-strategies/20261121T203952524644Z"
MODEL = "p6_adjusted_linguistic_strategies"
STRATEGIES = (
    "net_sentiment", "op_cosine", "strategy::Direct start", "strategy::1st person pl.",
    "strategy::Factuality", "strategy::Direct question",
)
ADDITIONS = [f"a_minus_b_{feature}" for feature in STRATEGIES]
FEATURES = [*P4_FEATURES, *ADDITIONS]


def load_inputs():
    p4_report = json.loads((P4_RUN / "results.json").read_text())
    strategy_report = json.loads((STRATEGY_RUN / "results.json").read_text())
    if p4_report["split"] != "train" or strategy_report["split"] != "train" or p4_report["features"] != P4_FEATURES:
        raise ValueError("Expected the frozen P4 and strategy training runs.")
    p4 = pd.read_parquet(P4_RUN / "oof_predictions.parquet")
    frozen = pd.read_parquet(DATA / "training_pair_assignments.parquet", columns=IDENTIFIERS)
    pd.testing.assert_frame_equal(p4[IDENTIFIERS], frozen)
    input_path = STRATEGY_RUN / "pair_strategy_scores.parquet"
    columns = [f"{side}_{feature}" for feature in STRATEGIES for side in ("a", "b")]
    strategy = pd.read_parquet(input_path, columns=[*IDENTIFIERS, *columns])
    if not strategy["pair_id"].is_unique or len(strategy) != len(p4):
        raise ValueError("Expected one strategy row for every archived P4 pair.")
    strategy = strategy.set_index("pair_id").loc[p4["pair_id"]].reset_index()
    pd.testing.assert_frame_equal(p4[IDENTIFIERS], strategy[IDENTIFIERS])
    difference = pd.DataFrame({
        name: strategy[f"a_{feature}"] - strategy[f"b_{feature}"]
        for feature, name in zip(STRATEGIES, ADDITIONS)
    })
    complete = p4[IDENTIFIERS + P4_FEATURES].join(difference)
    if not np.isfinite(complete[FEATURES].to_numpy()).all():
        raise ValueError("P6 requires the same complete training cohort as P4.")
    return complete, p4["oof_probability_a_wins"], input_fingerprint(input_path)


def summarize_predictions(pairs, p4, p6):
    target = pairs["a_wins"].to_numpy()
    base, extended = prediction_metrics(target, p4), prediction_metrics(target, p6)
    draws = np.stack([
        prediction_metrics(target, p6, weights) - prediction_metrics(target, p4, weights)
        for weights in discussion_weights(pairs["post_id"])
        if weights[target == 0].any() and weights[target == 1].any()
    ])
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return pd.DataFrame({
        "metric": METRICS, "p4": base, "p6": extended,
        "p6_minus_p4": extended - base, "ci_low": low, "ci_high": high,
    })


def adjusted_coefficients(pairs, features=FEATURES):
    """Refit on resampled discussions while keeping the full-training feature scale fixed."""
    x = pairs[features]
    y = pairs["a_wins"].to_numpy()
    pipeline = paired.make_paired_classifier().fit(x, y)
    scaler = pipeline.named_steps["standardscaler"]
    scaled = scaler.transform(x)
    point = pipeline.named_steps["logisticregression"].coef_[0].copy()
    model = paired.make_paired_classifier().named_steps["logisticregression"]
    draws = np.stack([
        model.fit(scaled, y, sample_weight=weights).coef_[0].copy()
        for weights in discussion_weights(pairs["post_id"])
    ])
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    informative = {feature: pairs[feature].abs().gt(paired.TIE_TOLERANCE) for feature in features}
    return pd.DataFrame({
        "feature": features, "coefficient": point, "ci_low": low, "ci_high": high,
        "training_feature_sd": scaler.scale_,
        "informative_pairs": [int(mask.sum()) for mask in informative.values()],
        "informative_discussions": [pairs.loc[mask, "post_id"].nunique() for mask in informative.values()],
    })


def main():
    pairs, baseline, strategy_hash = load_inputs()
    with threadpool_limits(limits=1):
        reproduced, _ = paired.evaluate_differences(pairs, pairs[P4_FEATURES])
        np.testing.assert_allclose(reproduced, baseline, atol=1e-12, rtol=1e-10)
        predictions, _ = paired.evaluate_differences(pairs, pairs[FEATURES])
        metrics = summarize_predictions(pairs, baseline.to_numpy(), predictions)
        coefficients = adjusted_coefficients(pairs)
    output = OUT / MODEL
    output.mkdir(parents=True, exist_ok=True)
    destination = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        pairs.assign(p4_oof_probability=baseline, p6_oof_probability=predictions).to_parquet(
            stage / "oof_predictions.parquet", index=False,
        )
        metrics.to_csv(stage / "predictive_comparison.csv", index=False)
        coefficients.to_csv(stage / "adjusted_coefficients.csv", index=False)
        (stage / "results.json").write_text(json.dumps({
            "model": MODEL, "split": "train", "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "folds": paired.FOLDS, "features": FEATURES, "baselines": P4_FEATURES, "added_features": ADDITIONS,
            "p4_run": P4_RUN.name, "p4_predictions_sha256": input_fingerprint(P4_RUN / "oof_predictions.parquet"),
            "strategy_run": STRATEGY_RUN.name, "strategy_input_sha256": strategy_hash,
            "bootstrap_draws": DRAWS, "seed": paired.SEED,
        }, indent=2) + "\n")
        stage.rename(destination)
    print(metrics.round(5).to_string(index=False))
    print(coefficients.round(4).to_string(index=False))
    print(f"Saved P6 comparison to {destination}")


if __name__ == "__main__":
    main()
