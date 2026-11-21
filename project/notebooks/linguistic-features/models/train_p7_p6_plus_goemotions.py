"""Compare emotion-only E0 and P6 plus GoEmotions on the frozen P6 training folds."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from paired_model_comparison import DATA, IDENTIFIERS, OUT, input_fingerprint, paired
from train_p5_topic_interactions import DRAWS, METRICS, discussion_weights, scores as prediction_metrics
from train_p6_adjusted_linguistic_strategies import FEATURES as P6_FEATURES, MODEL as P6_MODEL


MODEL = "p7_p6_plus_goemotions"
P6_RUN = OUT / P6_MODEL / "20261121T204840655027Z"
EMOTION_RUNS = DATA / "training-opening-emotions"
LABELS = ("joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral")
SCORES = [f"score_{label}" for label in LABELS]
DETECTIONS = [f"detected_{label}" for label in LABELS]
ADDITIONS = [f"a_minus_b_goemotions_{label}" for label in LABELS]


def load_inputs(emotion_run):
    baseline = json.loads((P6_RUN / "results.json").read_text())
    emotion = json.loads((emotion_run / "results.json").read_text())
    if baseline["split"] != "train" or baseline["features"] != P6_FEATURES or emotion["split"] != "train":
        raise ValueError("Expected the archived P6 training and full training opening emotion runs.")

    pairs = pd.read_parquet(P6_RUN / "oof_predictions.parquet")
    frozen = pd.read_parquet(DATA / "training_pair_assignments.parquet")
    pd.testing.assert_frame_equal(pairs[IDENTIFIERS], frozen[IDENTIFIERS])
    if not pairs.groupby("post_id")["fold"].nunique().eq(1).all():
        raise ValueError("The same discussion may not occur in multiple folds.")

    opening_scores = pd.read_parquet(emotion_run / "opening_emotion_scores.parquet", columns=SCORES + DETECTIONS)
    ids = set(pairs["a_comment_id"]) | set(pairs["b_comment_id"])
    if not opening_scores.index.is_unique or set(opening_scores.index) != ids:
        raise ValueError("Expected full emotion score coverage over the frozen P6 cohort.")
    if not np.isfinite(opening_scores[SCORES]).to_numpy().all():
        raise ValueError("P7 requires finite scores for every opening.")
    if not np.array_equal(opening_scores[SCORES].ge(emotion["threshold"]), opening_scores[DETECTIONS]):
        raise ValueError("Emotion detections must match the declared score threshold.")

    sides = [
        opening_scores.loc[pairs[f"{side}_comment_id"]].set_axis(pairs.index)
        for side in ("a", "b")
    ]
    close = (frozen["a_word_count"] - frozen["b_word_count"]).abs().le(50)
    return pairs, sides, close, emotion


def descriptive_summary(pairs, side_a, side_b, close):
    sign = 2 * pairs["a_wins"] - 1
    a_detected = side_a[DETECTIONS].astype(int).set_axis(LABELS, axis=1)
    b_detected = side_b[DETECTIONS].astype(int).set_axis(LABELS, axis=1)
    winner = a_detected.where(pairs["a_wins"].eq(1), b_detected, axis=0)
    comparison = b_detected.where(pairs["a_wins"].eq(1), a_detected, axis=0)
    prevalence = pd.DataFrame({
        "emotion": LABELS, "winner_count": winner.sum().to_numpy(),
        "comparison_count": comparison.sum().to_numpy(),
        "reply_appearances_per_role": len(pairs),
        "unique_openings": len(set(pairs["a_comment_id"]) | set(pairs["b_comment_id"])),
    })
    differences = {
        "detected": (winner - comparison),
        "continuous_score": (side_a[SCORES] - side_b[SCORES]).set_axis(LABELS, axis=1).mul(sign, axis=0),
    }
    results = []
    for cohort, subset in (("All training pairs", pairs), ("Within 50 words", pairs.loc[close])):
        for measure, difference in differences.items():
            for label in LABELS:
                result = paired.paired_higher(subset, difference.loc[subset.index, label])
                low, high = result.pop("ci_95")
                results.append({
                    "cohort": cohort, "measure": measure, "emotion": label,
                    "informative_pairs": result["winner_higher"] + result["winner_lower"],
                    **result, "ci_low": low, "ci_high": high,
                })
    return prevalence, pd.DataFrame(results)


def summarize_predictions(pairs, predictions):
    target = pairs["a_wins"].to_numpy()
    names = list(predictions)
    point = np.stack([prediction_metrics(target, predictions[name]) for name in names])
    draws = np.stack([
        [prediction_metrics(target, predictions[name], weights) for name in names]
        for weights in discussion_weights(pairs["post_id"])
        if weights[target == 0].any() and weights[target == 1].any()
    ])
    base = names.index("P6")
    rows = []
    for index, name in enumerate(names):
        low, high = np.quantile(draws[:, index] - draws[:, base], [0.025, 0.975], axis=0)
        for metric, value, delta, lcl, ucl in zip(METRICS, point[index], point[index] - point[base], low, high):
            rows.append({"model": name, "metric": metric, "value": value, "delta_vs_p6": delta,
                         "delta_ci_low": lcl, "delta_ci_high": ucl})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion-run", type=Path)
    args = parser.parse_args()
    runs = sorted(EMOTION_RUNS.glob("*/opening_emotion_scores.parquet"))
    if args.emotion_run is None and not runs:
        parser.error("Run score_training_opening_emotions.py or provide --emotion-run.")
    emotion_run = args.emotion_run if args.emotion_run else runs[-1].parent
    pairs, (a, b), close, emotion = load_inputs(emotion_run)
    differences = (a[SCORES] - b[SCORES]).set_axis(ADDITIONS, axis=1)
    features = pairs[P6_FEATURES].join(differences)
    with threadpool_limits(limits=1):
        p6, _ = paired.evaluate_differences(pairs, pairs[P6_FEATURES])
        np.testing.assert_allclose(p6, pairs["p6_oof_probability"], atol=1e-12, rtol=1e-10)
        e0, _ = paired.evaluate_differences(pairs, differences)
        p7, _ = paired.evaluate_differences(pairs, features)
        predictions = {"E0": e0, "P6": p6, "P7": p7}
        metrics = summarize_predictions(pairs, predictions)
        prevalence, associations = descriptive_summary(pairs, a, b, close)

    output = OUT / MODEL
    output.mkdir(parents=True, exist_ok=True)
    destination = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        pairs.join(differences).assign(e0_oof_probability=e0, p7_oof_probability=p7).to_parquet(
            stage / "oof_predictions.parquet", index=False,
        )
        metrics.to_csv(stage / "predictive_comparison.csv", index=False)
        prevalence.to_csv(stage / "emotion_prevalence.csv", index=False)
        associations.to_csv(stage / "paired_emotion_associations.csv", index=False)
        (stage / "results.json").write_text(json.dumps({
            "model": MODEL, "split": "train", "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "folds": paired.FOLDS, "seed": paired.SEED, "bootstrap_draws": DRAWS,
            "features_by_model": {"E0": ADDITIONS, "P6": P6_FEATURES, "P7": [*features]},
            "p6_run": P6_RUN.name, "p6_oof_reproduced": True,
            "p6_predictions_sha256": input_fingerprint(P6_RUN / "oof_predictions.parquet"),
            "emotion_run": emotion_run.name, "emotion": emotion,
            "emotion_scores_sha256": input_fingerprint(emotion_run / "opening_emotion_scores.parquet"),
        }, indent=2) + "\n")
        stage.rename(destination)
    print(metrics.round(5).to_string(index=False))
    print(f"Saved the P7 comparison to {destination}")


if __name__ == "__main__":
    main()
