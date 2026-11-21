"""Compare categorical tone against P6 on the frozen discussion-grouped training pairs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from paired_model_comparison import DATA, IDENTIFIERS, OUT, input_fingerprint, paired
from train_p7_p6_plus_goemotions import DRAWS, P6_FEATURES, P6_RUN, summarize_predictions


MODEL = "p7_p6_plus_categorical_tone"
EMOTION_RUN = DATA / "training-opening-emotions" / "20261121T221841717125Z"
CATEGORY_RULES = {
    "non_neutral": "At least one non-neutral label detected, including replies also flagged neutral.",
    "neutral_only": "Only the neutral label detected.",
    "other": "No label detected.",
}
ADDITIONS = ["a_minus_b_tone_non_neutral", "a_minus_b_tone_other"]


def load_inputs(emotion_run):
    p6_manifest = json.loads((P6_RUN / "results.json").read_text())
    score_manifest = json.loads((emotion_run / "results.json").read_text())
    if (p6_manifest["split"] != "train" or p6_manifest["features"] != P6_FEATURES
            or score_manifest["split"] != "train" or score_manifest["threshold"] != 0.5):
        raise ValueError("Expected archived P6 training inputs and emotion detections at 0.50.")

    pairs = pd.read_parquet(P6_RUN / "oof_predictions.parquet")
    frozen = pd.read_parquet(DATA / "training_pair_assignments.parquet", columns=IDENTIFIERS)
    pd.testing.assert_frame_equal(pairs[IDENTIFIERS], frozen)
    if not pairs.groupby("post_id")["fold"].nunique().eq(1).all():
        raise ValueError("All pairs from a discussion must share one fold.")

    opening_path = emotion_run / "opening_emotion_scores.parquet"
    scores = pd.read_parquet(opening_path)
    ids = set(pairs["a_comment_id"]) | set(pairs["b_comment_id"])
    if not scores.index.is_unique or set(scores.index) != ids:
        raise ValueError("Expected one detection record per unique training opening.")
    flags = scores.filter(regex="^detected_")
    if flags.shape[1] != 28 or flags.isna().to_numpy().any():
        raise ValueError("Expected complete binary detections for all 28 labels.")
    neutral = flags["detected_neutral"]
    non_neutral = flags.drop(columns="detected_neutral").any(axis=1)
    if not scores["no_label_detected"].equals(~(neutral | non_neutral)):
        raise ValueError("The no-label indicator must agree with the detection flags.")

    tone = pd.Series(
        np.select([non_neutral, neutral], ["non_neutral", "neutral_only"], default="other"),
        index=flags.index,
    )
    pairs = pairs.assign(a_tone=pairs["a_comment_id"].map(tone), b_tone=pairs["b_comment_id"].map(tone))
    differences = pd.DataFrame({
        name: pairs["a_tone"].eq(category).astype(int) - pairs["b_tone"].eq(category).astype(int)
        for name, category in zip(ADDITIONS, ("non_neutral", "other"))
    }, index=pairs.index)
    return pairs, differences, tone.value_counts().to_dict(), input_fingerprint(opening_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion-run", type=Path, default=EMOTION_RUN)
    args = parser.parse_args()
    pairs, differences, counts, score_hash = load_inputs(args.emotion_run)
    features = pairs[P6_FEATURES].join(differences)
    with threadpool_limits(limits=1):
        p6, _ = paired.evaluate_differences(pairs, pairs[P6_FEATURES])
        np.testing.assert_allclose(p6, pairs["p6_oof_probability"], atol=1e-12, rtol=1e-10)
        e0, _ = paired.evaluate_differences(pairs, differences)
        p7, _ = paired.evaluate_differences(pairs, features)
        metrics = summarize_predictions(pairs, {"E0 categorical": e0, "P6": p6, "P7 categorical": p7})

    output = OUT / MODEL
    output.mkdir(parents=True, exist_ok=True)
    run = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        pairs.join(differences).assign(e0_oof_probability=e0, p7_oof_probability=p7).to_parquet(
            stage / "oof_predictions.parquet", index=False,
        )
        metrics.to_csv(stage / "predictive_comparison.csv", index=False)
        (stage / "results.json").write_text(json.dumps({
            "model": MODEL, "split": "train", "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "folds": paired.FOLDS, "seed": paired.SEED, "bootstrap_draws": DRAWS,
            "features_by_model": {"E0 categorical": ADDITIONS, "P6": P6_FEATURES, "P7 categorical": list(features)},
            "threshold": 0.5, "category_rules": CATEGORY_RULES, "reference_category": "neutral_only",
            "unique_openings_by_category": counts,
            "p6_run": P6_RUN.name, "p6_oof_reproduced": True,
            "emotion_run": args.emotion_run.name, "emotion_scores_sha256": score_hash,
        }, indent=2) + "\n")
        stage.rename(run)
    print(metrics.round(5).to_string(index=False))
    print(run)


if __name__ == "__main__":
    main()
