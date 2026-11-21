"""Test the continuous GoEmotions neutral score as the sole addition to P6."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits

from paired_model_comparison import DATA, OUT, input_fingerprint, paired
from train_p7_p6_plus_goemotions import DRAWS, P6_FEATURES, P6_RUN, load_inputs, summarize_predictions


MODEL = "p7_p6_plus_continuous_neutral"
EMOTION_RUN = DATA / "training-opening-emotions" / "20261121T221841717125Z"
FEATURE = "a_minus_b_goemotions_neutral"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emotion-run", type=Path, default=EMOTION_RUN)
    args = parser.parse_args()
    pairs, (a, b), close, emotion = load_inputs(args.emotion_run)
    neutral = (a["score_neutral"] - b["score_neutral"]).to_frame(FEATURE)
    features = pairs[P6_FEATURES].join(neutral)
    winner_difference = neutral[FEATURE].mul(2 * pairs["a_wins"] - 1)
    with threadpool_limits(limits=1):
        p6, _ = paired.evaluate_differences(pairs, pairs[P6_FEATURES])
        np.testing.assert_allclose(p6, pairs["p6_oof_probability"], atol=1e-12, rtol=1e-10)
        e0, _ = paired.evaluate_differences(pairs, neutral)
        p7, _ = paired.evaluate_differences(pairs, features)
        metrics = summarize_predictions(pairs, {"E0 neutral": e0, "P6": p6, "P7 neutral": p7})
        association = {
            "All training pairs": paired.paired_higher(pairs, winner_difference),
            "Within 50 words": paired.paired_higher(pairs.loc[close], winner_difference.loc[close]),
        }

    output = OUT / MODEL
    output.mkdir(parents=True, exist_ok=True)
    run = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        pairs.join(neutral).assign(e0_oof_probability=e0, p7_oof_probability=p7).to_parquet(
            stage / "oof_predictions.parquet", index=False,
        )
        metrics.to_csv(stage / "predictive_comparison.csv", index=False)
        (stage / "results.json").write_text(json.dumps({
            "model": MODEL, "split": "train", "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "folds": paired.FOLDS, "seed": paired.SEED, "bootstrap_draws": DRAWS,
            "features_by_model": {"E0 neutral": [FEATURE], "P6": P6_FEATURES, "P7 neutral": list(features)},
            "neutral_score_association": association,
            "p6_run": P6_RUN.name, "p6_oof_reproduced": True,
            "p6_predictions_sha256": input_fingerprint(P6_RUN / "oof_predictions.parquet"),
            "emotion_run": args.emotion_run.name, "emotion": emotion,
            "emotion_scores_sha256": input_fingerprint(args.emotion_run / "opening_emotion_scores.parquet"),
        }, indent=2) + "\n")
        stage.rename(run)
    print(metrics.round(5).to_string(index=False))
    print(run)


if __name__ == "__main__":
    main()
