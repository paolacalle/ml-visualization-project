"""Compare frozen P6 with P6 plus contextual politeness on the training folds."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_politeness import (
    DRAWS, IDENTIFIERS, METRICS, PHASE2, discussion_weights, extra_precision,
    load_inputs, paired, scores,
)
from paired_model_comparison import OUT, input_fingerprint
from train_p7_p6_plus_goemotions import P6_FEATURES, P6_RUN

MODEL = "p6_plus_contextual_politeness"
CONTENT_LIMIT = 256
FEATURE = "a_minus_b_polite_guard_256"


def compare_predictions(pairs, baseline, extended):
    outcome = pairs["a_wins"].to_numpy()

    def metrics(probabilities, weights):
        return np.array([*scores(outcome, probabilities, weights), extra_precision(outcome, probabilities, weights)])

    weights = np.ones(len(pairs))
    base_scores = metrics(baseline, weights)
    extended_scores = metrics(extended, weights)
    draws = np.stack([
        metrics(extended, sample_weights) - metrics(baseline, sample_weights)
        for sample_weights in discussion_weights(pairs["post_id"])
    ])
    lower, upper = np.nanquantile(draws, [.025, .975], axis=0)
    return pd.DataFrame({
        "metric": [*METRICS, "Precision at 0.5"], "p6": base_scores,
        "p6_plus_politeness": extended_scores, "change": extended_scores - base_scores,
        "ci_low": lower, "ci_high": upper,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run", type=Path)
    args = parser.parse_args()
    phase2 = args.phase2_run or sorted(PHASE2.glob("*/experiment_metadata.json"))[-1].parent
    pairs, _, wide, provenance = load_inputs(phase2)
    score = wide["politeness_score"][CONTENT_LIMIT].astype(float)
    input_path = P6_RUN / "oof_predictions.parquet"
    report = json.loads((P6_RUN / "results.json").read_text())
    if report["split"] != "train" or report["features"] != P6_FEATURES:
        raise ValueError("The archived P6 training run has different features or provenance.")
    archived = pd.read_parquet(input_path)
    pd.testing.assert_frame_equal(archived[IDENTIFIERS], pairs[IDENTIFIERS])
    features = archived[P6_FEATURES].assign(**{
        FEATURE: pairs["a_comment_id"].map(score) - pairs["b_comment_id"].map(score),
    })
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Expected complete P6 and politeness features on the same pairs.")

    with threadpool_limits(limits=1):
        baseline, _ = paired.evaluate_differences(pairs, features[P6_FEATURES])
        np.testing.assert_allclose(baseline, archived["p6_oof_probability"], atol=1e-12, rtol=1e-10)
        extended, _ = paired.evaluate_differences(pairs, features)
        comparison = compare_predictions(pairs, baseline, extended)

    truth = pairs["a_wins"].to_numpy()
    base_correct = (baseline >= .5) == truth
    extended_correct = (extended >= .5) == truth
    transitions = {
        "new_correct": int((~base_correct & extended_correct).sum()),
        "new_incorrect": int((base_correct & ~extended_correct).sum()),
    }
    output = OUT / MODEL
    output.mkdir(parents=True, exist_ok=True)
    destination = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        comparison.to_csv(stage / "predictive_comparison.csv", index=False)
        pairs[IDENTIFIERS].assign(
            **{FEATURE: features[FEATURE]}, p6_oof_probability=baseline, extended_oof_probability=extended,
        ).to_parquet(stage / "oof_predictions.parquet", index=False)
        (stage / "results.json").write_text(json.dumps({
            "model": MODEL, "split": "train", "p6_run": P6_RUN.name, "p6_oof_reproduced": True,
            "p6_predictions_sha256": input_fingerprint(input_path),
            "phase2_run": phase2.name, "content_limit": CONTENT_LIMIT, "added_feature": FEATURE,
            "features": list(features), "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "folds": paired.FOLDS, "seed": paired.SEED, "bootstrap_draws": DRAWS,
            "accuracy_transitions": transitions, **provenance,
        }, indent=2) + "\n")
        stage.rename(destination)
    print(comparison.round(5).to_string(index=False))
    print(json.dumps(transitions))
    print(f"Saved comparison to {destination}")


if __name__ == "__main__":
    main()
