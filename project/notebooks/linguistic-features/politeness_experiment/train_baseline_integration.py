"""Train P6 and its contextual-politeness ablations on the frozen training folds."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_p6_politeness import P6_FEATURES, P6_RUN
from evaluate_politeness import IDENTIFIERS, PHASE2, load_inputs, paired
from paired_model_comparison import OUT, input_fingerprint


MODEL = "p6_politeness_baseline_integration"
CONVOKIT_FEATURES = [
    "a_minus_b_strategy::Direct start",
    "a_minus_b_strategy::1st person pl.",
    "a_minus_b_strategy::Factuality",
    "a_minus_b_strategy::Direct question",
]
CONTEXTUAL_FEATURES = {size: f"a_minus_b_polite_guard_{size}" for size in (128, 256, 510)}


def build_specifications():
    inherited = [feature for feature in P6_FEATURES if feature not in CONVOKIT_FEATURES]
    if set(P6_FEATURES) - set(inherited) != set(CONVOKIT_FEATURES):
        raise ValueError("The archived P6 specification must include the four ConvoKit indicators.")
    return {
        "P6": P6_FEATURES,
        "P6a": inherited,
        "P6b": [*inherited, CONTEXTUAL_FEATURES[128]],
        "P6c": [*inherited, CONTEXTUAL_FEATURES[256]],
        "P6d": [*inherited, CONTEXTUAL_FEATURES[510]],
        "P6e": [*P6_FEATURES, CONTEXTUAL_FEATURES[256]],
    }


def pair_features(phase2):
    pairs, _, opening_scores, provenance = load_inputs(phase2)
    p6_path = P6_RUN / "oof_predictions.parquet"
    recipe = json.loads((P6_RUN / "results.json").read_text())
    if recipe["split"] != "train" or recipe["features"] != P6_FEATURES:
        raise ValueError("Expected the frozen P6 training specification.")
    archive = pd.read_parquet(p6_path)
    pd.testing.assert_frame_equal(archive[IDENTIFIERS], pairs[IDENTIFIERS])
    features = archive[P6_FEATURES].copy()
    for size, name in CONTEXTUAL_FEATURES.items():
        values = opening_scores["politeness_score"][size].astype(float)
        features[name] = pairs["a_comment_id"].map(values) - pairs["b_comment_id"].map(values)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Each fitted model must use the same complete training pairs.")
    provenance["p6_predictions_sha256"] = input_fingerprint(p6_path)
    return pairs, features, archive["p6_oof_probability"], provenance


def train_models(pairs, features, specifications):
    predictions = pairs[IDENTIFIERS].copy()
    fitted = []
    for name, columns in specifications.items():
        design = features[columns]
        oof = np.full(len(pairs), np.nan)
        for fold, validation, pipeline in paired.fit_paired_folds(pairs, design):
            oof[validation.to_numpy()] = pipeline.predict_proba(design.loc[validation])[:, 1]
            scale = pipeline.named_steps["standardscaler"]
            logistic = pipeline.named_steps["logisticregression"]
            fitted.append({
                "model": name, "fold": fold, "features": columns,
                "mean": scale.mean_.tolist(), "scale": scale.scale_.tolist(),
                "coefficient": logistic.coef_[0].tolist(), "intercept": logistic.intercept_.item(),
            })
        if not np.isfinite(oof).all():
            raise ValueError(f"Every pair must receive an out-of-fold prediction for {name}.")
        predictions[name] = oof
    return predictions, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run", type=Path)
    args = parser.parse_args()
    phase2 = args.phase2_run or sorted(PHASE2.glob("*/experiment_metadata.json"))[-1].parent
    pairs, features, p6, provenance = pair_features(phase2)
    specifications = build_specifications()
    with threadpool_limits(limits=1):
        predictions, fitted = train_models(pairs, features, specifications)
    np.testing.assert_allclose(predictions["P6"], p6, atol=1e-12, rtol=1e-10)

    output = OUT / MODEL
    output.mkdir(parents=True, exist_ok=True)
    destination = output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = {
        "experiment": MODEL, "split": "train", "pairs": len(pairs),
        "discussions": pairs["post_id"].nunique(), "folds": paired.FOLDS, "seed": paired.SEED,
        "p6_run": P6_RUN.name, "p6_oof_reproduced": True, "phase2_run": phase2.name,
        "ablation_removed_features": CONVOKIT_FEATURES, "specifications": specifications, **provenance,
    }
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        predictions.to_parquet(stage / "oof_predictions.parquet", index=False)
        (stage / "fold_models.json").write_text(json.dumps(fitted, indent=2) + "\n")
        (stage / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(destination)
    print(f"Trained {len(specifications)} models on {len(pairs):,} pairs using {paired.FOLDS} grouped folds.")
    print(f"Saved {len(fitted)} fitted fold models and out-of-fold probabilities to {destination}")


if __name__ == "__main__":
    main()
