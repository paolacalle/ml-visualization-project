"""Fit the frozen paired model specifications on training and score published held-out pairs."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score

from paired_model_comparison import CONTENT_ZIPF, WORD_COUNT, input_fingerprint, load_differences, paired


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lexical-complexity"))
from analyze_word_frequency_sensitivity import measure_variants

OUT = paired.DATA / "paired-heldout-evaluation"
P2_TRAINING_RUN = (
    paired.DATA / "paired-models/p2_word_count_and_content_word_zipf_differences/20261120T201405034740Z"
)
MODELS = {"P0": [WORD_COUNT], "P1": [CONTENT_ZIPF], "P2": [WORD_COUNT, CONTENT_ZIPF]}
BOOTSTRAP_DRAWS = 1000


def load_training():
    specification = json.loads((P2_TRAINING_RUN / "results.json").read_text())
    if specification["split"] != "train" or specification["features"] != MODELS["P2"]:
        raise ValueError("The archived P2 specification must use the frozen training features.")
    source = paired.DATA / "word-frequency-sensitivity" / specification["input_run"]
    report = json.loads((source / "results.json").read_text())
    if report["split"] != "train":
        raise ValueError("Expected a training input.")
    for package, key in (("nltk", "nltk_version"), ("wordfreq", "wordfreq_version"),
                         ("scikit-learn", "sklearn_version")):
        if version(package) != report[key]:
            raise ValueError(f"Use the archived training version of {package}.")
    input_path = source / "paired_scores.parquet"
    if input_fingerprint(input_path) != specification["input_sha256"]:
        raise ValueError("The P2 training data differs from the frozen source.")
    frame, features = load_differences(input_path)
    if len(frame) != specification["pairs"]:
        raise ValueError("P2 must use the archived training cohort.")
    return frame, features, specification


def load_heldout(train):
    pair_analysis = paired.import_script("heldout_pairs", paired.PAIR_SCRIPT)
    builder = paired.import_script("branch_builder", pair_analysis.BUILDER)
    metadata = builder.manifest(builder.METADATA)
    text = builder.manifest(builder.TEXT)
    archive = metadata["raw_archive"]["sha256"]

    # Check all published pair posts, including records excluded later for missing text.
    posts = {
        split: {row["post_id"] for row in builder.rows_for(metadata, "pairs", ["post_id"], split=split)}
        for split in ("train", "heldout")
    }
    if posts["train"] & posts["heldout"]:
        raise ValueError("Published train and held-out pairs share an OP discussion.")
    total, exclusions, pairs = pair_analysis.load_pairs(split="heldout")
    frame = paired.assign_pair_sides(pairs)
    ids = set(frame["a_comment_id"]) | set(frame["b_comment_id"])
    training_ids = set(train["a_comment_id"]) | set(train["b_comment_id"])
    if ids & training_ids or set(train["pair_id"]) & set(frame["pair_id"]):
        raise ValueError("Held-out openings or pairs overlap with P2 training data.")

    zipf = {}
    for row in builder.text_rows(text["files"], ids, "heldout"):
        comment_id = row["comment_id"]
        if comment_id in zipf:
            raise ValueError(f"Duplicate held-out text for {comment_id}.")
        zipf[comment_id] = measure_variants(row["body"])["content_words"]
    for side in ("a", "b"):
        frame[f"{side}_mean_content_word_zipf"] = frame[f"{side}_comment_id"].map(zipf)
    complete = frame[["a_mean_content_word_zipf", "b_mean_content_word_zipf"]].notna().all(axis=1)
    if not complete.all():
        exclusions["missing mean content-word Zipf"] = int((~complete).sum())
    frame = frame.loc[complete].reset_index(drop=True)
    if frame.empty or frame["a_wins"].nunique() < 2:
        raise ValueError("The held-out feature cohort must contain both A/B outcomes.")
    frame[WORD_COUNT] = frame["a_word_count"] - frame["b_word_count"]
    frame[CONTENT_ZIPF] = frame["a_mean_content_word_zipf"] - frame["b_mean_content_word_zipf"]
    if len(frame) + sum(exclusions.values()) != total:
        raise ValueError("Held-out exclusions must reconcile to the published pair count.")
    return frame, {
        "published_pairs": total, "published_discussions": len(posts["heldout"]),
        "pairs": len(frame), "discussions": frame["post_id"].nunique(),
        "exclusions": exclusions, "archive_sha256": archive,
    }


def metrics(y, probability, weights=None):
    return np.array([
        roc_auc_score(y, probability, sample_weight=weights),
        average_precision_score(y, probability, sample_weight=weights),
        accuracy_score(y, probability >= 0.5, sample_weight=weights),
        log_loss(y, probability, sample_weight=weights, labels=[0, 1]),
    ])


METRICS = ("roc_auc", "average_precision", "pairwise_accuracy", "log_loss")


def summarize(frame):
    y = frame["a_wins"].to_numpy()
    predictions = {name: frame[f"{name}_probability_a_wins"].to_numpy() for name in MODELS}
    point = {name: metrics(y, probability) for name, probability in predictions.items()}
    _, post_codes = np.unique(frame["post_id"], return_inverse=True)
    n_posts = frame["post_id"].nunique()
    probabilities = np.full(n_posts, 1 / n_posts)
    rng = np.random.default_rng(paired.SEED)
    replicate_rows = []
    for _ in range(BOOTSTRAP_DRAWS):
        weights = rng.multinomial(n_posts, probabilities)[post_codes]
        if not weights[y == 0].any() or not weights[y == 1].any():
            continue
        replicate_rows.append([metrics(y, probability, weights) for probability in predictions.values()])
    if not replicate_rows:
        raise ValueError("No bootstrap samples contained both A/B outcomes.")
    draws = np.array(replicate_rows)

    def intervals(values, bootstrap):
        limits = np.quantile(bootstrap, [0.025, 0.975], axis=0).T
        return {
            key: {"value": float(value), "ci_95": bounds.tolist()}
            for key, value, bounds in zip(METRICS, values, limits)
        }

    return {
        "bootstrap_draws": len(draws),
        "a_win_prevalence": float(y.mean()),
        "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
        "models": {name: intervals(point[name], draws[:, i]) for i, name in enumerate(MODELS)},
        "p2_minus_baseline": {
            name: intervals(point["P2"] - point[name], draws[:, 2] - draws[:, i])
            for i, name in enumerate(("P0", "P1"))
        },
    }


def main():
    training, training_features, specification = load_training()
    # Fit every model before loading held-out features or labels.
    fitted = {name: paired.make_paired_classifier().fit(training_features[features], training["a_wins"])
              for name, features in MODELS.items()}
    heldout, audit = load_heldout(training)
    for name, features in MODELS.items():
        heldout[f"{name}_probability_a_wins"] = fitted[name].predict_proba(heldout[features])[:, 1]
    report = {
        "split": "heldout", "seed": paired.SEED, "training_pairs": len(training),
        "p2_training_run": P2_TRAINING_RUN.name, "training_input_run": specification["input_run"],
        "training_input_sha256": specification["input_sha256"], "features": MODELS,
        **audit, **summarize(heldout),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        heldout.to_parquet(staging / "heldout_predictions.parquet", index=False)
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(destination)
    print(json.dumps(report, indent=2))
    print(f"Saved held-out evaluation to {destination}")


if __name__ == "__main__":
    main()
