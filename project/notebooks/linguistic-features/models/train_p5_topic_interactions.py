"""Compare symmetric P4 and partially pooled topic interactions on training folds."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from paired_model_comparison import DATA, IDENTIFIERS, input_fingerprint, paired
from train_p2_plus_reply_timing_and_external_link import FEATURES, MODEL as P4_MODEL

P4_RUN = DATA / "paired-models" / P4_MODEL / "20261120T220731301773Z"
TOPIC_RUN = DATA / "op-topic-discovery/20261120T221529517368Z"
OUT = DATA / "paired-models/p5_topic_interactions"
TOPICS = (
    "Arts, Media & Sports", "Education & Work", "Ethics & Religion", "Politics & Law",
    "Science & Health", "Society & Identity", "Technology & Economics",
)
OTHER = "Other / Unclear"
TIMING = FEATURES.index("a_minus_b_log1p_reply_delay_minutes")
MODELS = {"P4": [], "P5a": list(range(len(FEATURES))), "P5b": [TIMING]}
MIN_EXAMPLES = 3
COSINE_MARGIN = 0.025
PENALTY_RATIO = 4
INTERACTION_SCALE = 1 / np.sqrt(PENALTY_RATIO)
DRAWS = 1000
METRICS = ("ROC-AUC", "Average precision", "Pairwise accuracy", "Log loss")
CONTRAST = ("Technology & Economics", "Politics & Law")


def load_inputs():
    baseline = json.loads((P4_RUN / "results.json").read_text())
    discovery = json.loads((TOPIC_RUN / "results.json").read_text())
    if (baseline["split"] != "train" or discovery["split"] != "train"
            or baseline["features"] != FEATURES or discovery["p4_run"] != P4_RUN.name):
        raise ValueError("Expected the archived P4 training cohort and its training OP embeddings.")
    pairs = pd.read_parquet(P4_RUN / "oof_predictions.parquet")
    frozen = pd.read_parquet(DATA / "training_pair_assignments.parquet", columns=IDENTIFIERS)
    pd.testing.assert_frame_equal(pairs[IDENTIFIERS], frozen)
    if not pairs.groupby("post_id")["fold"].nunique().eq(1).all():
        raise ValueError("Each OP must belong to one validation fold.")
    # The OP parquet supplies the row order of the individually encoded document vectors.
    # BERTopic labels are deliberately absent from the model inputs.
    posts = pd.read_parquet(TOPIC_RUN / "op_topic_assignments.parquet", columns=["post_id"])
    embeddings = np.load(TOPIC_RUN / "op_embeddings.npy")
    if (len(embeddings) != len(posts) or not posts["post_id"].is_unique
            or hashlib.sha256(embeddings.tobytes()).hexdigest() != discovery["embedding_sha256"]):
        raise ValueError("OP identifiers must align to the archived embeddings.")
    fold = pairs.groupby("post_id")["fold"].first()
    if set(posts["post_id"]) != set(fold.index):
        raise ValueError("The embeddings and P4 cohort must contain the same OPs.")
    manual_path = TOPIC_RUN / "manual_title_topics.csv"
    labels = pd.read_csv(manual_path, usecols=["post_id", "manual_topic"]).set_index("post_id")["manual_topic"]
    if (not labels.index.is_unique or not set(labels.index) <= set(posts["post_id"])
            or not set(labels) <= {*TOPICS, OTHER}):
        raise ValueError("Expected at most one valid manual label per training discussion.")
    posts["fold"] = posts["post_id"].map(fold)
    return pairs, posts, embeddings, labels


def assign_topics(embeddings, manual, fit, score):
    """Fit named cosine centroids on permitted manual labels and abstain on close calls."""
    eligible = [(topic, embeddings[fit & (manual == topic)]) for topic in TOPICS]
    centers = [(topic, vectors.mean(axis=0)) for topic, vectors in eligible if len(vectors) >= MIN_EXAMPLES]
    predicted = np.full(len(score), OTHER, dtype=object)
    if len(centers) < 2:
        return predicted
    names, vectors = zip(*centers)
    similarities = embeddings[score] @ normalize(np.stack(vectors)).T
    order = np.argsort(similarities, axis=1, kind="stable")[:, -2:]
    second, first = order.T
    row = np.arange(len(score))
    confident = similarities[row, first] - similarities[row, second] >= COSINE_MARGIN
    predicted[confident] = np.asarray(names)[first[confident]]
    return predicted


def design(scaled, topics, interactions):
    blocks = [
        scaled[:, interactions] * (topics == topic)[:, None] * INTERACTION_SCALE
        for topic in TOPICS if interactions
    ]
    return np.column_stack([scaled, *blocks])


def classifier(x, y, weights=None):
    return LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000, tol=1e-7).fit(x, y, sample_weight=weights)


def cross_validate(pairs, posts, embeddings, labels):
    x = pairs[FEATURES].to_numpy()
    y = pairs["a_wins"].to_numpy()
    folds = posts["fold"].to_numpy()
    manual = posts["post_id"].map(labels).fillna("").to_numpy()
    op_index = pd.Index(posts["post_id"]).get_indexer(pairs["post_id"])
    out = {name: np.full(len(pairs), np.nan) for name in MODELS}
    assigned = np.full(len(posts), "", dtype=object)
    audit = []
    for fold in range(1, paired.FOLDS + 1):
        valid = folds == fold
        valid_pair = valid[op_index]
        fit_manual = ~valid & np.isin(manual, TOPICS)
        topics = assign_topics(embeddings, manual, fit_manual, np.arange(len(posts)))
        # Use leave-one-out assignments for labeled OPs entering the outcome model fit.
        for index in np.flatnonzero(fit_manual):
            without_self = fit_manual.copy()
            without_self[index] = False
            topics[index] = assign_topics(embeddings, manual, without_self, np.array([index]))[0]
        assigned[valid] = topics[valid]
        topic_pairs = topics[op_index]
        scaler = StandardScaler(with_mean=False).fit(x[~valid_pair])
        scaled = scaler.transform(x)
        for name, interactions in MODELS.items():
            matrix = design(scaled, topic_pairs, interactions)
            model = classifier(matrix[~valid_pair], y[~valid_pair])
            validation = matrix[valid_pair]
            prob = model.predict_proba(validation)[:, 1]
            np.testing.assert_allclose(prob + model.predict_proba(-validation)[:, 1], 1, atol=1e-12)
            out[name][valid_pair] = prob
        audit.append({"fold": fold, "topic_training_labels": int(fit_manual.sum()),
                      "validation_discussions": int(valid.sum()), "validation_pairs": int(valid_pair.sum())})

    if np.any(assigned == "") or not all(np.isfinite(p).all() for p in out.values()):
        raise ValueError("Each training OP and pair must receive one out-of-fold prediction.")
    topics = posts.assign(manual_topic=manual, oof_topic=assigned)
    predictions = pairs[IDENTIFIERS].assign(oof_topic=assigned[op_index], **out)
    predictions["Archived P4"] = pairs["oof_probability_a_wins"]
    return predictions, topics, audit


def scores(y, p, weights=None):
    return np.array([
        roc_auc_score(y, p, sample_weight=weights),
        average_precision_score(y, p, sample_weight=weights),
        accuracy_score(y, p >= 0.5, sample_weight=weights),
        log_loss(y, p, sample_weight=weights, labels=[0, 1]),
    ])


def discussion_weights(post_ids):
    rng = np.random.default_rng(paired.SEED)
    codes, posts = pd.factorize(post_ids)
    probabilities = np.full(len(posts), 1 / len(posts))
    for _ in range(DRAWS):
        yield rng.multinomial(len(posts), probabilities)[codes]


def predictive_summary(predictions):
    y = predictions["a_wins"].to_numpy()
    names = [*MODELS, "Archived P4"]
    point = np.stack([scores(y, predictions[name]) for name in names])
    draws = np.stack([
        [scores(y, predictions[name], weights) for name in names]
        for weights in discussion_weights(predictions["post_id"])
        if weights[y == 0].any() and weights[y == 1].any()
    ])
    rows = []
    for index, name in enumerate(names):
        values, draws_for_model = point[index], draws[:, index]
        increment = values - point[0]
        limits = np.quantile(draws_for_model - draws[:, 0], [0.025, 0.975], axis=0)
        rows.extend({
            "model": name, "metric": metric, "value": values[i],
            "delta_vs_p4": increment[i], "delta_ci_low": limits[0, i], "delta_ci_high": limits[1, i],
        } for i, metric in enumerate(METRICS))
    return pd.DataFrame(rows)


def coefficients(pairs, predictions):
    """Refit penalized models in OP bootstrap samples using fixed cross-fitted labels and scale."""
    topics = predictions["oof_topic"].to_numpy()
    x = pairs[FEATURES].to_numpy()
    y = pairs["a_wins"].to_numpy()
    scaler = StandardScaler(with_mean=False).fit(x)
    scaled = scaler.transform(x)
    population_topics = ["Shared", *TOPICS, OTHER]
    rows, contrasts = [], []
    for name in ("P5a", "P5b"):
        interactions = MODELS[name]
        matrix = design(scaled, topics, interactions)

        def fit_weights(weights=None):
            coefs = classifier(matrix, y, weights).coef_[0]
            values = np.tile(coefs[:len(FEATURES)], (len(population_topics), 1))
            adjustments = coefs[len(FEATURES):].reshape(len(TOPICS), len(interactions))
            values[1:-1, interactions] += INTERACTION_SCALE * adjustments
            return values

        point = fit_weights()
        draws = np.stack([fit_weights(weights) for weights in discussion_weights(pairs["post_id"])])
        lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
        for j, topic in enumerate(population_topics):
            mask = np.ones(len(pairs), dtype=bool) if topic == "Shared" else topics == topic
            for k, feature in enumerate(FEATURES):
                informative = mask & (np.abs(x[:, k]) > paired.TIE_TOLERANCE)
                rows.append({
                    "model": name, "topic": topic, "feature": feature, "pairs": int(mask.sum()),
                    "informative_pairs": int(informative.sum()),
                    "informative_discussions": pairs.loc[informative, "post_id"].nunique(),
                    "coefficient": point[j, k], "ci_low": lower[j, k], "ci_high": upper[j, k],
                })
        first, second = (population_topics.index(topic) for topic in CONTRAST)
        low, high = np.quantile(draws[:, first, TIMING] - draws[:, second, TIMING], [0.025, 0.975])
        contrasts.append({
            "model": name, "contrast": f"{CONTRAST[0]} minus {CONTRAST[1]}",
            "estimate": point[first, TIMING] - point[second, TIMING], "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(rows), pd.DataFrame(contrasts), dict(zip(FEATURES, scaler.scale_.tolist()))


def main():
    pairs, posts, embeddings, labels = load_inputs()
    with threadpool_limits(limits=1):
        predictions, topics, folds = cross_validate(pairs, posts, embeddings, labels)
        metrics = predictive_summary(predictions)
        weights, contrasts, scales = coefficients(pairs, predictions)
    reviewed = topics.loc[topics["manual_topic"].ne("")].copy()
    named = reviewed["manual_topic"].ne(OTHER)
    assigned = reviewed["oof_topic"].ne(OTHER)
    match = reviewed["manual_topic"].eq(reviewed["oof_topic"])
    report = {
        "split": "train", "pairs": len(pairs), "discussions": len(posts), "folds": folds,
        "archived_p4_run": P4_RUN.name, "topic_run": TOPIC_RUN.name,
        "manual_labels_sha256": input_fingerprint(TOPIC_RUN / "manual_title_topics.csv"),
        "seed": paired.SEED, "bootstrap_draws": DRAWS, "penalty_ratio": PENALTY_RATIO,
        "minimum_examples_per_topic": MIN_EXAMPLES, "cosine_margin": COSINE_MARGIN,
        "coefficient_feature_scales": scales,
        "topic_diagnostics": {
            "manual_reviewed": len(reviewed), "manual_named": int(named.sum()),
            "correct_manual_reviewed": int(match.sum()), "correct_manual_named": int((named & match).sum()),
            "assigned_reviewed": int(assigned.sum()), "correct_when_assigned": int((assigned & match).sum()),
            "assigned_all": int(topics["oof_topic"].ne(OTHER).sum()),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temp:
        stage = Path(temp)
        predictions.to_parquet(stage / "oof_predictions.parquet", index=False)
        topics.to_parquet(stage / "oof_topics.parquet", index=False)
        metrics.to_csv(stage / "predictive_metrics.csv", index=False)
        weights.to_csv(stage / "topic_coefficients.csv", index=False)
        contrasts.to_csv(stage / "timing_contrasts.csv", index=False)
        (stage / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(destination)
    print(metrics.round(4).to_string(index=False))
    print(contrasts.round(4).to_string(index=False))
    print(f"Saved P5 comparison to {destination}")


if __name__ == "__main__":
    main()
