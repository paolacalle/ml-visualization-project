"""Discover topics from unique training OP posts, blind to pair outcomes."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paired_evaluation as paired
from text_preprocessing import minimal_text


OUT = paired.DATA / "op-topic-discovery"
P4 = paired.DATA / "paired-models/p2_plus_reply_timing_and_external_link/20261120T220731301773Z"
BUILDER = paired.NOTEBOOKS / "nested-replies-analysis/op_engagement_study/build_branch_dataset.py"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MIN_TOPIC_SIZE = 40
BODY_WEIGHT = 0.6
TITLE_SAMPLE_SIZE = 100


def load_posts():
    report = json.loads((P4 / "results.json").read_text())
    if report["split"] != "train":
        raise ValueError("Expected a training-only P4 reference.")
    predictions = P4 / "oof_predictions.parquet"
    posts = pd.read_parquet(predictions, columns=["post_id"])["post_id"]
    frozen = pd.read_parquet(paired.DATA / "training_pair_assignments.parquet", columns=["post_id"])["post_id"]
    pd.testing.assert_series_equal(posts, frozen)
    builder = paired.import_script("branch_builder", BUILDER)
    manifest = builder.manifest(builder.TEXT)
    shards = [
        pq.ParquetFile(builder.TEXT / entry["path"]).read(columns=["post_id", "title", "body"]).to_pandas()
        for entry in manifest["files"] if entry["path"].startswith("posts/split=train/")
    ]
    frame = pd.concat(shards, ignore_index=True)
    # This join checks the train partition and pins one text record per OP.
    frame = posts.drop_duplicates().sort_values().to_frame().merge(
        frame, on="post_id", how="left", validate="one_to_one", indicator=True,
    )
    if frame["_merge"].ne("both").any():
        raise ValueError("Some archived P4 discussion IDs lack a training OP text row.")
    frame = frame.drop(columns="_merge")
    clean_title = frame["title"].fillna("").str.replace(r"(?i)^\s*(?:\[\s*)?cmv(?:\s*\])?\s*[:\-–—]?\s*", "", regex=True)
    frame["title_clean"] = clean_title.map(minimal_text)
    frame["body_clean"] = frame["body"].map(lambda body: minimal_text("\n".join(builder.opening_text_lines(body))))
    return frame, manifest["raw_archive"]["sha256"]


def embed(posts, model):
    tokenizer = model.tokenizer
    width = model.max_seq_length - 16
    items, owner, weights = [], [], []
    titles = posts["title_clean"].tolist()
    for index, body in enumerate(posts["body_clean"]):
        if not body.strip():
            continue
        offsets = tokenizer(body, add_special_tokens=False, truncation=False, return_offsets_mapping=True)["offset_mapping"]
        for start in range(0, len(offsets), width):
            chunk = offsets[start:start + width]
            items.append(body[chunk[0][0]:chunk[-1][1]])
            owner.append(index)
            weights.append(len(chunk))
    if not items:
        raise ValueError("Expected at least some available OP bodies.")
    print(f"Encoding {len(titles):,} OP titles and {len(items):,} body chunks.", flush=True)
    title_vectors = model.encode(titles, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    vectors = model.encode(items, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    body_vectors = np.zeros_like(title_vectors)
    np.add.at(body_vectors, owner, vectors * np.asarray(weights, dtype=np.float32)[:, None])
    body_vectors = normalize(body_vectors)
    title_available = posts["title_clean"].str.strip().ne("").to_numpy()[:, None]
    combined = (1 - BODY_WEIGHT) * title_vectors * title_available + BODY_WEIGHT * body_vectors
    return normalize(combined), len(items)


def clustering(posts, embeddings):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=paired.SEED, n_jobs=1),
        hdbscan_model=HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, min_samples=10,
                              metric="euclidean", prediction_data=False),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
        top_n_words=12,
    )
    documents = (posts["title_clean"] + "\n" + posts["body_clean"]).tolist()
    topics, strength = model.fit_transform(documents, embeddings=embeddings)
    return model, np.asarray(topics), np.asarray(strength)


def review_tables(posts, embeddings, model):
    summary = []
    examples = []
    for topic, group in posts.groupby("topic_id"):
        terms = model.get_topic(topic)
        summary.append({
            "topic_id": int(topic), "discussions": len(group),
            "keywords": ", ".join(word for word, _ in terms if word),
        })
        # Rank ordinary clusters by distance to their embedding centroid. Show random outliers.
        if topic == -1:
            selected = group.sample(min(10, len(group)), random_state=paired.SEED)
        else:
            vectors = embeddings[group.index]
            centroid = normalize(vectors.mean(axis=0, keepdims=True))[0]
            central = group.iloc[np.argsort(-(vectors @ centroid), kind="stable")[:5]]
            diverse = group.drop(index=central.index).sample(min(5, len(group) - len(central)), random_state=paired.SEED)
            selected = pd.concat([central, diverse])
        for index, (_, post) in enumerate(selected.iterrows()):
            examples.append({
                "topic_id": int(topic), "example_type": "Representative" if topic != -1 and index < 5 else "Random",
                "post_id": post["post_id"], "title": post["title"],
                "body_excerpt": " ".join(post["body_clean"].split())[:300],
            })
    return pd.DataFrame(summary), pd.DataFrame(examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-folder", type=Path)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(4)
    posts, raw_sha = load_posts()
    title_sample = posts[["post_id", "title"]].sample(TITLE_SAMPLE_SIZE, random_state=paired.SEED).copy()
    title_sample["manual_topic"] = ""
    model = SentenceTransformer(
        MODEL, revision=REVISION, cache_folder=args.cache_folder, local_files_only=True, device="cpu",
    )
    embeddings, chunks = embed(posts, model)
    print("Fitting BERTopic on training OP embeddings.", flush=True)
    learner, posts["topic_id"], posts["membership_strength"] = clustering(posts, embeddings)
    summary, examples = review_tables(posts, embeddings, learner)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        posts[["post_id", "title", "topic_id", "membership_strength"]].to_parquet(
            staging / "op_topic_assignments.parquet", index=False,
        )
        np.save(staging / "op_embeddings.npy", embeddings)
        summary.to_csv(staging / "cluster_summary.csv", index=False)
        examples.to_csv(staging / "cluster_review.csv", index=False)
        title_sample.to_csv(staging / "random_title_review.csv", index=False)
        (staging / "results.json").write_text(json.dumps({
            "split": "train", "p4_run": P4.name, "source_archive_sha256": raw_sha,
            "discussions": len(posts), "seed": paired.SEED, "embedding_model": MODEL,
            "embedding_revision": REVISION, "body_weight": BODY_WEIGHT, "body_chunks": chunks,
            "min_topic_size": MIN_TOPIC_SIZE, "clusters": int(posts.loc[posts["topic_id"].ge(0), "topic_id"].nunique()),
            "outliers": int(posts["topic_id"].eq(-1).sum()), "title_sample": len(title_sample),
            "embedding_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
            "versions": {name: version(name) for name in ("bertopic", "sentence-transformers", "umap-learn", "hdbscan")},
        }, indent=2) + "\n")
        staging.rename(destination)
    print(summary.to_string(index=False))
    print(f"Saved topic review to {destination}")


if __name__ == "__main__":
    main()
