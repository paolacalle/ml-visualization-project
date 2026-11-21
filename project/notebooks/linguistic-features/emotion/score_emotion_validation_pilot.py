"""Score a seeded pilot of unique training openings for blinded emotion review."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data" / "processed" / "linguistic-features"
OPENINGS = DATA / "paired-linguistic-strategies" / "20261121T203952524644Z" / "opening_strategy_scores.parquet"
P6 = DATA / "paired-models" / "p6_adjusted_linguistic_strategies" / "20261121T204840655027Z" / "oof_predictions.parquet"
OUTPUT = DATA / "emotion-validation-pilot"
CACHE = PROJECT / "data" / "model-cache" / "goemotions"
MODEL = "SamLowe/roberta-base-go_emotions"
REVISION = "9045a3949b4d643baee6296673cea5bbfdc66c04"
LABELS = ("joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral")
THRESHOLD = 0.5
MAX_TOKENS = 512


def load_training_openings():
    pairs = pd.read_parquet(P6, columns=["a_comment_id", "b_comment_id"])
    ids = sorted(set(pairs["a_comment_id"]) | set(pairs["b_comment_id"]))
    openings = pd.read_parquet(OPENINGS, columns=["opening_text"])
    if not openings.index.is_unique or set(openings.index) != set(ids):
        raise ValueError("The opening cache must match the unique P6 training replies.")
    openings = openings.loc[ids]
    if not openings["opening_text"].fillna("").str.strip().ne("").all():
        raise ValueError("Every training opening must have usable text.")
    return openings


def score(candidates, cache_dir, batch_size, threads):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    settings = {"revision": REVISION, "cache_dir": cache_dir, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **settings)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, **settings).eval()
    all_labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    if len(all_labels) != 28 or not set(LABELS).issubset(all_labels):
        raise ValueError("The checkpoint must supply the expected GoEmotions labels.")

    # Include every overflow chunk and keep the strongest local cue for each label.
    encoded = tokenizer(candidates["opening_text"].tolist(), max_length=MAX_TOKENS,
                        truncation=True, return_overflowing_tokens=True)
    chunks = encoded["input_ids"]
    owners = np.asarray(encoded["overflow_to_sample_mapping"])
    order = sorted(range(len(chunks)), key=lambda index: len(chunks[index]))
    scores = np.zeros((len(candidates), len(all_labels)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            inputs = tokenizer.pad(
                {"input_ids": [chunks[index] for index in batch]},
                padding=True, return_attention_mask=True, return_tensors="pt",
            )
            probabilities = model(**inputs).logits.sigmoid().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("The emotion classifier returned a nonfinite score.")
            np.maximum.at(scores, owners[batch], probabilities)
            if start % (batch_size * 8) == 0:
                print(f"Scored {min(start + batch_size, len(order))} of {len(order)} chunks", flush=True)

    labels = pd.DataFrame(scores, index=candidates.index, columns=all_labels)
    scores = labels.add_prefix("score_").join(labels.ge(THRESHOLD).add_prefix("detected_"))
    scores["chunk_count"] = np.bincount(owners, minlength=len(candidates))
    if scores["chunk_count"].lt(1).any():
        raise ValueError("Every candidate opening must have at least one scored chunk.")
    scores["no_label_detected"] = ~labels.ge(THRESHOLD).any(axis=1)
    return scores


def review_sample(scores, size, rng):
    # Reserve a random portion, then enrich for rare labels without reading delta outcomes.
    random_ids = rng.choice(scores.index, size=size // 3, replace=False)
    selected = dict.fromkeys(random_ids, "random")
    per_label = (size - len(selected)) // len(LABELS)
    labels_by_rarity = sorted(LABELS, key=lambda label: scores[f"detected_{label}"].sum())
    for label in labels_by_rarity:
        remaining = scores.loc[~scores.index.isin(selected)]
        detected = remaining.loc[remaining[f"detected_{label}"]]
        hits = rng.choice(detected.index, size=min(per_label, len(detected)), replace=False)
        selected.update(dict.fromkeys(hits, f"detected_{label}"))
        shortage = per_label - len(hits)
        remaining = remaining.loc[~remaining.index.isin(hits)]
        near_misses = remaining.nlargest(shortage, f"score_{label}").index
        selected.update(dict.fromkeys(near_misses, f"below_threshold_{label}"))
    remaining = scores.index[~scores.index.isin(selected)]
    fill = rng.choice(remaining, size=size - len(selected), replace=False)
    selected.update(dict.fromkeys(fill, "random_fill"))
    ids = rng.permutation(list(selected))
    return pd.Series([selected[comment_id] for comment_id in ids], index=ids, name="selection_reason")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--pool-size", type=int, default=384)
    parser.add_argument("--review-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.pool_size, args.review_size, args.batch_size, args.threads) < 1 or args.review_size > args.pool_size:
        parser.error("Sizes and threads must be positive, with review size at most pool size.")

    openings = load_training_openings()
    if args.pool_size > len(openings):
        parser.error("Pool size exceeds the unique training openings.")
    rng = np.random.default_rng(args.seed)
    candidates = openings.loc[rng.choice(openings.index, size=args.pool_size, replace=False)]
    scores = score(candidates, args.cache_dir, args.batch_size, args.threads)
    selected = review_sample(scores, args.review_size, rng)
    scores["selection_reason"] = selected
    review = candidates.loc[selected.index].rename_axis("comment_id").reset_index()
    review.insert(0, "review_id", [f"E{i:03d}" for i in range(1, len(review) + 1)])
    review["manual_labels"] = ""
    review["review_notes"] = ""

    run = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    scores.to_parquet(run / "candidate_emotion_scores.parquet")
    review.to_csv(run / "blind_emotion_review.csv", index=False)
    manifest = {
        "split": "train", "seed": args.seed, "model": MODEL, "revision": REVISION,
        "threshold": THRESHOLD, "chunk_max_tokens": MAX_TOKENS, "chunk_pooling": "labelwise_max_sigmoid",
        "candidate_count": len(scores), "review_count": len(review), "review_labels": LABELS,
        "source_openings": str(OPENINGS.relative_to(PROJECT)), "source_pairs": str(P6.relative_to(PROJECT)),
        "source_openings_sha256": hashlib.sha256(OPENINGS.read_bytes()).hexdigest(),
    }
    (run / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(run)


if __name__ == "__main__":
    main()
