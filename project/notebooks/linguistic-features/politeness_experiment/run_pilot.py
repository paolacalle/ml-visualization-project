"""Run Phase 1 on 100 distinct archived training openings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import transformers

from politeness_extractor import (
    BatchedTransformerClassifier, POLITE_GUARD, POLITE_GUARD_REVISION,
    POLITENESS_WEIGHTS, Segmentation, politeness_features,
)

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data/processed/linguistic-features"
SOURCE = DATA / "paired-linguistic-strategies/20261121T203952524644Z"
RESULTS = DATA / "politeness-experiment/phase-1"
SEED = 42
WEIGHTS_SHA256 = "121f459840c7a8d215f063916939a8e3df18dba3fc576ea10b10ac7974ea6192"


def pilot_sample():
    manifest = json.loads((SOURCE / "results.json").read_text())
    if manifest["split"] != "train":
        raise ValueError("Expected the archived matched training openings.")
    file = SOURCE / "opening_strategy_scores.parquet"
    openings = pd.read_parquet(file, columns=["opening_text", "analyzed_word_count"]).sort_index()
    if not openings.index.is_unique or len(openings) != manifest["unique_openings"]:
        raise ValueError("Expected one row per archived opening.")
    # Equal representation across archived opening-length quartiles.
    quartiles = np.array_split(np.argsort(openings["analyzed_word_count"].to_numpy(), kind="stable"), 4)
    rng = np.random.default_rng(SEED)
    selected = np.concatenate([rng.choice(group, 25, replace=False) for group in quartiles])
    with file.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    return openings.iloc[selected], fingerprint


def validate(arguments, chunks, tokenized, config):
    probabilities = list(arguments.filter(like="politeness_prob_"))
    np.testing.assert_allclose(arguments[probabilities].sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(chunks[probabilities].sum(axis=1), 1, atol=1e-6)
    if len(arguments) != 100 or not arguments["argument_id"].is_unique:
        raise AssertionError("Expected 100 unique pilot predictions per configuration.")
    if not arguments["politeness_score"].between(0, 3).all():
        raise AssertionError("Ordinal scores must be in [0, 3].")
    for key, group in chunks.groupby("argument_id", sort=False):
        ranges = list(zip(group["token_start"], group["token_end"], strict=True))
        if (ranges[0][0] != 0 or ranges[-1][1] != len(tokenized[key])
                or any(end != following for (_, end), (following, _) in zip(ranges, ranges[1:]))
                or not group["n_tokens"].between(1, config.content_limit).all()):
            raise AssertionError(f"Invalid token coverage for {key}.")
        pooled = np.average(group[probabilities], weights=group["n_tokens"], axis=0)
        prediction = arguments.loc[arguments["argument_id"].eq(key), probabilities].iloc[0]
        np.testing.assert_allclose(prediction, pooled, atol=1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DATA / "polite-guard-checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = perf_counter()
    sample, fingerprint = pilot_sample()
    with (args.checkpoint / "model.safetensors").open("rb") as weights:
        weights_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
    if weights_sha256 != WEIGHTS_SHA256:
        raise ValueError("The local weights do not match the pinned Polite Guard revision.")
    classifier = BatchedTransformerClassifier(
        POLITE_GUARD, POLITE_GUARD_REVISION, checkpoint=args.checkpoint,
        batch_size=args.batch_size, device=args.device,
    )
    if set(classifier.labels) != POLITENESS_WEIGHTS.keys() or classifier.content_limit != 510:
        raise ValueError("Unexpected Polite Guard label mapping or token budget.")
    tokenized = classifier.tokenize(sample["opening_text"].to_dict())
    arg_frames, chunk_frames, timings = [], [], []
    for config in [Segmentation(128), Segmentation(256), Segmentation(510), Segmentation(128, SEED)]:
        begin = perf_counter()
        arguments, chunks = classifier.score(tokenized, config, probability_prefix="politeness_prob")
        timing = {
            "segmentation": config.name, "chunks": len(chunks),
            "new_predictions": int((~chunks["cache_hit"]).sum()), "cache_hits": int(chunks["cache_hit"].sum()),
            "seconds": round(perf_counter() - begin, 3),
        }
        print(json.dumps(timing), flush=True)
        timings.append(timing)
        arguments = politeness_features(arguments, chunks)
        validate(arguments, chunks, tokenized, config)
        arg_frames.append(arguments)
        chunk_frames.append(chunks)

    repeat, repeat_chunks = classifier.score(tokenized, Segmentation(128, SEED), probability_prefix="politeness_prob")
    comparison_columns = repeat.columns.drop("inference_time")
    pd.testing.assert_frame_equal(repeat[comparison_columns], arg_frames[-1][comparison_columns])
    pd.testing.assert_frame_equal(
        repeat_chunks.drop(columns=["inference_time", "cache_hit"]),
        chunk_frames[-1].drop(columns=["inference_time", "cache_hit"]),
    )
    if not repeat_chunks["cache_hit"].all() or repeat["inference_time"].any():
        raise AssertionError("Expected the repeat to use only cached predictions.")
    fixed_spans = chunk_frames[0].groupby("argument_id")["token_end"].agg(tuple)
    random_spans = chunk_frames[-1].groupby("argument_id")["token_end"].agg(tuple)
    changed = int(fixed_spans.ne(random_spans).sum())
    if changed == 0:
        raise AssertionError("The seeded pilot did not exercise randomized boundaries.")

    results = {
        "phase": 1, "split": "train", "model": POLITE_GUARD, "revision": POLITE_GUARD_REVISION,
        "weights_sha256": weights_sha256,
        "labels_by_id": classifier.labels, "ordinal_weights": POLITENESS_WEIGHTS,
        "max_content_tokens": classifier.content_limit, "seed": SEED, "arguments": len(sample),
        "source_run": SOURCE.name, "input_sha256": fingerprint, "device": args.device,
        "batch_size": args.batch_size, "torch_threads": args.threads, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "configurations": timings,
        "changed_boundaries": changed, "repeat_cache_hits": int(repeat_chunks["cache_hit"].sum()),
        "wall_seconds_before_export": round(perf_counter() - started, 3),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=RESULTS) as temporary:
        stage = Path(temporary)
        pd.concat(arg_frames, ignore_index=True).to_parquet(stage / "argument_predictions.parquet", index=False)
        pd.concat(chunk_frames, ignore_index=True).to_parquet(stage / "chunk_predictions.parquet", index=False)
        (stage / "experiment_metadata.json").write_text(json.dumps(results, indent=2) + "\n")
        stage.rename(destination)
    print(f"Saved pilot to {destination}", flush=True)


if __name__ == "__main__":
    main()
