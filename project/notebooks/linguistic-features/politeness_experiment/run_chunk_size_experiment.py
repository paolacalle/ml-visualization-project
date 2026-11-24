"""Score every archived training opening at three Polite Guard chunk sizes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
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
from run_pilot import DATA, SOURCE, WEIGHTS_SHA256

RESULTS = DATA / "politeness-experiment/phase-2"
SIZES = (128, 256, 510)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def training_openings(source):
    manifest = json.loads((source / "results.json").read_text())
    if manifest["split"] != "train":
        raise ValueError("Expected archived matched training openings.")
    openings = pd.read_parquet(source / "opening_strategy_scores.parquet", columns=["opening_text"]).sort_index()
    if not openings.index.is_unique or len(openings) != manifest["unique_openings"]:
        raise ValueError("Expected one row per archived opening.")
    return openings


def validate_scores(arguments, chunks, tokenized, size):
    expected = pd.Series({key: len(ids) for key, ids in tokenized.items()}, name="n_tokens")
    expected.index.name = "argument_id"
    scored = arguments.set_index("argument_id")
    pd.testing.assert_series_equal(scored["n_tokens"], expected)
    np.testing.assert_array_equal(scored["n_chunks"], (expected + size - 1) // size)
    spans = chunks.groupby("argument_id", sort=False)
    np.testing.assert_array_equal(spans["n_tokens"].sum(), expected)
    np.testing.assert_array_equal(spans["token_end"].last(), expected)
    np.testing.assert_array_equal(spans["token_end"].shift(fill_value=0), chunks["token_start"])
    if not chunks["n_tokens"].between(1, size).all():
        raise AssertionError(f"Invalid chunk token count for size {size}.")
    for table in (arguments, chunks):
        np.testing.assert_allclose(table.filter(like="politeness_prob_").sum(axis=1), 1, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DATA / "polite-guard-checkpoint")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    started = perf_counter()
    openings = training_openings(args.source)
    fingerprint = sha256(args.source / "opening_strategy_scores.parquet")
    if sha256(args.checkpoint / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("The local weights do not match the pinned Polite Guard revision.")
    classifier = BatchedTransformerClassifier(
        POLITE_GUARD, POLITE_GUARD_REVISION, checkpoint=args.checkpoint,
        batch_size=args.batch_size, device=args.device,
    )
    if set(classifier.labels) != POLITENESS_WEIGHTS.keys() or classifier.content_limit != max(SIZES):
        raise ValueError("Unexpected Polite Guard label mapping or token budget.")
    tokenized = classifier.tokenize(openings["opening_text"].to_dict())
    argument_frames, chunk_frames, timings = [], [], []
    for size in SIZES:
        config = Segmentation(size)
        begin = perf_counter()
        logging.info("Scoring %s openings at %s content tokens", f"{len(openings):,}", size)
        arguments, chunks = classifier.score(tokenized, config, probability_prefix="politeness_prob")
        timings.append({
            "content_limit": size, "chunks": len(chunks),
            "cache_hits": int(chunks["cache_hit"].sum()),
            "new_predictions": int((~chunks["cache_hit"]).sum()),
            "seconds": round(perf_counter() - begin, 3),
        })
        logging.info("%s", json.dumps(timings[-1]))
        arguments = politeness_features(arguments, chunks)
        validate_scores(arguments, chunks, tokenized, size)
        argument_frames.append(arguments.assign(content_limit=size))
        chunk_frames.append(chunks.assign(content_limit=size))

    metadata = {
        "phase": 2, "split": "train", "model": POLITE_GUARD, "revision": POLITE_GUARD_REVISION,
        "weights_sha256": WEIGHTS_SHA256, "ordinal_weights": POLITENESS_WEIGHTS,
        "source_run": args.source.name, "input_sha256": fingerprint,
        "arguments": len(openings), "device": args.device, "batch_size": args.batch_size,
        "torch_threads": args.threads, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "configurations": timings,
        "wall_seconds_before_export": round(perf_counter() - started, 3),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=RESULTS) as temporary:
        stage = Path(temporary)
        pd.concat(argument_frames, ignore_index=True).to_parquet(stage / "chunk_size_scores.parquet", index=False)
        pd.concat(chunk_frames, ignore_index=True).to_parquet(stage / "chunk_predictions.parquet", index=False)
        (stage / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(destination)
    print(f"Saved Phase 2 scores to {destination}", flush=True)


if __name__ == "__main__":
    main()
