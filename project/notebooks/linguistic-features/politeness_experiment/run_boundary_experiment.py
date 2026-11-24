"""Phase 3A. Change only fixed-limit boundaries, retaining every content token."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from math import comb
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import transformers

from politeness_extractor import (
    BatchedTransformerClassifier, POLITE_GUARD, POLITE_GUARD_REVISION,
    Segmentation, politeness_features,
)
from run_chunk_size_experiment import RESULTS as PHASE2, sha256, training_openings
from run_pilot import DATA, SEED, WEIGHTS_SHA256

RESULTS = DATA / "politeness-experiment/phase-3"
SIZES = (256, 510)
TRIALS = 5


def experiment_sample(scores):
    lengths = scores.loc[scores["content_limit"].eq(510)].set_index("argument_id")["n_tokens"]
    long = lengths.loc[lengths.gt(510)].sort_index()
    medium = lengths.loc[lengths.between(257, 510)].sort_index().sort_values(kind="stable")
    if (len(medium) + len(long), len(long)) != (2711, 854):
        raise ValueError("The Phase 2 cohort does not match the Phase 3 sampling plan.")
    rng = np.random.default_rng(SEED)
    medium_groups = np.array_split(medium.index.to_numpy(), 4)
    draws = [rng.choice(group, n, replace=False) for group, n in zip(medium_groups, (37, 37, 36, 36), strict=True)]
    supplement = np.concatenate(draws)
    return {256: sorted([*long.index, *supplement]), 510: sorted(long.index)}


def randomized_lengths(size, count, slack, argument_id):
    """Sample uniformly from distinct layouts excluding the fixed layout."""
    possible = comb(slack + count - 1, count - 1) - 1
    key = f"{SEED}:{size}:{argument_id}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    fixed = (size,) * (count - 1) + (size - slack,)
    seen = {fixed}
    layouts = []
    while len(layouts) < min(TRIALS, possible):
        # Stars and bars distributes unused capacity among the original number of chunks.
        bars = [-1, *sorted(rng.sample(range(slack + count - 1), count - 1)), slack + count - 1]
        lengths = tuple(size - (right - left - 1) for left, right in zip(bars, bars[1:]))
        if lengths not in seen:
            seen.add(lengths)
            layouts.append(lengths)
    return layouts, possible


def build_layouts(sample, tokenized, size):
    arguments, design, spans = [], [], {}
    for argument_id in sample:
        n_tokens = len(tokenized[argument_id])
        fixed = Segmentation(size).spans(n_tokens, argument_id)
        slack = len(fixed) * size - n_tokens
        layouts, possible = randomized_lengths(size, len(fixed), slack, argument_id)
        arguments.append({
            "argument_id": argument_id, "content_limit": size, "n_tokens": n_tokens, "n_chunks": len(fixed),
            "max_boundary_displacement": slack, "possible_changed_layouts": str(possible),
            "distinct_changed_layouts": len(layouts), "at_most_one_token_of_flexibility": slack <= 1,
        })
        fixed_boundaries = np.array([end for _, end in fixed[:-1]])
        for trial, lengths in enumerate(layouts, 1):
            boundaries = np.cumsum(lengths)
            displacement = fixed_boundaries - boundaries[:-1]
            if boundaries[-1] != n_tokens or not 0 < min(lengths) <= max(lengths) <= size or not displacement.any():
                raise AssertionError("An invalid or unchanged layout was generated.")
            layout_id = f"{argument_id}:{trial}"
            spans[layout_id] = list(zip([0, *boundaries[:-1].tolist()], boundaries.tolist(), strict=True))
            design.append({
                "layout_id": layout_id, "opening_id": argument_id, "trial": trial, "content_limit": size,
                "chunk_lengths": list(lengths), "boundary_displacements": displacement.tolist(),
                "mean_boundary_displacement": float(displacement.mean()),
                "max_observed_displacement": int(displacement.max()),
                "fraction_of_available_displacement": float(displacement.mean() / slack),
            })
    return pd.DataFrame(arguments), pd.DataFrame(design), spans


def prime_cache(classifier, run, tokenized):
    fields = [f"politeness_prob_{label.replace(' ', '_')}" for label in classifier.labels]
    chunks = pd.read_parquet(run / "chunk_predictions.parquet",
                             columns=["argument_id", "token_start", "token_end", *fields])
    chunks = chunks.loc[chunks["argument_id"].isin(tokenized)]
    for argument_id, start, end, *probabilities in chunks.itertuples(index=False, name=None):
        ids = tokenized[argument_id][start:end]
        if len(ids) != end - start or not ids:
            raise ValueError("Phase 2 chunk positions do not match the retokenized opening.")
        probability = np.asarray(probabilities, dtype=np.float32)
        cached = classifier.cache.setdefault(ids, probability)
        if not np.array_equal(cached, probability):
            raise ValueError("Phase 2 archive contains inconsistent scores for identical chunks.")
    return len(classifier.cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DATA / "polite-guard-checkpoint")
    parser.add_argument("--content-limits", type=int, nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    sizes = tuple(size for size in SIZES if size in args.content_limits)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    start = perf_counter()
    phase2 = args.phase2_run or sorted(PHASE2.glob("*/experiment_metadata.json"))[-1].parent
    reference = json.loads((phase2 / "experiment_metadata.json").read_text())
    if (reference["phase"], reference["split"], reference["revision"], reference["weights_sha256"]) != (
        2, "train", POLITE_GUARD_REVISION, WEIGHTS_SHA256,
    ):
        raise ValueError("Expected the matching frozen Phase 2 run.")
    if (reference["device"], reference["torch_version"], reference["transformers_version"]) != (
        "cpu", torch.__version__, transformers.__version__,
    ):
        raise ValueError("Use the Phase 2 CPU environment to reuse the cached scores.")
    source = DATA / "paired-linguistic-strategies" / reference["source_run"]
    if sha256(source / "opening_strategy_scores.parquet") != reference["input_sha256"]:
        raise ValueError("The archived openings differ from Phase 2.")
    if sha256(args.checkpoint / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("The checkpoint does not match the frozen Polite Guard weights.")
    baseline = pd.read_parquet(phase2 / "chunk_size_scores.parquet")
    sample = experiment_sample(baseline)
    openings = training_openings(source)
    scorer = BatchedTransformerClassifier(
        POLITE_GUARD, POLITE_GUARD_REVISION, batch_size=args.batch_size, checkpoint=args.checkpoint,
    )
    if scorer.content_limit != 510 or set(scorer.labels) != set(reference["ordinal_weights"]):
        raise ValueError("The model labels or content-token budget changed.")
    selected_ids = sorted({key for size in sizes for key in sample[size]})
    tokenized = scorer.tokenize(openings.loc[selected_ids, "opening_text"].to_dict())
    expected = baseline.loc[baseline["content_limit"].eq(256)].set_index("argument_id")["n_tokens"]
    if any(expected.loc[key] != len(ids) for key, ids in tokenized.items()):
        raise ValueError("Phase 2 and Phase 3 token counts differ.")
    cached = prime_cache(scorer, phase2, tokenized)
    logging.info("Loaded %s cached Phase 2 chunk predictions", f"{cached:,}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = RESULTS / run_id
    timings = []
    with TemporaryDirectory(dir=RESULTS) as temporary:
        stage = Path(temporary)
        for size in sizes:
            begin = perf_counter()
            cohort, design, spans = build_layouts(sample[size], tokenized, size)
            logging.info("Scoring %s layouts across %s openings at %s content tokens",
                         f"{len(design):,}", f"{len(cohort):,}", size)
            indices = dict(zip(design["layout_id"], design["opening_id"], strict=True))
            arguments, chunks = scorer.score_layouts(
                {layout_id: tokenized[key] for layout_id, key in indices.items()},
                spans, f"{size}_randomized", probability_prefix="politeness_prob",
            )
            arguments = politeness_features(arguments, chunks).rename(columns={"argument_id": "layout_id"})
            trials = design.merge(arguments, on="layout_id", validate="one_to_one")
            if len(trials) != cohort["distinct_changed_layouts"].sum():
                raise AssertionError("Expected exactly one score per distinct valid layout.")
            np.testing.assert_allclose(trials.filter(like="politeness_prob_").sum(axis=1), 1, atol=1e-6)
            trials.to_parquet(stage / f"randomized_scores_{size}.parquet", index=False)
            cohort.to_parquet(stage / f"boundary_flexibility_{size}.parquet", index=False)
            timings.append({
                "content_limit": size, "openings": len(cohort), "layouts": len(trials), "chunks": len(chunks),
                "cache_hits": int(chunks["cache_hit"].sum()), "seconds": round(perf_counter() - begin, 3),
            })
            logging.info("%s", json.dumps(timings[-1]))
        metadata = {
            "phase": 3, "split": "train", "model": POLITE_GUARD, "revision": POLITE_GUARD_REVISION,
            "weights_sha256": WEIGHTS_SHA256, "source_run": source.name, "phase2_run": phase2.name,
            "phase2_scores_sha256": sha256(phase2 / "chunk_size_scores.parquet"), "seed": SEED,
            "max_trials": TRIALS, "initial_cached_chunks": cached,
            "torch_threads": args.threads, "batch_size": args.batch_size,
            "configurations": timings, "wall_seconds": round(perf_counter() - start, 3),
        }
        (stage / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        stage.rename(destination)
    print(f"Saved Phase 3A scores to {destination}", flush=True)


if __name__ == "__main__":
    main()
