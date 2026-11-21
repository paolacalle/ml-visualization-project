"""Score every unique opening in the archived P6 training cohort."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from score_emotion_validation_pilot import (
    CACHE, DATA, MAX_TOKENS, MODEL, OPENINGS, PROJECT, REVISION, THRESHOLD,
    load_training_openings, score,
)


OUTPUT = DATA / "training-opening-emotions"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("Batch size and threads must be positive.")

    openings = load_training_openings()
    scores = score(openings, args.cache_dir, args.batch_size, args.threads)
    manifest = {
        "split": "train", "model": MODEL, "revision": REVISION,
        "openings": len(scores), "chunks": int(scores["chunk_count"].sum()),
        "threshold": THRESHOLD, "chunk_max_tokens": MAX_TOKENS,
        "chunk_pooling": "labelwise_max_sigmoid",
        "source_openings": str(OPENINGS.relative_to(PROJECT)),
        "versions": {name: version(name) for name in ("torch", "transformers", "tokenizers")},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUTPUT) as temporary:
        stage = Path(temporary)
        scores.to_parquet(stage / "opening_emotion_scores.parquet")
        (stage / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(run)
    print(f"Scored {len(scores):,} training openings in {manifest['chunks']:,} chunks")
    print(run)


if __name__ == "__main__":
    main()
