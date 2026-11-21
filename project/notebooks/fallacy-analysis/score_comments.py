"""Score CMV comments with the fine-tuned fallacy model, window by window.

Training texts are short (median ~50 words), so each comment is split into
overlapping windows of up to 3 sentences (stride 2) after removing quoted
('>') lines, since a quote's fallacies belong to the person being quoted. Each window
gets calibrated class probabilities; the comment gets aggregates.

Sample (default): every eligible comment with an OP-awarded delta, plus up to
--controls-per-delta random eligible non-delta comments from the same post.
Eligible = not by the OP or a bot, body not deleted, >= --min-words words.
Use --all-eligible to score everything (~824k comments, many hours).

Outputs in project/data/processed/cmv-fallacy/<run>/ (resumable; finished
chunks are skipped on rerun):
    sample.parquet            the sampled comments and their delta labels
    windows/chunk-*.parquet   per-window text + probabilities
    comments/chunk-*.parquet  per-comment aggregates

Usage (repo root):
    .venv/bin/python project/notebooks/fallacy-analysis/score_comments.py --limit 200   # smoke test
    .venv/bin/python project/notebooks/fallacy-analysis/score_comments.py
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "project" / "notebooks" / "fallacy-model"))
from build_dataset import clean_text  # noqa: E402  same normalisation as training
from train import ensure_special_tokens, pick_device  # noqa: E402

DELTAS_PATH = REPO_ROOT / "project" / "data" / "processed" / "cmv-deltas" / "comments.parquet"
TEXT_DIR = REPO_ROOT / "project" / "data" / "processed" / "cmv-text" / "v1" / "comments"
MODEL_DIR = REPO_ROOT / "project" / "data" / "models" / "fallacy" / "deberta-base" / "best"
OUT_ROOT = REPO_ROOT / "project" / "data" / "processed" / "cmv-fallacy"

SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DELTA_MARK_RE = re.compile(r"∆|Δ|!delta", re.I)


# --------------------------------------------------------------------------- #
# Sampling and text
# --------------------------------------------------------------------------- #
def build_sample(args: argparse.Namespace) -> pd.DataFrame:
    meta = pd.read_parquet(args.deltas)
    elig = meta[~meta["is_op"] & ~meta["is_bot"] & ~meta["is_deleted"] & (meta["n_words"] >= args.min_words)]
    if args.all_eligible:
        sample = elig
    else:
        pos = elig[elig["got_delta_op"]]
        # Controls: same post, no delta from anyone.
        pool = elig[~elig["got_delta_any"] & elig["post_id"].isin(pos["post_id"])]
        quota = pool["post_id"].map(pos.groupby("post_id").size() * args.controls_per_delta)
        # Shuffle, then keep the first `quota` controls within each post.
        shuffled = pool.sample(frac=1, random_state=args.seed)
        rank = shuffled.groupby("post_id").cumcount()
        controls = shuffled[rank < quota.loc[shuffled.index]]
        sample = pd.concat([pos, controls])
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        sample = sample.head(args.limit)
    return sample


def load_text(comment_ids: list[str]) -> dict[str, str]:
    dataset = ds.dataset(TEXT_DIR, format="parquet", partitioning="hive")
    table = dataset.to_table(columns=["comment_id", "body"],
                             filter=pc.field("comment_id").isin(comment_ids))
    return dict(zip(table["comment_id"].to_pylist(), table["body"].to_pylist()))


def to_windows(body: str, max_sents: int = 3, stride: int = 2) -> list[str]:
    text = html.unescape(html.unescape(body or ""))
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(">")]
    # Delta symbols are awarding chatter, not argument content.
    text = DELTA_MARK_RE.sub("", "\n".join(lines))
    sents = [s for s in (clean_text(s) for s in SENT_RE.split(text)) if len(s.split()) >= 2]
    if not sents:
        return []
    if len(sents) <= max_sents:
        return [" ".join(sents)]
    windows = []
    for i in range(0, len(sents), stride):
        windows.append(" ".join(sents[i:i + max_sents]))
        if i + max_sents >= len(sents):
            break
    return windows


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_windows(texts: list[str], tokenizer, model, device, temperature: float,
                  batch_size: int, max_length: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])  # length-sorted batches waste less padding
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length,
                        padding=True, pad_to_multiple_of=32, return_tensors="pt").to(device)
        logits = model(**enc).logits.float() / temperature
        probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def aggregate(win: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    fallacies = [l for l in labels if l != "none"]
    cols = {f"p_{l.replace(' ', '_')}": l for l in labels}
    win = win.assign(p_fallacy=1.0 - win["p_none"],
                     top=win[list(cols)].to_numpy().argmax(axis=1))
    g = win.groupby("comment_id")
    out = pd.DataFrame({
        "n_windows": g.size(),
        "max_p_fallacy": g["p_fallacy"].max(),
        "mean_p_fallacy": g["p_fallacy"].mean(),
        # Share of windows whose top class is a fallacy (length-robust density).
        "frac_windows_fallacy": g["top"].apply(lambda s: (s != labels.index("none")).mean()),
    })
    for l in fallacies:
        key = l.replace(" ", "_")
        out[f"max_p_{key}"] = g[f"p_{key}"].max()
        out[f"frac_windows_{key}"] = g["top"].apply(lambda s, i=labels.index(l): (s == i).mean())
    best = win.loc[g["p_fallacy"].idxmax(), ["comment_id", "text"]].set_index("comment_id")
    out["top_window"] = best["text"]
    return out.reset_index()


def main() -> None:
    args = parse_args()
    out_dir = args.out_root / args.run_name
    (out_dir / "windows").mkdir(parents=True, exist_ok=True)
    (out_dir / "comments").mkdir(parents=True, exist_ok=True)

    sample_path = out_dir / "sample.parquet"
    if sample_path.exists():
        sample = pd.read_parquet(sample_path)
        print(f"reusing {sample_path} ({len(sample):,} comments)")
    else:
        sample = build_sample(args)
        sample.to_parquet(sample_path, index=False)
    print(f"sample: {len(sample):,} comments, {sample['got_delta_op'].sum():,} with an OP delta, "
          f"{sample['post_id'].nunique():,} posts")

    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ensure_special_tokens(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, dtype=torch.float32).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    calib = args.model_dir / "calibration.json"
    temperature = json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0
    (out_dir / "run.json").write_text(json.dumps(
        {"model_dir": str(args.model_dir), "temperature": temperature, "labels": labels,
         **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}, indent=2))

    n_chunks = (len(sample) + args.chunk_size - 1) // args.chunk_size
    t0, done_windows = time.time(), 0
    for c in range(n_chunks):
        chunk_file = f"chunk-{c:04d}.parquet"
        if (out_dir / "comments" / chunk_file).exists():
            continue
        part = sample.iloc[c * args.chunk_size:(c + 1) * args.chunk_size]
        bodies = load_text(part["comment_id"].tolist())
        rows = [(cid, w_i, w) for cid in part["comment_id"]
                for w_i, w in enumerate(to_windows(bodies.get(cid, "")))]
        win = pd.DataFrame(rows, columns=["comment_id", "window", "text"])
        probs = score_windows(win["text"].tolist(), tokenizer, model, device, temperature,
                              args.batch_size, args.max_length)
        for i, l in enumerate(labels):
            win[f"p_{l.replace(' ', '_')}"] = probs[:, i]
        win.to_parquet(out_dir / "windows" / chunk_file, index=False)
        aggregate(win, labels).to_parquet(out_dir / "comments" / chunk_file, index=False)
        if device.type == "mps":
            torch.mps.empty_cache()

        done_windows += len(win)
        elapsed = time.time() - t0
        rate = done_windows / elapsed
        remaining = (n_chunks - c - 1) * args.chunk_size * (len(win) / max(len(part), 1)) / rate
        print(f"chunk {c + 1}/{n_chunks}: {len(part)} comments, {len(win)} windows, "
              f"{rate:.0f} windows/s, eta {remaining / 60:.0f} min", flush=True)

    print(f"done -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="matched-5")
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--deltas", type=Path, default=DELTAS_PATH)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--controls-per-delta", type=int, default=5)
    p.add_argument("--all-eligible", action="store_true")
    p.add_argument("--min-words", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Score only the first N sampled comments")
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


if __name__ == "__main__":
    main()
