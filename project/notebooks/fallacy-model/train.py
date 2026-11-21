"""Fine-tune an encoder (default DeBERTa-v3-base) to classify fallacies.

Reads the parquet splits written by build_dataset.py, trains a 9-way
sequence classifier (8 fallacies + "none"), keeps the checkpoint with the best
dev macro-F1, then scores the test split and fits a temperature on dev so the
output probabilities are calibrated enough to use as downstream features.

Usage (from the repo root, with the project venv):
    # quick smoke test, a few minutes
    .venv/bin/python project/notebooks/fallacy-model/train.py --run-name smoke \
        --max-train-samples 400 --max-eval-samples 200 --epochs 1

    # full run (base model, ~32 min on an M4)
    .venv/bin/python project/notebooks/fallacy-model/train.py --run-name deberta-base

    # overnight run
    .venv/bin/python project/notebooks/fallacy-model/train.py --run-name deberta-large \
        --model microsoft/deberta-v3-large --batch-size 8 --grad-accum 2 --lr 1e-5

Outputs go to project/data/models/fallacy/<run-name>/ (git-ignored):
    best/                    model + tokenizer (load with AutoModelForSequenceClassification)
    metrics.json             dev/test metrics, temperature, training args
    test_report.txt          per-class precision/recall/F1
    test_confusion.csv       confusion matrix
    {dev,test}_predictions.parquet   per-row probabilities, for error analysis
    train_log.jsonl          loss/throughput log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "project" / "data" / "processed" / "fallacy"
RUNS_DIR = REPO_ROOT / "project" / "data" / "models" / "fallacy"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, with_context: bool):
        self.texts = df["text"].tolist()
        self.contexts = df["context"].fillna("").tolist() if with_context else None
        self.labels = df["label_id"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int):
        ctx = self.contexts[i] if self.contexts is not None else None
        return self.texts[i], ctx, self.labels[i]


def make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts, contexts, labels = zip(*batch)
        if contexts[0] is not None:
            # Title first, comment second; truncate the comment only if needed.
            enc = tokenizer(list(contexts), list(texts), truncation="longest_first",
                            max_length=max_length, padding=True, pad_to_multiple_of=32, return_tensors="pt")
        else:
            enc = tokenizer(list(texts), truncation=True, max_length=max_length,
                            padding=True, pad_to_multiple_of=32, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


def ensure_special_tokens(tokenizer) -> None:
    """Make sure [CLS] ... [SEP] get added.

    transformers 5.3 converts DeBERTa-v3's sentencepiece model without a
    post-processor, so encodings silently lack [CLS]/[SEP]. The classifier
    pools the first token, so without [CLS] it learns from an arbitrary word
    (macro-F1 ~0.3 instead of ~0.7). Patch the fast tokenizer in place; the fix
    is saved with the tokenizer, so inference code that loads best/ gets it too.
    """
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        return
    ids = tokenizer("hello world")["input_ids"]
    if ids[0] == cls_id and ids[-1] == sep_id:
        return
    from tokenizers.processors import TemplateProcessing

    cls, sep = tokenizer.cls_token, tokenizer.sep_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{cls} $A {sep}",
        pair=f"{cls} $A {sep} $B:1 {sep}:1",
        special_tokens=[(cls, cls_id), (sep, sep_id)],
    )
    ids = tokenizer("hello world")["input_ids"]
    assert ids[0] == cls_id and ids[-1] == sep_id, "could not add [CLS]/[SEP]"
    print(f"patched tokenizer to add {cls}/{sep}")


def load_split(name: str, args: argparse.Namespace, limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(args.data_dir / f"{name}.parquet")
    if args.drop_sources:
        df = df[~df["source"].isin(args.drop_sources)]
    if limit is not None and limit < len(df):
        # Stratified subsample so smoke tests still see every class.
        df = df.groupby("label_id").sample(frac=limit / len(df), random_state=args.seed)
    return df.reset_index(drop=True)


def class_weights(labels: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    if mode == "sqrt":
        w = np.sqrt(w)
    return torch.tensor(w / w.mean(), dtype=torch.float)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, probs: np.ndarray, none_id: int) -> dict:
    y_pred = probs.argmax(axis=1)
    fallacy_ids = [i for i in range(probs.shape[1]) if i != none_id]
    bin_true, bin_pred = y_true != none_id, y_pred != none_id
    p, r, f, _ = precision_recall_fscore_support(bin_true, bin_pred, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fallacy_macro_f1": f1_score(y_true, y_pred, labels=fallacy_ids, average="macro", zero_division=0),
        "binary_precision": p,
        "binary_recall": r,
        "binary_f1": f,
    }


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        all_logits.append(model(**batch).logits.float().cpu())
        all_labels.append(labels)
    model.train()
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single-parameter temperature scaling (Guo et al., 2017) on dev logits."""
    t_logits = torch.tensor(logits, dtype=torch.float)
    t_labels = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(t_logits / log_t.exp(), t_labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(ece)


def save_predictions(df: pd.DataFrame, probs: np.ndarray, labels: list[str], path: Path) -> None:
    out = df[["text", "label", "source", "group"]].copy()
    out["pred"] = [labels[i] for i in probs.argmax(axis=1)]
    out["p_fallacy"] = 1.0 - probs[:, labels.index("none")]
    for i, name in enumerate(labels):
        out[f"p_{name.replace(' ', '_')}"] = probs[:, i]
    out.to_parquet(path, index=False)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = json.loads((args.data_dir / "labels.json").read_text())
    none_id = labels.index("none")

    train_df = load_split("train", args, args.max_train_samples)
    dev_df = load_split("dev", args, args.max_eval_samples)
    test_df = load_split("test", args, args.max_eval_samples)
    print(f"device={device}  train={len(train_df)}  dev={len(dev_df)}  test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ensure_special_tokens(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)},
        # transformers v5 keeps the checkpoint's stored dtype (fp16 for DeBERTa-v3),
        # which breaks MPS kernels and AdamW; train in fp32.
        dtype=torch.float32,
    ).to(device)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(TextDataset(train_df, args.with_context), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    eval_bs = args.batch_size * 2
    dev_loader = DataLoader(TextDataset(dev_df, args.with_context), batch_size=eval_bs, collate_fn=collate)
    test_loader = DataLoader(TextDataset(test_df, args.with_context), batch_size=eval_bs, collate_fn=collate)

    weights = class_weights(train_df["label_id"].to_numpy(), len(labels), args.class_weighting)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    if weights is not None:
        print("class weights:", {l: round(float(w), 2) for l, w in zip(labels, weights)})

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    log_file = (run_dir / "train_log.jsonl").open("w")
    best_score, best_epoch, epochs_without_gain = -1.0, -1, 0
    best_dir = run_dir / "best"
    step, t0 = 0, time.time()
    model.train()

    for epoch in range(1, args.epochs + 1):
        running_loss, seen = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader, start=1):
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(model(**batch).logits, y) / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum
            seen += 1

            if i % args.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if device.type == "mps" and step % 50 == 0:
                    torch.mps.empty_cache()  # MPS otherwise holds on to freed buffers
                if step % args.log_every == 0 or step == total_steps:
                    elapsed = time.time() - t0
                    eta = elapsed / step * (total_steps - step)
                    rec = {"epoch": epoch, "step": step, "loss": running_loss / seen,
                           "lr": scheduler.get_last_lr()[0], "elapsed_min": elapsed / 60}
                    log_file.write(json.dumps(rec) + "\n")
                    log_file.flush()
                    print(f"epoch {epoch} step {step}/{total_steps}  loss {running_loss / seen:.4f}  "
                          f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)
                    running_loss, seen = 0.0, 0

        dev_logits, dev_labels = predict(model, dev_loader, device)
        dev_metrics = compute_metrics(dev_labels, softmax(dev_logits), none_id)
        log_file.write(json.dumps({"epoch": epoch, "dev": dev_metrics}) + "\n")
        log_file.flush()
        print(f"[epoch {epoch}] dev " + "  ".join(f"{k}={v:.3f}" for k, v in dev_metrics.items()), flush=True)

        if dev_metrics[args.select_metric] > best_score:
            best_score, best_epoch, epochs_without_gain = dev_metrics[args.select_metric], epoch, 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  new best {args.select_metric}={best_score:.4f}, saved to {best_dir}")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"  no gain for {args.patience} epoch(s), stopping early")
                break
        if device.type == "mps":
            torch.mps.empty_cache()
    log_file.close()

    # ---- Final evaluation with the best checkpoint ---------------------------
    model = AutoModelForSequenceClassification.from_pretrained(best_dir, dtype=torch.float32).to(device)
    dev_logits, dev_labels = predict(model, dev_loader, device)
    test_logits, test_labels = predict(model, test_loader, device)

    temperature = fit_temperature(dev_logits, dev_labels)
    dev_probs, test_probs = softmax(dev_logits, temperature), softmax(test_logits, temperature)
    test_metrics = compute_metrics(test_labels, test_probs, none_id)
    test_metrics["ece_raw"] = expected_calibration_error(softmax(test_logits), test_labels)
    test_metrics["ece_calibrated"] = expected_calibration_error(test_probs, test_labels)
    test_by_source = {
        src: compute_metrics(test_labels[m], test_probs[m], none_id)
        for src in test_df["source"].unique()
        for m in [(test_df["source"] == src).to_numpy()]
    }

    # Store the temperature with the model so inference code can reuse it.
    (best_dir / "calibration.json").write_text(json.dumps({"temperature": temperature}, indent=2))

    report = classification_report(test_labels, test_probs.argmax(1), labels=list(range(len(labels))),
                                   target_names=labels, digits=3, zero_division=0)
    (run_dir / "test_report.txt").write_text(report)
    pd.DataFrame(confusion_matrix(test_labels, test_probs.argmax(1), labels=list(range(len(labels)))),
                 index=labels, columns=labels).to_csv(run_dir / "test_confusion.csv")
    save_predictions(dev_df, dev_probs, labels, run_dir / "dev_predictions.parquet")
    save_predictions(test_df, test_probs, labels, run_dir / "test_predictions.parquet")

    summary = {
        "best_epoch": best_epoch,
        f"best_dev_{args.select_metric}": best_score,
        "temperature": temperature,
        "dev": compute_metrics(dev_labels, dev_probs, none_id),
        "test": test_metrics,
        "test_by_source": test_by_source,
        "train_minutes": (time.time() - t0) / 60,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n" + report)
    print("test " + "  ".join(f"{k}={v:.3f}" for k, v in test_metrics.items()))
    print(f"temperature={temperature:.3f}  outputs in {run_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2, help="Stop after this many epochs without dev gain")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=160, help="Covers ~97%% of training texts")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weighting", choices=["none", "sqrt", "balanced"], default="sqrt",
                   help="'none' is ~55%% of train; sqrt weighting softens that without over-correcting")
    p.add_argument("--select-metric", default="macro_f1",
                   choices=["macro_f1", "fallacy_macro_f1", "binary_f1", "accuracy"])
    p.add_argument("--with-context", action="store_true",
                   help="Feed the thread/article title as a first segment")
    p.add_argument("--drop-sources", nargs="*", default=[],
                   help="e.g. sahai_random_neg to train without unverified negatives")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


if __name__ == "__main__":
    main()
