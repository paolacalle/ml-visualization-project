# Logical-fallacy model

Fine-tuned classifier that tags argument text with one of 8 informal fallacies
or `none`. Its per-comment scores become features for the interpretable
delta classifier.

## Labels

`none`, `appeal to authority`, `appeal to majority`, `appeal to nature`,
`appeal to tradition`, `appeal to worse problems`, `false dilemma`,
`hasty generalization`, `slippery slope` (the order in `labels.json` gives `label_id`).

## Building the dataset

Install into the project venv (Anaconda's older protobuf breaks the DeBERTa
tokenizer):

```bash
.venv/bin/python -m pip install -r requirements.txt -r project/notebooks/fallacy-model/requirements.txt
.venv/bin/python project/notebooks/fallacy-model/build_dataset.py
```

This downloads the raw data (pinned commits) into `project/data/raw/fallacy/`
and writes `train/dev/test.parquet` + `labels.json` to
`project/data/processed/fallacy/`. Both folders are git-ignored; rerunning is
deterministic (`--seed 13`).

| Source | What it adds | Split |
|---|---|---|
| [CoCoLoFa](https://github.com/Crowd-AI-Lab/cocolofa) (EMNLP 2024) | 7.7k crowd-written news comments, each with a fallacy or `none` | Official, by article |
| [Sahai et al.](https://github.com/sahaisaumya/informal_fallacies) (ACL 2021), annotated | Reddit comments another user called fallacious; crowd-verified positives, rejected ones become `none` | 80/10/10 by thread |
| Sahai et al., negative samples | Random Reddit comments, not human-verified (`--n-random-neg`, default 3000 of ~6.2k) | 80/10/10 by thread |

Current build: 13,856 rows (train 10,279 / dev 2,159 / test 1,418); `none` is
about 55%, and each fallacy has 710–870 examples. Median length is 49 words.

Columns: `text`, `label`, `label_id`, `source`, `group` (article or thread, no
group crosses splits), `context` (article or thread title), `n_words`.

### Known caveats

- **Sahai hard negatives are only negative for the flagged fallacy.** Crowd
  workers checked for the one fallacy a Reddit user named, so a few `none` rows
  contain a different fallacy. Use `source` to down-weight or drop them if they
  turn out to hurt.
- **Random negatives are unverified** and often off-topic chatter, so they are
  easier than real CMV arguments.
- **Short texts.** Training examples are mostly under 70 words, while CMV comments
  are often several hundred. At inference, score overlapping 2–4 sentence
  windows and aggregate them, rather than truncating the whole comment.
- **Not CMV.** Hand-label a few hundred CMV windows as the real test set and
  for threshold tuning.

## Training

```bash
# smoke test (~1 min): checks the pipeline, too few steps to learn much
.venv/bin/python project/notebooks/fallacy-model/train.py --run-name smoke \
    --max-train-samples 480 --max-eval-samples 240 --epochs 1

# DeBERTa-v3-base, full data (~8 min/epoch on an M4, up to 5 epochs)
.venv/bin/python project/notebooks/fallacy-model/train.py --run-name deberta-base

# DeBERTa-v3-large, overnight
.venv/bin/python project/notebooks/fallacy-model/train.py --run-name deberta-large \
    --model microsoft/deberta-v3-large --batch-size 8 --grad-accum 2 --lr 1e-5
```

`train.py` is a plain PyTorch loop (no `accelerate`/`datasets`), running on
Apple's GPU (MPS) when available. It:

- trains a 9-way classifier with sqrt-inverse-frequency class weights
  (`--class-weighting`), linear warmup/decay and gradient clipping;
- evaluates on dev after each epoch, keeps the best checkpoint by
  `--select-metric` (default `macro_f1`), and stops early after `--patience`
  epochs without improvement;
- fits a temperature on dev logits (saved as `best/calibration.json`) so
  `softmax(logits / T)` gives calibrated probabilities for downstream features;
- reports on test: 9-way macro-F1, fallacy-only macro-F1, binary
  fallacy-vs-none precision/recall/F1, calibration error, and each metric per `source`.

Outputs go to `project/data/models/fallacy/<run-name>/`: `best/` (model,
tokenizer, temperature), `metrics.json`, `test_report.txt`,
`test_confusion.csv`, `{dev,test}_predictions.parquet` (per-class probabilities
per row, for error analysis) and `train_log.jsonl`.

### Results: `deberta-base` (DeBERTa-v3-base, default settings)

Best dev epoch 4 of 5, ~32 min of training on an M4. Test (1,418 rows):
accuracy 0.81, **macro-F1 0.77**, binary fallacy-vs-none F1 0.82. Calibration
error drops from 0.13 to 0.03 with the dev-fitted temperature (T = 1.60).

| Test subset | Macro-F1 | Binary F1 |
|---|---|---|
| CoCoLoFa | 0.83 | 0.88 |
| Sahai annotated (real Reddit) | 0.60 | 0.68 |

Per class, F1 ranges from 0.84 (appeal to nature) to 0.58 (hasty
generalization, the weakest). Most errors are fallacies predicted as `none`
and vice versa, not confusions between fallacy types. The real-Reddit numbers
are the better guide to CMV performance.

Useful flags: `--drop-sources sahai_random_neg` (train without unverified
negatives), `--with-context` (feed the thread/article title as a first
segment), `--max-length` (default 160 tokens, covering ~97% of texts).

### Apple-GPU (MPS) notes

- transformers v5 loads DeBERTa-v3 in its stored fp16, which crashes MPS, so
  the script forces fp32.
- Batches are padded to multiples of 32 tokens and the MPS cache is cleared every
  50 steps. Without that, every new batch length compiles a new graph and the
  run ran out of memory after about 100 steps.
- transformers 5.3 converts DeBERTa-v3's sentencepiece tokenizer without adding
  `[CLS]`/`[SEP]`. The classifier reads the first token, so without the fix
  DeBERTa learned very slowly (0.32 dev macro-F1 after 2 epochs, vs DistilRoBERTa's
  0.68 on 3k examples). `ensure_special_tokens()` in `train.py` patches this and
  the fix is saved with `best/`. **Any other script that loads
  `microsoft/deberta-v3-*` straight from the Hub must call it too.**
