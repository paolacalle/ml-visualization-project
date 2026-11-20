# Logical-fallacy model

Fine-tuned classifier that tags argument text with one of 8 informal fallacies
or `none`. Its per-comment scores become features for the interpretable
delta classifier.

## Labels

`none`, `appeal to authority`, `appeal to majority`, `appeal to nature`,
`appeal to tradition`, `appeal to worse problems`, `false dilemma`,
`hasty generalization`, `slippery slope` (the order in `labels.json` gives `label_id`).

## Building the dataset

```bash
python -m pip install -r project/notebooks/fallacy-model/requirements.txt
python project/notebooks/fallacy-model/build_dataset.py
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
