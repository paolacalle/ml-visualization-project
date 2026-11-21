"""How do fallacy flags relate to a comment winning an OP delta?

Reads the scored matched sample (score_comments.py) and reports, for "any
fallacy" and each of the 8 fallacy types:

* prevalence among delta vs non-delta comments, and the raw odds ratio;
* the point-biserial correlation between the model's score and the delta label;
* the odds ratio from a conditional (fixed-effects) logistic regression that
  compares comments only within the same post and controls for log length.
  Length matters: delta comments are ~2x longer, and a longer comment has more
  windows that can be flagged.
* a joint conditional model with all 8 flags at once (relative importance);
* a length-robust variant using the share of windows flagged instead of a
  0/1 flag.

A comment is "flagged" for fallacy k when any window has calibrated
P(k) >= --threshold (default 0.5).

Usage (repo root):
    .venv/bin/python project/notebooks/fallacy-analysis/analyze_fallacy_delta.py
Writes results/ and figures/ next to this script.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUN_DIR = REPO_ROOT / "project" / "data" / "processed" / "cmv-fallacy" / "matched-5"

FALLACIES = [
    "appeal_to_authority", "appeal_to_majority", "appeal_to_nature", "appeal_to_tradition",
    "appeal_to_worse_problems", "false_dilemma", "hasty_generalization", "slippery_slope",
]
PRETTY = {f: f.replace("_", " ") for f in FALLACIES} | {"fallacy": "any fallacy"}

# Reference palette (dataviz skill), light mode.
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(run_dir: Path) -> pd.DataFrame:
    sample = pd.read_parquet(run_dir / "sample.parquet")
    scores = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(run_dir / "comments" / "*.parquet"))))
    df = sample.merge(scores, on="comment_id", how="inner")
    missing = len(sample) - len(df)
    if missing:
        print(f"note: {missing:,} sampled comments are not scored (no windows, or scoring unfinished)")
    df["y"] = df["got_delta_op"].astype(int)
    df["log_words"] = np.log1p(df["n_words"])
    # Conditional logit only learns from posts with both outcomes.
    mixed = df.groupby("post_id")["y"].transform(lambda s: 0 < s.mean() < 1)
    return df[mixed].reset_index(drop=True)


def conditional_or(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols + ["log_words"]].astype(float)
    res = ConditionalLogit(df["y"], X, groups=df["post_id"]).fit(disp=False, method="bfgs", maxiter=500)
    ci = res.conf_int()
    return pd.DataFrame({
        "or": np.exp(res.params), "or_lo": np.exp(ci[0]), "or_hi": np.exp(ci[1]), "p": res.pvalues,
    }).loc[cols]


def raw_or(flag: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    a = ((flag == 1) & (y == 1)).sum() + 0.5
    b = ((flag == 1) & (y == 0)).sum() + 0.5
    c = ((flag == 0) & (y == 1)).sum() + 0.5
    d = ((flag == 0) & (y == 0)).sum() + 0.5
    log_or = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return np.exp(log_or), np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se)


def holm(p: pd.Series) -> pd.Series:
    return pd.Series(multipletests(p.to_numpy(), method="holm")[1], index=p.index)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()

    df = load(args.run_dir)
    keys = ["fallacy"] + FALLACIES
    for k in keys:
        df[f"flag_{k}"] = (df[f"max_p_{k}"] >= args.threshold).astype(int)
    print(f"{len(df):,} comments ({df['y'].sum():,} with an OP delta) in {df['post_id'].nunique():,} posts\n")

    rows = []
    for k in keys:
        flag, score = df[f"flag_{k}"], df[f"max_p_{k}"]
        r, r_p = stats.pointbiserialr(df["y"], score)
        r_or, r_lo, r_hi = raw_or(flag, df["y"])
        adj = conditional_or(df, [f"flag_{k}"]).iloc[0]
        dens = df[f"frac_windows_{k}"]
        df[f"z_frac_{k}"] = (dens - dens.mean()) / dens.std()
        adj_d = conditional_or(df, [f"z_frac_{k}"]).iloc[0]
        rows.append({
            "fallacy": PRETTY[k],
            "flagged_delta": flag[df["y"] == 1].mean(),
            "flagged_no_delta": flag[df["y"] == 0].mean(),
            "r_pointbiserial": r,
            "raw_or": r_or, "raw_or_lo": r_lo, "raw_or_hi": r_hi,
            "adj_or": adj["or"], "adj_or_lo": adj["or_lo"], "adj_or_hi": adj["or_hi"], "adj_p": adj["p"],
            "density_or_per_sd": adj_d["or"], "density_p": adj_d["p"],
        })
    res = pd.DataFrame(rows).set_index("fallacy")
    res["adj_p_holm"] = holm(res["adj_p"])

    joint = conditional_or(df, [f"flag_{k}" for k in FALLACIES])
    joint.index = [PRETTY[k] for k in FALLACIES]
    joint["p_holm"] = holm(joint["p"])

    (args.out_dir / "results").mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_dir / "results" / "fallacy_delta_univariate.csv")
    joint.to_csv(args.out_dir / "results" / "fallacy_delta_joint.csv")

    pd.set_option("display.width", 200)
    show = res[["flagged_delta", "flagged_no_delta", "r_pointbiserial", "raw_or",
                "adj_or", "adj_or_lo", "adj_or_hi", "adj_p_holm", "density_or_per_sd", "density_p"]]
    print("Per fallacy (adjusted = within-post conditional logit + log length):")
    print(show.round(3).to_string())
    print("\nJoint model, all 8 flags + log length:")
    print(joint.round(3).to_string())

    plot(res, args.out_dir / "figures" / "fallacy_delta_odds_ratios.png")


def plot(res: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["any fallacy"] + res.drop("any fallacy")["adj_or"].sort_values().index.tolist()[::-1]
    res = res.loc[order]
    y = np.arange(len(res))[::-1]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(1, color=BASE, lw=1, zorder=1)
    ax.axhline(y[0] - 0.5, color=GRID, lw=0.8, zorder=1)

    # Fixed, symmetric (in log space) range; intervals past it are clipped at the edge.
    lo, hi = 0.4, 2.5
    clip = lambda s: s.clip(lo * 1.01, hi / 1.01)  # noqa: E731
    off = 0.14
    ax.hlines(y + off, clip(res["raw_or_lo"]), clip(res["raw_or_hi"]), color=ORANGE, lw=2, zorder=2)
    ax.scatter(clip(res["raw_or"]), y + off, s=34, marker="D", facecolor=SURFACE, edgecolor=ORANGE,
               lw=1.6, zorder=3, label="Raw (no controls)")
    ax.hlines(y - off, clip(res["adj_or_lo"]), clip(res["adj_or_hi"]), color=BLUE, lw=2, zorder=2)
    ax.scatter(clip(res["adj_or"]), y - off, s=40, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4,
               label="Same post, length-adjusted")

    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xlim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels(res.index)
    for lab in ax.get_yticklabels():
        if lab.get_text() == "any fallacy":
            lab.set_fontweight("bold")
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Odds of winning an OP delta when flagged (log scale)", color=INK2, labelpad=16)
    ax.text(0, -0.075, "← less likely to win a delta", transform=ax.transAxes,
            ha="left", va="top", color=MUTED, fontsize=8)
    ax.text(1, -0.075, "more likely →", transform=ax.transAxes,
            ha="right", va="top", color=MUTED, fontsize=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK2, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.text(0.02, 0.975, "Fallacy flags and CMV deltas", ha="left", va="top",
             color=INK, fontsize=13, fontweight="bold")
    fig.text(0.02, 0.925, "Odds ratios, 95% CIs. Delta comments vs. non-delta comments from the same posts.",
             ha="left", va="top", color=INK2, fontsize=9)
    fig.savefig(path, facecolor=SURFACE)
    print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
