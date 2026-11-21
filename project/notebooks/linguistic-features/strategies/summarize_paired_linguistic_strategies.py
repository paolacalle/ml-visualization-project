"""Describe paired strategy differences using OP-discussion bootstrap intervals."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from extract_paired_linguistic_strategies import MAIN_FEATURES, OUT, paired

RESULTS = OUT.parent / "paired-linguistic-strategy-summary"
DRAWS = 1000
ALL = "All training pairs"
CLOSE = "Within 50 words"
OTHER = "Other / unclear"
WINDOW = "topic"


def intervals(frame, differences, *, rng=None, return_draws=False):
    """Use the same resampled OPs for every statistic within a cohort."""
    groups, posts = pd.factorize(frame["post_id"])
    g = len(posts)
    if rng is None:
        rng = np.random.default_rng(paired.SEED)
    weights = rng.multinomial(g, np.full(g, 1 / g), size=DRAWS)
    rows, draws = [], {}
    for name in differences:
        values = differences[name].to_numpy(dtype=float)
        present = np.isfinite(values)
        measured = values[present]
        codes = groups[present]
        if not len(measured):
            rows.append({"feature": name, "complete_pairs": 0})
            continue
        wins = measured > paired.TIE_TOLERANCE
        losses = measured < -paired.TIE_TOLERANCE
        nontied = wins | losses
        stats = np.column_stack([
            np.bincount(codes, weights=measured, minlength=g),
            np.bincount(codes, minlength=g),
            np.bincount(codes, weights=wins, minlength=g),
            np.bincount(codes, weights=nontied, minlength=g),
        ])
        sums, count, wins_draw, unequal_draw = (weights @ stats).T
        mean_draw = np.divide(sums, count, out=np.full(DRAWS, np.nan), where=count > 0)
        higher_draw = np.divide(wins_draw, unequal_draw, out=np.full(DRAWS, np.nan), where=unequal_draw > 0)
        order = np.argsort(measured, kind="stable")
        ordered_values = measured[order]
        sorted_codes = codes[order]
        median_draw = []
        for start in range(0, DRAWS, 50):
            cumulative = weights[start:start + 50, sorted_codes].cumsum(axis=1)
            total = cumulative[:, -1]
            low = (cumulative < ((total + 1) // 2)[:, None]).sum(axis=1)
            high = (cumulative < ((total + 2) // 2)[:, None]).sum(axis=1)
            median_draw.extend(np.where(
                total > 0, (ordered_values[low.clip(max=len(measured) - 1)]
                            + ordered_values[high.clip(max=len(measured) - 1)]) / 2, np.nan,
            ))
        mean_ci = np.nanquantile(mean_draw, [0.025, 0.975])
        median_ci = np.nanquantile(median_draw, [0.025, 0.975])
        higher_ci = np.nanquantile(higher_draw, [0.025, 0.975]) if nontied.any() else [np.nan, np.nan]
        if return_draws:
            draws[name] = {"mean_difference": mean_draw, "winner_higher_rate": higher_draw}
        rows.append({
            "feature": name, "complete_pairs": len(measured), "discussions": len(np.unique(codes)),
            "ties": int((~nontied).sum()), "non_ties": int(nontied.sum()),
            "winner_higher": int(wins.sum()), "winner_lower": int(losses.sum()),
            "mean_difference": measured.mean(), "mean_ci_low": mean_ci[0], "mean_ci_high": mean_ci[1],
            "median_difference": np.median(measured), "median_ci_low": median_ci[0], "median_ci_high": median_ci[1],
            "winner_higher_rate": wins.sum() / nontied.sum() if nontied.any() else np.nan,
            "winner_higher_ci_low": higher_ci[0], "winner_higher_ci_high": higher_ci[1],
        })
    result = pd.DataFrame(rows)
    return (result, draws) if return_draws else result


def summarize(pairs):
    features = [*MAIN_FEATURES, "sentiment_negative", "sentiment_neutral", "sentiment_positive"]
    features.extend(column[2:] for column in pairs if column.startswith("a_strategy::"))
    a = pairs[[f"a_{feature}" for feature in features]].set_axis(features, axis=1)
    b = pairs[[f"b_{feature}" for feature in features]].set_axis(features, axis=1)
    difference = (a - b).mul(2 * pairs["a_wins"] - 1, axis=0)
    cohorts = {ALL: pairs, CLOSE: pairs.loc[pairs["within_50_words"]]}
    results = [
        intervals(group, difference.loc[group.index]).assign(cohort=name, group_type="overall", pairs=len(group))
        for name, group in cohorts.items()
    ]
    for name, group in pairs.groupby("broad_topic"):
        results.append(intervals(group, difference.loc[group.index]).assign(cohort=name, group_type=WINDOW, pairs=len(group)))
    return pd.concat(results, ignore_index=True)


def paired_plot(result):
    fig, ax = plt.subplots(figsize=(9.4, 4.2), layout="constrained")
    x = np.arange(len(MAIN_FEATURES))
    for cohort, color, offset in [(ALL, "#176d83", -0.11), (CLOSE, "#805999", 0.11)]:
        rows = result.loc[result["cohort"].eq(cohort)].set_index("feature").loc[list(MAIN_FEATURES)]
        y = x + offset
        ax.hlines(y, rows["winner_higher_ci_low"], rows["winner_higher_ci_high"], color=color, lw=2)
        ax.scatter(rows["winner_higher_rate"], y, color=color, s=38, label=cohort, zorder=3)
        for position, count in zip(y, rows["non_ties"]):
            ax.text(1.03, position, f"n = {int(count):,}", transform=ax.get_yaxis_transform(),
                    color=color, fontweight="bold", fontsize=10, va="center", clip_on=False)
    ax.axvline(0.5, color="#627080", ls="--", lw=1)
    ax.set_yticks(x, MAIN_FEATURES.values())
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="How often did the winning reply score higher?", xlabel="Share among unequal-score pairs · 95% discussion-bootstrap interval")
    ax.text(1.03, 1.02, "Unequal pairs", transform=ax.transAxes, fontsize=10, weight="bold")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.20)
    return fig


def topic_plot(result, *, topics=None, title="Paired linguistic strategies by draft OP topic"):
    topic = result.loc[result["group_type"].eq(WINDOW) & result["feature"].isin(MAIN_FEATURES)]
    order = topic.groupby("cohort")["pairs"].first().sort_values(ascending=False)
    if topics is None:
        topics = [*order.index.drop(OTHER), OTHER]
    fields = list(MAIN_FEATURES)
    rows = topic.set_index(["cohort", "feature"])
    matrix = topic.pivot(index="cohort", columns="feature", values="winner_higher_rate").loc[topics, fields]
    spread = max(0.25, float(np.nanmax(np.abs(matrix.to_numpy() - 0.5))))
    palette = LinearSegmentedColormap.from_list("strategy_share", ["#a75144", "#f8f7f3", "#267589"])
    norm = TwoSlopeNorm(vmin=0.5 - spread, vcenter=0.5, vmax=0.5 + spread)
    fig, ax = plt.subplots(figsize=(11.1, 7.2 if len(topics) > 4 else 4.5), layout="constrained")
    image = ax.imshow(matrix.to_numpy(), cmap=palette, norm=norm, aspect="auto")
    ax.set_xticks(range(4), ["Courtesy\nmarkers", "Hedges per\n100 words", "Net positive\nsentiment", "OP–reply\nsemantic similarity"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(topics)), [f"{name}\n{int(order[name]):,} pairs" for name in topics])
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, len(topics)), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.spines[:].set_visible(False)
    for y, group in enumerate(topics):
        for x, feature in enumerate(fields):
            cell = rows.loc[(group, feature)]
            share, low, high = cell[["winner_higher_rate", "winner_higher_ci_low", "winner_higher_ci_high"]]
            if not np.isfinite(share):
                ax.text(x, y, "No unequal pairs", ha="center", va="center")
                continue
            r, g, b, _ = palette(norm(share))
            ink = "#17202a" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"
            ax.text(x, y - 0.15, f"{share:.1%}", ha="center", va="center", fontweight="bold", color=ink)
            ax.text(x, y + 0.13, f"{low:.1%}–{high:.1%} · n={int(cell['non_ties']):,}", ha="center", va="center",
                    fontsize=8.6, color=ink)
    fig.suptitle(title, weight="bold", fontsize=17)
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.04, shrink=0.72, aspect=28,
                       format=PercentFormatter(1), ticks=[0.5-spread, 0.5, 0.5+spread])
    bar.set_label("Winner scored higher · 95% discussion-bootstrap interval · n = unequal pairs", fontsize=9)
    bar.outline.set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-run")
    args = parser.parse_args()
    runs = sorted(OUT.glob("*/results.json"))
    source = OUT / args.feature_run if args.feature_run else runs[-1].parent
    report = json.loads((source / "results.json").read_text())
    pairs = pd.read_parquet(source / "pair_strategy_scores.parquet")
    if report["split"] != "train" or len(pairs) != report["pairs"]:
        raise ValueError("Expected the frozen training-pair features.")
    frozen = pd.read_parquet(paired.DATA / "training_pair_assignments.parquet")
    pd.testing.assert_frame_equal(pairs[frozen.columns], frozen)
    with threadpool_limits(limits=1):
        results = summarize(pairs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=RESULTS) as directory:
        stage = Path(directory)
        results.to_csv(stage / "paired_strategy_associations.csv", index=False)
        for figure, name in [(paired_plot(results), "paired_strategy_differences.png"),
                             (topic_plot(results), "topic_strategy_heatmap.png")]:
            figure.savefig(stage / name, dpi=175, facecolor="white")
            plt.close(figure)
        (stage / "results.json").write_text(json.dumps({
            "split": "train", "feature_run": source.name, "bootstrap_draws": DRAWS,
            "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "length_matched_pairs": int(pairs["within_50_words"].sum()),
            "length_matched_discussions": pairs.loc[pairs["within_50_words"], "post_id"].nunique(),
        }, indent=2) + "\n")
        stage.rename(dest)
    print(results.loc[results["feature"].isin(MAIN_FEATURES) & results["group_type"].eq("overall")].round(4).to_string(index=False))
    print(f"Saved strategy comparison to {dest}")


if __name__ == "__main__":
    main()
