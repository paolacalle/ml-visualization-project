"""Regroup archived linguistic scores and compare the broad topic associations."""

from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from summarize_paired_linguistic_strategies import ALL, CLOSE, DRAWS, MAIN_FEATURES, OTHER, OUT, WINDOW, intervals, paired, topic_plot

SOURCE_RUN = "20261121T203952524644Z"
RESULTS = OUT.parent / "paired-linguistic-broad-topics"
MAPPING = Path(__file__).with_name("broad_topic_groups.csv")
GROUPS = ("Society & Values", "Work, Technology & Economics", "Culture & Everyday Life")
EXPECTED_COUNTS = dict(zip([*GROUPS, OTHER], [1841, 739, 554, 317]))


def load_pairs():
    source = OUT / SOURCE_RUN
    report = json.loads((source / "results.json").read_text())
    pairs = pd.read_parquet(source / "pair_strategy_scores.parquet")
    frozen = pd.read_parquet(paired.DATA / "training_pair_assignments.parquet")
    pd.testing.assert_frame_equal(pairs[frozen.columns], frozen)
    if report["split"] != "train" or len(pairs) != 3451 or not pairs["pair_id"].is_unique:
        raise ValueError("Expected the 3,451 frozen training pairs.")
    mapping = pd.read_csv(MAPPING).set_index("broad_topic", drop=True)
    if not mapping.index.is_unique or set(mapping.index) != set(pairs["broad_topic"]):
        raise ValueError("The mapping must cover each archived topic exactly once.")
    pairs["topic_group"] = pairs["broad_topic"].map(mapping["topic_group"])
    if pairs["topic_group"].value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Unexpected assignment counts for the approved regrouping.")
    if not pairs.groupby("post_id")["topic_group"].nunique().eq(1).all():
        raise ValueError("A discussion's pairs must share one broad group.")
    return pairs


def compare_groups(summary, draws, cohort):
    rows = []
    for first, second in combinations(GROUPS, 2):
        for feature in MAIN_FEATURES:
            for metric in ("winner_higher_rate", "mean_difference"):
                first_draw, second_draw = (draws[group][feature][metric] for group in (first, second))
                delta = first_draw - second_draw
                valid = delta[np.isfinite(delta)]
                low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else [np.nan, np.nan]
                rows.append({
                    "subset": cohort, "group_a": first, "group_b": second, "feature": feature, "metric": metric,
                    "difference": summary.loc[(first, feature), metric] - summary.loc[(second, feature), metric],
                    "ci_low": low, "ci_high": high, "valid_draws": len(valid),
                    "unequal_pairs_a": summary.loc[(first, feature), "non_ties"],
                    "unequal_pairs_b": summary.loc[(second, feature), "non_ties"],
                })
    return pd.DataFrame(rows)


def analyze(pairs):
    orientation = 2 * pairs["a_wins"] - 1
    differences = pd.DataFrame({
        name: (pairs[f"a_{name}"] - pairs[f"b_{name}"]) * orientation for name in MAIN_FEATURES
    })
    rng = np.random.default_rng(paired.SEED)
    summaries, contrasts, detail = [], [], []
    for cohort, frame in [(ALL, pairs), (CLOSE, pairs.loc[pairs["within_50_words"]])]:
        results, draws = [], {}
        for name, group in frame.groupby("topic_group"):
            stats, draws[name] = intervals(group, differences.loc[group.index], rng=rng, return_draws=True)
            results.append(stats.assign(topic_group=name, subset=cohort, pairs=len(group)))
        summary = pd.concat(results, ignore_index=True)
        summaries.append(summary)
        contrasts.append(compare_groups(summary.set_index(["topic_group", "feature"]), draws, cohort))
        for name, group in frame.groupby("broad_topic"):
            stats = intervals(group, differences.loc[group.index])
            detail.append(stats.assign(broad_topic=name, topic_group=group["topic_group"].iloc[0],
                                       subset=cohort, pairs=len(group)))
    return tuple(pd.concat(frames, ignore_index=True) for frames in (summaries, contrasts, detail))


def main():
    pairs = load_pairs()
    with threadpool_limits(limits=1):
        summaries, contrasts, detail = analyze(pairs)
    chart = summaries.loc[summaries["subset"].eq(ALL)].rename(columns={"topic_group": "cohort"}).assign(group_type=WINDOW)
    fig = topic_plot(chart, topics=GROUPS, title="Paired linguistic strategies across three OP topic groups")
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=RESULTS) as temporary:
        out = Path(temporary)
        pairs.to_parquet(out / "pair_strategy_scores_with_topic_group.parquet", index=False)
        summaries.to_csv(out / "broad_topic_strategy_associations.csv", index=False)
        contrasts.to_csv(out / "broad_topic_bootstrap_contrasts.csv", index=False)
        detail.to_csv(out / "eight_topic_strategy_associations.csv", index=False)
        fig.savefig(out / "broad_topic_strategy_heatmap.png", dpi=175, facecolor="white")
        plt.close(fig)
        (out / "results.json").write_text(json.dumps({
            "split": "train", "feature_run": SOURCE_RUN, "bootstrap_draws": DRAWS, "mapping": MAPPING.name,
            "pairs": len(pairs), "discussions": pairs["post_id"].nunique(),
            "pairs_by_topic_group": pairs["topic_group"].value_counts().to_dict(),
            "length_matched_pairs_by_topic_group": pairs.loc[
                pairs["within_50_words"], "topic_group"
            ].value_counts().to_dict(),
        }, indent=2) + "\n")
        out.rename(destination)
    print(summaries[["subset", "topic_group", "feature", "pairs", "non_ties", "winner_higher_rate",
                     "winner_higher_ci_low", "winner_higher_ci_high"]].round(4).to_string(index=False))
    print(f"Saved regrouped training analysis to {destination}")


if __name__ == "__main__":
    main()
