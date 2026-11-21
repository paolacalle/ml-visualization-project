"""Plot paired associations for P4's features by draft OP topic."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from discover_op_topics import OUT as TOPIC_RUNS, P4, paired


DISCOVERY_RUN = "20261120T221529517368Z"
MAPPING = Path(__file__).with_name(f"draft_topic_labels_{DISCOVERY_RUN}.csv")
OUT = paired.DATA / "topic-feature-associations"
UNCLASSIFIED = "Other / unclear"
FEATURES = {
    "Longer reply": ("a_minus_b_word_count", 1),
    "Less familiar\nvocabulary": ("a_minus_b_mean_content_word_zipf", -1),
    "Earlier reply": ("a_minus_b_log1p_reply_delay_minutes", -1),
    "External link": ("a_minus_b_has_external_link", 1),
}


def load_pairs():
    topic_run = TOPIC_RUNS / DISCOVERY_RUN
    topic_report = json.loads((topic_run / "results.json").read_text())
    p4_report = json.loads((P4 / "results.json").read_text())
    if topic_report["split"] != "train" or p4_report["split"] != "train" or topic_report["p4_run"] != P4.name:
        raise ValueError("Expected draft topics for the frozen P4 training cohort.")
    assignments = pd.read_parquet(topic_run / "op_topic_assignments.parquet", columns=["post_id", "topic_id"])
    mapping = pd.read_csv(MAPPING, usecols=["topic_id", "broad_topic"])
    if set(assignments["topic_id"]) != set(mapping["topic_id"]):
        raise ValueError("The draft mapping must cover every discovered cluster.")
    topics = assignments.merge(mapping, on="topic_id", validate="many_to_one")
    cols = [column for column, _ in FEATURES.values()]
    keys = ["pair_id", "post_id", "a_wins", "fold"]
    pairs = pd.read_parquet(P4 / "oof_predictions.parquet", columns=[*keys, *cols])
    frozen = pd.read_parquet(paired.DATA / "training_pair_assignments.parquet", columns=keys)
    pd.testing.assert_frame_equal(pairs[keys], frozen)
    pairs = pairs.merge(topics, on="post_id", how="left", validate="many_to_one")
    if len(pairs) != p4_report["pairs"] or pairs["broad_topic"].isna().any():
        raise ValueError("Every P4 training pair must have a topic.")
    if not np.isfinite(pairs[cols].to_numpy()).all():
        raise ValueError("Each P4 pair must have finite feature differences.")
    return pairs


def summarize(pairs):
    ordered = pairs["broad_topic"].value_counts().drop(index=UNCLASSIFIED).index.tolist()
    topics = ["All topics", *ordered, UNCLASSIFIED]
    rows = []
    for topic in topics:
        group = pairs if topic == "All topics" else pairs.loc[pairs["broad_topic"].eq(topic)]
        orientation = 2 * group["a_wins"] - 1
        for name, (column, direction) in FEATURES.items():
            winner_difference = group[column] * orientation * direction
            stat = paired.paired_higher(group, winner_difference)
            low, high = stat["ci_95"]
            rows.append({
                "topic": topic, "feature": name.replace("\n", " "),
                "pairs": stat["pairs"], "discussions": group["post_id"].nunique(),
                "non_tied_pairs": stat["pairs"] - stat["ties"],
                "non_tied_discussions": group.loc[
                    winner_difference.abs().gt(paired.TIE_TOLERANCE), "post_id"
                ].nunique(),
                "winner_has_feature": stat["winner_higher"],
                "comparison_has_feature": stat["winner_lower"], "ties": stat["ties"],
                "winner_share": stat["winner_higher_rate"], "ci_95_lower": low, "ci_95_upper": high,
            })
    return pd.DataFrame(rows), topics


def plot_heatmap(frame, topics):
    columns = list(FEATURES)
    labels = [name.replace("\n", " ") for name in columns]
    cells = frame.set_index(["topic", "feature"])
    shares = frame.pivot(index="topic", columns="feature", values="winner_share").loc[topics, labels]
    colors = LinearSegmentedColormap.from_list("win_share", ["#a75144", "#f8f7f3", "#267589"])
    # The scale is symmetric around chance, with margin around the observed cells.
    spread = max(0.25, float(np.nanmax(np.abs(shares.to_numpy() - 0.5))))
    norm = TwoSlopeNorm(vmin=0.5 - spread, vcenter=0.5, vmax=0.5 + spread)
    fig, ax = plt.subplots(figsize=(11.4, 8.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.10)
    image = ax.imshow(shares.to_numpy(), cmap=colors, norm=norm, aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=10, labelsize=10)
    ax.set_yticks(range(len(topics)), [
        f"{topic}\n{int(cells.loc[(topic, labels[0]), 'pairs']):,} pairs" for topic in topics
    ])
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(topics), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False, top=False)
    ax.axhline(0.5, color="#313742", lw=1.2)
    ax.spines[:].set_visible(False)
    for y, topic in enumerate(topics):
        for x, feature in enumerate(labels):
            row = cells.loc[(topic, feature)]
            share = row["winner_share"]
            if pd.isna(share):
                ax.text(x, y, "No unequal pairs", ha="center", va="center", fontsize=9)
                continue
            low, high = row[["ci_95_lower", "ci_95_upper"]]
            red, green, blue, _ = colors(norm(share))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ink = "#17202a" if luminance > 0.56 else "white"
            ax.text(x, y - 0.16, f"{share:.1%}", ha="center", va="center", weight="bold", fontsize=12, color=ink)
            ax.text(x, y + 0.13, f"{low:.1%}–{high:.1%}  ·  n={int(row['non_tied_pairs']):,}",
                    ha="center", va="center", fontsize=8.7, color=ink)

    fig.suptitle("Which features did winning replies have?", x=0.04, ha="left", fontsize=21, weight="bold")
    bar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.04, shrink=0.72, aspect=30,
                       ticks=[0.5 - spread, 0.5, 0.5 + spread], format=PercentFormatter(1))
    bar.outline.set_visible(False)
    bar.set_label("Winner had the feature among pairs that differ · 95% discussion-bootstrap interval and unequal-pair n\n"
                  "Draft OP topics · Published training pairs",
                  fontsize=9.5, labelpad=8)
    return fig


def main():
    pairs = load_pairs()
    frame, topics = summarize(pairs)
    fig = plot_heatmap(frame, topics)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        frame.to_csv(staging / "topic_feature_associations.csv", index=False)
        fig.savefig(staging / "topic_feature_association_heatmap.png", dpi=180, facecolor="white")
        staging.rename(destination)
    plt.close(fig)
    print(f"Saved topic feature heatmap and data to {destination}")


if __name__ == "__main__":
    main()
