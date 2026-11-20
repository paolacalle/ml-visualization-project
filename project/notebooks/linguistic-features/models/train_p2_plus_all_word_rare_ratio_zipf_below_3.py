"""Add the primary all-word Zipf < 3 rarity ratio to the P2 feature set."""

from paired_model_comparison import CONTENT_ZIPF, RARE_WORD_RATIO, WORD_COUNT, run_model


if __name__ == "__main__":
    run_model(
        "p2_plus_all_word_rare_ratio_zipf_below_3",
        [WORD_COUNT, CONTENT_ZIPF, RARE_WORD_RATIO],
    )
