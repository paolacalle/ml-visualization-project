"""P2 predicts the winning side using word-count and mean content-word Zipf differences."""

from paired_model_comparison import CONTENT_ZIPF, WORD_COUNT, run_model


if __name__ == "__main__":
    run_model("p2_word_count_and_content_word_zipf_differences", [WORD_COUNT, CONTENT_ZIPF])
