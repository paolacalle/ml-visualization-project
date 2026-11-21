"""P0 predicts the winning side from the opening word-count difference alone."""

from paired_model_comparison import WORD_COUNT, run_model


if __name__ == "__main__":
    run_model("p0_word_count_difference", [WORD_COUNT])
