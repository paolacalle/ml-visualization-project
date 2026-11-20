"""Add the average letters-per-word difference to the original P2 feature set."""

from paired_model_comparison import AVERAGE_WORD_LENGTH, CONTENT_ZIPF, WORD_COUNT, run_model


if __name__ == "__main__":
    run_model(
        "p2_plus_average_word_length",
        [WORD_COUNT, CONTENT_ZIPF, AVERAGE_WORD_LENGTH],
    )
