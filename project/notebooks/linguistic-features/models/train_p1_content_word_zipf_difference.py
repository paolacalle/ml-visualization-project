"""P1 predicts the winning side from the mean content-word Zipf difference alone."""

from paired_model_comparison import CONTENT_ZIPF, run_model


if __name__ == "__main__":
    run_model("p1_content_word_zipf_difference", [CONTENT_ZIPF])
