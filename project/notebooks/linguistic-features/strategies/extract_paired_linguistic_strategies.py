"""Extract four exploratory strategies from training-pair openings."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "topics"))
from discover_op_topics import BUILDER, MODEL as SBERT, OUT as TOPIC_RUNS, REVISION as SBERT_REVISION, paired
from text_preprocessing import WORDS, minimal_text

OUT = paired.DATA / "paired-linguistic-strategies"
TOPIC_RUN = "20261120T221529517368Z"
SENTIMENT = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_REVISION = "3216a57"
SENTIMENT_CHUNK = 192
TOPICS = Path(__file__).resolve().parents[1] / "topics" / f"draft_topic_labels_{TOPIC_RUN}.csv"
COURTESY = ("Gratitude", "Deference", "Apologizing", "Please")
MAIN_FEATURES = {
    "courtesy_count": "Courtesy markers",
    "hedges_per_100_words": "Hedges per 100 words",
    "net_sentiment": "Net positive sentiment",
    "op_cosine": "OP–reply semantic similarity",
}
HEDGE_WORDS = {"perhaps", "maybe", "probably", "possibly", "apparently", "arguably", "seem", "appear", "suggest"}
HEDGE_PHRASES = (
    "i think", "i believe", "i suspect", "i suppose", "i guess", "i would argue",
    "it seems", "it appears", "in my opinion", "as far as i know", "to some extent",
)
NORMALIZE_MENTION = re.compile(r"(?<!\w)(?:@\w+|/?u/[\w-]+)", re.I)


def load_text():
    assignments, audit = paired.load_training_pairs()
    builder = paired.import_script("strategy_branch_builder", BUILDER)
    manifest = builder.manifest(builder.TEXT)
    ids = set(assignments["a_comment_id"]) | set(assignments["b_comment_id"])
    replies = pd.DataFrame([
        {"comment_id": row["comment_id"],
         "opening_text": minimal_text("\n".join(builder.opening_text_lines(row["body"])))}
        for row in builder.text_rows(manifest["files"], ids, "train")
    ]).set_index("comment_id").sort_index()
    if not replies.index.is_unique or set(replies.index) != ids:
        raise ValueError("The training Parquet must supply one opening per comment ID.")
    audit["raw_archive_sha256"] = manifest["raw_archive"]["sha256"]
    return assignments, replies, audit


def hedge_spans(doc):
    phrases = {tuple(phrase.split()) for phrase in HEDGE_PHRASES}
    length = max(map(len, phrases))
    tokens = [token.lower_ for token in doc]
    matches = []
    i = 0
    while i < len(doc):
        word = doc[i]
        phrase_len = next((n for n in range(length, 1, -1) if tuple(tokens[i:i+n]) in phrases), 0)
        if phrase_len:
            matches.append(doc[i:i+phrase_len].text)
            i += phrase_len
            continue
        if ((word.lower_ in {"may", "might", "could"} and word.tag_ == "MD")
                or (word.lemma_.lower() in HEDGE_WORDS and word.pos_ in {"ADV", "VERB", "AUX"})):
            matches.append(word.text)
        i += 1
    return matches


def lexical_features(replies):
    import spacy
    from convokit.politenessStrategies import PolitenessStrategies

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    extractor = PolitenessStrategies(strategy_collection="politeness_api")
    rows = []
    for doc in nlp.pipe(replies["opening_text"], batch_size=32):
        if not WORDS.search(doc.text):
            rows.append({})
            continue
        utterance = extractor.transform_utterance(doc.text, spacy_nlp=lambda _: doc)
        strategies = utterance.meta["politeness_strategies"]
        binary = {
            key.removeprefix("feature_politeness_==").removesuffix("==").replace("_", " "): int(value)
            for key, value in strategies.items()
        }
        if not {*COURTESY, "Please start"} <= binary.keys():
            raise ValueError(f"Unrecognized ConvoKit strategy keys: {list(strategies)}")
        courtesy = sum(binary[name] for name in COURTESY[:-1]) + int(binary["Please"] or binary["Please start"])
        hedges = hedge_spans(doc)
        word_count = len(WORDS.findall(doc.text))
        rows.append({
            "analyzed_word_count": word_count, "hedge_count": len(hedges), "hedges_per_100_words": 100 * len(hedges) / word_count,
            "hedge_examples": ", ".join(dict.fromkeys(hedges)), "courtesy_count": courtesy,
            "courtesy_examples": ", ".join(name for name in COURTESY
                                           if binary[name] or (name == "Please" and binary["Please start"])),
            **{f"strategy::{key}": value for key, value in binary.items()},
        })
    return pd.DataFrame(rows, index=replies.index)


def semantic_vectors(replies, cache_folder):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT, revision=SBERT_REVISION, cache_folder=cache_folder,
                                local_files_only=True, device="cpu")
    width = model.max_seq_length - 16
    chunks, owners, weights = [], [], []
    for i, text in enumerate(replies["opening_text"]):
        offsets = model.tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
                                  verbose=False)["offset_mapping"]
        for start in range(0, len(offsets), width):
            segment = offsets[start:start + width]
            chunks.append(text[segment[0][0]:segment[-1][1]])
            owners.append(i)
            weights.append(len(segment))
    encoded = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    pooled = np.zeros((len(replies), encoded.shape[1]), dtype=encoded.dtype)
    np.add.at(pooled, owners, encoded * np.asarray(weights, dtype=np.float32)[:, None])
    return normalize(pooled), len(chunks)


def sentiment_scores(replies, cache_folder):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = {"revision": SENTIMENT_REVISION, "cache_dir": cache_folder, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT, **settings)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT, **settings).eval()
    # Encode each overflow chunk, including the tail of long Reddit openings.
    texts = [NORMALIZE_MENTION.sub("@user", text) for text in replies["opening_text"]]
    encoded = tokenizer(texts, max_length=SENTIMENT_CHUNK, truncation=True,
                        return_overflowing_tokens=True, return_attention_mask=False)
    ids = encoded["input_ids"]
    owners = np.asarray(encoded["overflow_to_sample_mapping"])
    order = sorted(range(len(ids)), key=lambda index: len(ids[index]))
    special = tokenizer.num_special_tokens_to_add(pair=False)
    weights = np.asarray([max(len(item) - special, 0) for item in ids])
    sums = np.zeros((len(replies), model.config.num_labels), dtype=float)
    token_counts = np.bincount(owners, weights=weights, minlength=len(replies))
    start = 0
    with torch.inference_mode():
        while start < len(order):
            batch = order[start:start + 32]
            prepared = tokenizer.pad({"input_ids": [ids[index] for index in batch]},
                                     padding=True, return_attention_mask=True, return_tensors="pt")
            probabilities = model(**prepared).logits.softmax(dim=-1).numpy()
            np.add.at(sums, owners[batch], probabilities * weights[batch, None])
            if start % 1024 == 0:
                print(f"Sentiment chunks {start:,}/{len(order):,}", flush=True)
            start += len(batch)
    avg = np.divide(sums, token_counts[:, None], out=np.full_like(sums, np.nan), where=token_counts[:, None] > 0)
    labels = {label.lower(): int(index) for index, label in model.config.id2label.items()}
    result = pd.DataFrame({f"sentiment_{label}": avg[:, labels[label]] for label in ("negative", "neutral", "positive")},
                          index=replies.index)
    result["net_sentiment"] = result["sentiment_positive"] - result["sentiment_negative"]
    return result, len(order)


def topic_context(assignments, replies, reply_vectors):
    source = TOPIC_RUNS / TOPIC_RUN
    report = json.loads((source / "results.json").read_text())
    op = pd.read_parquet(source / "op_topic_assignments.parquet", columns=["post_id", "topic_id", "title"])
    vectors = np.load(source / "op_embeddings.npy")
    if (report["split"] != "train" or report["embedding_revision"] != SBERT_REVISION
            or len(op) != len(vectors) or not op["post_id"].is_unique
            or report["embedding_sha256"] != hashlib.sha256(vectors.tobytes()).hexdigest()):
        raise ValueError("Unexpected OP embedding provenance or row order.")
    mapping = pd.read_csv(TOPICS, usecols=["topic_id", "broad_topic"])
    posts = op.merge(mapping, on="topic_id", validate="many_to_one")
    if set(posts["post_id"]) != set(assignments["post_id"]):
        raise ValueError("Every matched training discussion needs its stored OP topic and embedding.")
    post_idx = pd.Index(op["post_id"]).get_indexer(assignments["post_id"])

    scores = assignments.merge(posts, on="post_id", validate="many_to_one", sort=False)
    pd.testing.assert_series_equal(scores["pair_id"], assignments["pair_id"])
    for side in ("a", "b"):
        reply_idx = replies.index.get_indexer(scores[f"{side}_comment_id"])
        vector = reply_vectors[reply_idx]
        cosine = np.einsum("ij,ij->i", vector, vectors[post_idx])
        cosine[(np.linalg.norm(vector, axis=1) == 0) | (np.linalg.norm(vectors[post_idx], axis=1) == 0)] = np.nan
        scores[f"{side}_op_cosine"] = np.clip(cosine, -1, 1)
    return scores, report["source_archive_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-folder", type=Path)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(6)
    assignments, replies, audit = load_text()
    print(f"Scoring {len(replies):,} training openings.", flush=True)
    lexical = lexical_features(replies)
    print("Politeness and hedge extraction complete.", flush=True)
    vectors, semantic_chunks = semantic_vectors(replies, args.cache_folder)
    scores, topic_archive = topic_context(assignments, replies, vectors)
    if topic_archive != audit["raw_archive_sha256"]:
        raise ValueError("Reply and OP text must come from the same source archive.")
    print("Scoring CardiffNLP sentiment locally.", flush=True)
    sentiment, sentiment_chunks = sentiment_scores(replies, args.cache_folder)
    openings = replies.join(lexical).join(sentiment)
    for side in ("a", "b"):
        values = openings.drop(columns="opening_text").add_prefix(f"{side}_")
        scores = scores.join(values, on=f"{side}_comment_id", validate="many_to_one")
    scores["within_50_words"] = (scores["a_word_count"] - scores["b_word_count"]).abs().le(50)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with TemporaryDirectory(dir=OUT) as temporary:
        stage = Path(temporary)
        scores.to_parquet(stage / "pair_strategy_scores.parquet", index=False)
        openings.to_parquet(stage / "opening_strategy_scores.parquet")
        (stage / "results.json").write_text(json.dumps({
            "split": "train", "pairs": len(scores), "discussions": scores["post_id"].nunique(),
            "unique_openings": len(openings), "topic_run": TOPIC_RUN, **audit,
            "semantic_model": SBERT, "semantic_revision": SBERT_REVISION, "semantic_chunks": semantic_chunks,
            "sentiment_model": SENTIMENT, "sentiment_revision": SENTIMENT_REVISION,
            "sentiment_chunks": sentiment_chunks, "sentiment_chunk_limit": SENTIMENT_CHUNK,
            "versions": {name: version(name) for name in ("convokit", "spacy", "en_core_web_sm", "sentence-transformers", "transformers")},
        }, indent=2) + "\n")
        stage.rename(destination)
    print(f"Saved linguistic strategy scores to {destination}", flush=True)


if __name__ == "__main__":
    main()
