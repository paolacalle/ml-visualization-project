"""Transparent, lexicon-based argument features.

Each feature is a simple count or flag, so the tool can point at the exact words that triggered it.
They are rough proxies, not validated measures of ethos/pathos/logos: validate against hand labels
before making claims about rhetorical appeals.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
URL_RE = re.compile(r"https?://|www\.", re.I)
NUMBER_RE = re.compile(r"\b\d[\d,.]*%?")
LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S", re.M)

# Single-word lexicons (matched against lowercase tokens).
WORDS = {
    "hedges": {"might", "may", "perhaps", "possibly", "probably", "could", "seems", "seem", "likely",
               "arguably", "maybe", "somewhat", "generally", "usually", "often", "sometimes", "suggests",
               "appears", "apparently", "typically"},
    "certainty": {"always", "never", "definitely", "certainly", "clearly", "obviously", "undeniably",
                  "absolutely", "must", "undoubtedly", "proven", "fact"},
    "you": {"you", "your", "you're", "yours", "yourself", "you've", "you'd", "you'll"},
    "i": {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"},
    "we": {"we", "us", "our", "ours", "ourselves", "we're", "we've"},
    "contrast": {"but", "however", "although", "though", "whereas", "nevertheless", "nonetheless"},
    "causal": {"because", "since", "therefore", "thus", "hence", "consequently"},
    "research": {"study", "studies", "research", "evidence", "data", "statistics", "source", "sources",
                 "survey", "paper", "experiment", "according"},
    "politeness": {"please", "thanks", "thank", "sorry", "appreciate"},
    "hostile": {"fuck", "fucking", "shit", "damn", "stupid", "idiot", "idiotic", "ridiculous", "absurd",
                "dumb", "moron", "bullshit", "crap", "nonsense", "ignorant"},
}
# Multi-word phrases (matched against normalized lowercase text).
PHRASES = {
    "hedges": ["i think", "i believe", "i guess", "in my opinion", "imo", "it seems"],
    "examples": ["for example", "for instance", "e.g", "such as", "consider"],
    "experience": ["my experience", "i work", "i worked", "i've seen", "as a former", "when i was",
                   "i used to", "i know someone", "a friend of mine", "personally"],
    "acknowledge": ["good point", "fair point", "i agree", "you're right", "you are right", "i understand",
                    "i see your point", "that's true", "i can see"],
}

# Feature name -> (family, description). Families group features in charts.
FEATURES = {
    "causal_rate":       ("Logos / evidence", "causal connectives (because, therefore...) per 100 words"),
    "number_rate":       ("Logos / evidence", "numbers and percentages per 100 words"),
    "has_link":          ("Logos / evidence", "contains a URL"),
    "research_rate":     ("Logos / evidence", "research words (study, evidence, data...) per 100 words"),
    "examples_rate":     ("Logos / evidence", "example phrases (for example, such as...) per 100 words"),
    "experience_rate":   ("Ethos", "personal-experience phrases (my experience, I work...) per 100 words"),
    "i_rate":            ("Ethos", "first-person singular per 100 words"),
    "emotion":           ("Pathos / tone", "VADER positive + negative share of words"),
    "negativity":        ("Pathos / tone", "VADER negative share of words"),
    "hostile_rate":      ("Pathos / tone", "hostile or profane words per 100 words"),
    "exclaim_rate":      ("Pathos / tone", "exclamation marks per 100 words"),
    "politeness_rate":   ("Politeness / stance", "please, thanks, sorry, appreciate per 100 words"),
    "acknowledge_rate":  ("Politeness / stance", "acknowledgments (good point, I agree...) per 100 words"),
    "hedge_rate":        ("Politeness / stance", "hedges (might, perhaps, I think...) per 100 words"),
    "certainty_rate":    ("Politeness / stance", "certainty words (clearly, always...) per 100 words"),
    "you_rate":          ("Engagement with OP", "second-person (you, your) per 100 words"),
    "we_rate":           ("Engagement with OP", "first-person plural (we, us) per 100 words"),
    "op_overlap":        ("Engagement with OP", "share of reply content words that also appear in the OP"),
    "quotes_op":         ("Engagement with OP", "quotes another message with '>'"),
    "question_rate":     ("Engagement with OP", "question marks per 100 words"),
    "contrast_rate":     ("Structure / style", "contrast words (but, however...) per 100 words"),
    "has_list":          ("Structure / style", "uses a bulleted or numbered list"),
    "paragraph_rate":    ("Structure / style", "paragraphs per 100 words"),
    "words_per_sentence": ("Structure / style", "mean words per sentence"),
    "word_length":       ("Structure / style", "mean characters per word"),
}
BINARY_FEATURES = {"has_link", "quotes_op", "has_list"}

# Short display labels for charts and the tool UI.
LABELS = {
    "causal_rate": "Causal words (because, therefore)",
    "number_rate": "Numbers and percentages",
    "has_link": "Includes a link",
    "research_rate": "Research words (study, evidence)",
    "examples_rate": "Examples (for example, such as)",
    "experience_rate": "Personal experience (I work, ...)",
    "i_rate": "I / me / my",
    "emotion": "Emotional words (VADER pos + neg)",
    "negativity": "Negative words (VADER neg)",
    "hostile_rate": "Hostile or profane words",
    "exclaim_rate": "Exclamation marks",
    "politeness_rate": "Politeness (please, thanks, sorry)",
    "acknowledge_rate": "Acknowledgment (good point, I agree)",
    "hedge_rate": "Hedges (might, perhaps, I think)",
    "certainty_rate": "Certainty (clearly, always, never)",
    "you_rate": "You / your",
    "we_rate": "We / us / our",
    "op_overlap": "Word overlap with the OP",
    "quotes_op": "Quotes the OP ('>')",
    "question_rate": "Question marks",
    "contrast_rate": "Contrast (but, however)",
    "has_list": "Uses a list",
    "paragraph_rate": "Paragraphs (per 100 words)",
    "words_per_sentence": "Words per sentence",
    "word_length": "Word length",
}

_vader = SentimentIntensityAnalyzer()


def _content_words(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2}


def extract(text: str, op_text: str = "", raw_text: str | None = None) -> dict:
    """Compute FEATURES for one argument. `text` should already be cleaned (cmv_data.clean_text);
    `raw_text` (uncleaned) is only used to detect quoting."""
    low = text.lower()
    tokens = TOKEN_RE.findall(low)
    n = max(len(tokens), 1)
    per100 = 100.0 / n
    padded = " " + " ".join(tokens) + " "
    counts = {k: sum(t in vocab for t in tokens) for k, vocab in WORDS.items()}
    for k, phrases in PHRASES.items():
        counts[k] = counts.get(k, 0) + sum(padded.count(f" {p} ") for p in phrases)
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    reply_content = _content_words(tokens)
    op_content = _content_words(TOKEN_RE.findall(op_text.lower()))
    vs = _vader.polarity_scores(text)
    raw = raw_text if raw_text is not None else text
    return {
        "causal_rate": counts["causal"] * per100,
        "number_rate": len(NUMBER_RE.findall(text)) * per100,
        "has_link": int(bool(URL_RE.search(text))),
        "research_rate": counts["research"] * per100,
        "examples_rate": counts["examples"] * per100,
        "experience_rate": counts["experience"] * per100,
        "i_rate": counts["i"] * per100,
        "emotion": vs["pos"] + vs["neg"],
        "negativity": vs["neg"],
        "hostile_rate": counts["hostile"] * per100,
        "exclaim_rate": text.count("!") * per100,
        "politeness_rate": counts["politeness"] * per100,
        "acknowledge_rate": counts["acknowledge"] * per100,
        "hedge_rate": counts["hedges"] * per100,
        "certainty_rate": counts["certainty"] * per100,
        "you_rate": counts["you"] * per100,
        "we_rate": counts["we"] * per100,
        "op_overlap": len(reply_content & op_content) / max(len(reply_content), 1),
        "quotes_op": int(any(l.strip().startswith((">", "&gt;")) for l in raw.splitlines())),
        "question_rate": text.count("?") * per100,
        "contrast_rate": counts["contrast"] * per100,
        "has_list": int(bool(LIST_LINE_RE.search(text))),
        "paragraph_rate": len(paragraphs) * per100,
        "words_per_sentence": len(tokens) / max(len(sentences), 1),
        "word_length": sum(map(len, tokens)) / n,
    }


def extract_frame(args: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Features for every row of cmv_data.load_pairs()'s `args`, joined to the OP text via pair_id."""
    op_text = args.pair_id.map(pairs.set_index("pair_id").op_text)
    rows = [extract(t, o, r) for t, o, r in zip(args.text, op_text, args.text_raw)]
    return pd.DataFrame(rows, index=args.index)
