"""Shared minimal Markdown cleaning and word tokenization for opening replies."""

from html import unescape
import re


WORDS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
LINK = re.compile(r"!?\[([^\]\n]+)\]\(<?https?://[^)\n]+\)", re.I)
URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\])]+", re.I)
BLOCK_FORMAT = re.compile(r"^\s*(?:\x23{1,6}\s+|[-+*]\s+|\d+[.)]\s+|\x60{3,})")

def minimal_text(body):
    """Remove formatting and quoted material while preserving readable link labels."""
    text = unescape(unescape(body))
    lines = [
        BLOCK_FORMAT.sub("", line) for line in text.splitlines()
        if not line.lstrip().startswith(">")
    ]
    text = LINK.sub(r"\1", "\n".join(lines))
    text = URL.sub(lambda match: re.search(r"[.?!,;:]*$", match[0])[0], text)
    return re.sub(r"[`*_~]+", "", text)
