"""Code-aware tokenizer.

Embedding models and naive tokenizers both fail on the things engineers actually
search for: `CUE-E1042`, `stream_window_ms`, `pgvector`, `Retry-After`. We emit
the compound token *and* its sub-tokens so BM25 matches both the literal
identifier and its prose form.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_RE = re.compile(r"[_.\-/]+")

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "just", "like", "may", "me", "might", "more", "most", "my", "no", "nor", "not", "of", "on", "or", "our", "out", "over", "same", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "too", "under", "up", "us", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with", "would", "you", "your"]
)


def _variants(token: str) -> list[str]:
    """`getUserID` -> getUserID, get, user, id, getuserid ..."""
    out = [token]
    for part in _CAMEL_RE.split(token):
        out.extend(_SPLIT_RE.split(part))
    # join error-code style identifiers: CUE-E1042 -> cuee1042 / e1042
    joined = "".join(_SPLIT_RE.split(token))
    if joined != token:
        out.append(joined)
        if len(joined) > 2:
            out.append(joined[-4:])
    return out


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        raw = raw.strip(".-")
        if not raw:
            continue
        for variant in _variants(raw):
            token = variant.lower()
            if len(token) < 2 and not token.isdigit():
                continue
            if token in STOPWORDS:
                continue
            tokens.append(token)
    return tokens


def char_ngrams(text: str, n: int = 3, max_ngrams: int = 400) -> list[str]:
    """Character n-grams over the normalised text: typo + morphology robustness."""
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    grams: list[str] = []
    for i in range(min(len(normalised) - n + 1, max_ngrams)):
        gram = normalised[i : i + n]
        if gram.strip():
            grams.append(gram)
    return grams
