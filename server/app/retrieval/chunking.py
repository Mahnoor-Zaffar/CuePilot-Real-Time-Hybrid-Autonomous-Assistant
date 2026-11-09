"""Paragraph-aware chunking with overlap.

Chunk boundaries respect blank lines where possible; long paragraphs are split
on sentence boundaries. Deterministic ids make re-ingest idempotent.
"""

from __future__ import annotations

import hashlib
import re

from app.infra.db import Chunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(text):
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            pieces.append(current)
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        current = sentence
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            blocks.append(paragraph)
        else:
            blocks.extend(_split_long(paragraph, max_chars))

    chunks: list[str] = []
    buffer = ""
    for block in blocks:
        candidate = f"{buffer}\n\n{block}".strip() if buffer else block
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        tail = buffer[-overlap:] if buffer and overlap else ""
        buffer = f"{tail}\n\n{block}".strip() if tail else block
        while len(buffer) > max_chars:
            chunks.append(buffer[:max_chars])
            buffer = buffer[max_chars - overlap :] if overlap else buffer[max_chars:]
    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if c.strip()]


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    *,
    max_chars: int = 900,
    overlap: int = 120,
    meta: dict | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, piece in enumerate(split_text(text, max_chars, overlap)):
        digest = hashlib.sha1(f"{doc_id}:{index}:{piece}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=f"{doc_id}#{index}-{digest}",
                doc_id=doc_id,
                title=title,
                content=piece,
                embedding=[],
                meta={**(meta or {}), "chunk_index": index},
            )
        )
    return chunks
