from app.infra.db import Chunk
from app.retrieval.bm25 import BM25Index


def chunk(doc_id: str, title: str, content: str) -> Chunk:
    return Chunk(id=doc_id, doc_id=doc_id, title=title, content=content, embedding=[], meta={})


DOCS = [
    chunk(
        "errors",
        "Gateway error code reference",
        "CUE-E1042 means the model context window was exceeded. CUE-E2041 is a rate limit.",
    ),
    chunk(
        "reconnect",
        "Reconnect policy",
        "Clients reconnect with exponential backoff and full jitter. retry_backoff_cap_ms is 8000.",
    ),
    chunk(
        "retrieval",
        "Hybrid retrieval",
        "BM25 and dense vectors are merged with reciprocal rank fusion at k=60.",
    ),
]


def test_exact_error_code_ranks_first() -> None:
    index = BM25Index(DOCS)
    hits = index.search("CUE-E1042", k=3)
    assert hits
    assert hits[0].chunk.doc_id == "errors"
    assert hits[0].score > 0


def test_identifier_query_beats_prose() -> None:
    index = BM25Index(DOCS)
    hits = index.search("retry_backoff_cap_ms", k=3)
    assert hits[0].chunk.doc_id == "reconnect"


def test_unseen_term_returns_nothing() -> None:
    index = BM25Index(DOCS)
    assert index.search("zzzznonexistentterm", k=3) == []


def test_empty_corpus_is_safe() -> None:
    index = BM25Index([])
    assert len(index) == 0
    assert index.search("anything", k=3) == []


def test_stats() -> None:
    stats = BM25Index(DOCS).stats()
    assert stats["documents"] == 3
    assert stats["vocabulary"] > 10
