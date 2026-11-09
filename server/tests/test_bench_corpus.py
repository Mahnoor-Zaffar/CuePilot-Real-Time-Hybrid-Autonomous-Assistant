"""Invariants for the evaluation corpus and its labels.

A retrieval benchmark is only as trustworthy as its labels. These tests fail if
the corpus and the query set drift apart: a renamed document, a typo in an
expected id, or a query labelled against a distractor would otherwise show up as
a mysterious recall drop rather than a test failure.
"""

from __future__ import annotations

from bench.corpus import HARD_NEGATIVES, benchmark_documents, primary_ids
from bench.qrels import QUERIES, families


def test_corpus_is_deterministic() -> None:
    first = benchmark_documents()
    second = benchmark_documents()
    assert [doc["doc_id"] for doc in first] == [doc["doc_id"] for doc in second]
    assert [doc["text"] for doc in first] == [doc["text"] for doc in second]


def test_corpus_size_and_unique_ids() -> None:
    docs = benchmark_documents()
    assert len(docs) == 412
    ids = [doc["doc_id"] for doc in docs]
    assert len(ids) == len(set(ids)), "duplicate doc_id in benchmark corpus"
    assert all(doc["title"].strip() for doc in docs)
    assert all(doc["text"].strip() for doc in docs)


def test_primary_documents_are_present() -> None:
    ids = {doc["doc_id"] for doc in benchmark_documents()}
    assert primary_ids() <= ids


def test_hard_negatives_are_never_expected() -> None:
    negative_ids = {doc["doc_id"] for doc in HARD_NEGATIVES}
    for query in QUERIES:
        assert not (set(query.expected) & negative_ids), (
            f"query {query.text!r} is labelled against a hard negative; "
            "hard negatives exist to be confusable, not correct"
        )


def test_every_expected_document_exists() -> None:
    ids = {doc["doc_id"] for doc in benchmark_documents()}
    for query in QUERIES:
        missing = set(query.expected) - ids
        assert not missing, f"query {query.text!r} expects unknown docs {missing}"


def test_queries_are_well_formed() -> None:
    assert QUERIES, "no queries"
    assert all(query.text.strip() for query in QUERIES)
    assert all(query.expected for query in QUERIES)
    assert len({query.text for query in QUERIES}) == len(QUERIES), "duplicate query text"
    assert set(families()) == {"identifier", "paraphrase", "near_miss"}


def test_every_primary_document_is_covered() -> None:
    """A document no query asks about cannot regress, so it is untested."""
    covered = {doc_id for query in QUERIES for doc_id in query.expected}
    uncovered = primary_ids() - covered
    assert not uncovered, f"primary documents with no labelled query: {sorted(uncovered)}"


def test_each_family_has_enough_queries_to_be_meaningful() -> None:
    for family in families():
        count = sum(1 for query in QUERIES if query.family == family)
        assert count >= 15, f"family {family!r} has only {count} queries"
