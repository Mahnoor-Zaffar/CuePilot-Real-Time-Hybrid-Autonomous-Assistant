import pytest
from app.infra.db import Chunk
from app.retrieval.rrf import reciprocal_rank_fusion


def chunk(cid: str) -> Chunk:
    return Chunk(id=cid, doc_id=cid, title=cid, content=cid, embedding=[], meta={})


def test_scores_follow_the_formula() -> None:
    a, b = chunk("a"), chunk("b")
    fused = reciprocal_rank_fusion(
        {"bm25": [(a, 12.0), (b, 8.0)], "dense": [(b, 0.91), (a, 0.72)]}, k=60
    )
    by_id = {hit.chunk.id: hit for hit in fused}
    # a: rank1 lexical, rank2 dense
    assert by_id["a"].score == pytest.approx(1 / 61 + 1 / 62, abs=1e-7)
    assert by_id["b"].score == pytest.approx(1 / 62 + 1 / 61, abs=1e-7)
    assert by_id["a"].ranks == {"bm25": 1, "dense": 2}


def test_deduplicates_by_chunk_id() -> None:
    a = chunk("a")
    fused = reciprocal_rank_fusion({"bm25": [(a, 1.0)], "dense": [(a, 0.9)]})
    assert len(fused) == 1
    assert fused[0].contributions["bm25"] == pytest.approx(1 / 61)


def test_weights_shift_ranking() -> None:
    a, b = chunk("a"), chunk("b")
    lexical_heavy = reciprocal_rank_fusion(
        {"bm25": [(a, 9.0), (b, 8.0)], "dense": [(b, 0.9), (a, 0.1)]},
        weights={"bm25": 3.0, "dense": 1.0},
    )
    assert lexical_heavy[0].chunk.id == "a"

    dense_heavy = reciprocal_rank_fusion(
        {"bm25": [(a, 9.0), (b, 8.0)], "dense": [(b, 0.9), (a, 0.1)]},
        weights={"bm25": 0.1, "dense": 3.0},
    )
    assert dense_heavy[0].chunk.id == "b"


def test_top_k_truncates_after_fusion() -> None:
    lists = {"bm25": [(chunk(str(i)), 1.0) for i in range(5)]}
    fused = reciprocal_rank_fusion(lists, top_k=2)
    assert [hit.chunk.id for hit in fused] == ["0", "1"]


def test_public_payload_explains_provenance() -> None:
    a = chunk("a")
    fused = reciprocal_rank_fusion({"bm25": [(a, 4.2)], "dense": [(a, 0.5)]})
    payload = fused[0].to_public()
    assert payload["explain"]["ranks"] == {"bm25": 1, "dense": 1}
    assert payload["explain"]["raw_scores"]["bm25"] == pytest.approx(4.2)
