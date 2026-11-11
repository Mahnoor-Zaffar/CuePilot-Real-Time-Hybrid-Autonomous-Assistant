from __future__ import annotations

from app.main import create_app
from fastapi.testclient import TestClient
from tests.conftest import drain_ws, make_settings


def test_root_and_health(client: TestClient) -> None:
    assert client.get("/").json()["service"] == "cuepilot-gateway"
    health = client.get("/api/health").json()
    assert health["status"] == "degraded"  # no Postgres/Redis in CI
    assert health["backends"]["vector_store"] == "memory"
    assert health["kb"]["chunks"] > 0
    assert health["provider"]["mode"] == "mock"
    assert health["limits"]["enabled"] is True
    assert health["limits"]["algorithm"] == "sliding_window"
    assert health["budget"]["enabled"] is True
    assert health["budget"]["token_budget"] == "post_hoc_from_completed_usage"
    assert health["durability"]["degraded"] is False
    assert health["journal"]["backend"] == "none"


def test_search_endpoint(client: TestClient) -> None:
    response = client.get("/api/kb/search", params={"q": "CUE-E1042", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"]
    assert payload["hits"][0]["doc_id"] == "kb-error-codes"
    assert payload["timings_ms"]["bm25"] >= 0


def test_ingest_bumps_version_and_invalidates_cache(client: TestClient) -> None:
    first = client.get("/api/kb/search", params={"q": "cobalt deployment window"}).json()
    assert first["cache_hit"] is False

    ingest = client.post(
        "/api/kb/ingest",
        json={
            "documents": [
                {
                    "doc_id": "cobalt-runbook",
                    "title": "Cobalt deployment window",
                    "text": "The cobalt deployment window opens every Thursday at 02:00 UTC.",
                }
            ]
        },
    )
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["ok"] is True
    assert body["kb_version"] > first["kb_version"]

    after = client.get("/api/kb/search", params={"q": "cobalt deployment window"}).json()
    assert after["cache_hit"] is False
    assert after["hits"][0]["doc_id"] == "cobalt-runbook"


def test_ingest_requires_documents(client: TestClient) -> None:
    assert client.post("/api/kb/ingest", json={}).status_code == 422


def test_metrics_and_traces_after_a_turn(client: TestClient) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({"type": "hello"})
        ws.receive_json()
        ws.send_json({"type": "prompt", "id": "t1", "text": "What is the reconnect policy?"})
        drain_ws(ws, until="done")

    metrics = client.get("/api/metrics").json()
    assert metrics["counters"]["turns_total"] >= 1
    assert metrics["histograms"]["ttft_ms"]["count"] >= 1
    assert metrics["backends"]["provider"] == "mock"
    assert metrics["aggregation"]["instances"] == 1
    assert metrics["aggregation"]["merged"] is False
    assert metrics["aggregation"]["quantile_method"] == "cumulative_buckets"
    assert metrics["aggregation"]["bucket_bounds_ms"]
    assert "samples" not in metrics["histograms"]["ttft_ms"]
    assert metrics["histograms"]["ttft_ms"]["buckets"]["+Inf"] >= 1

    # The Prometheus endpoint serialises the same merged snapshot as text.
    prometheus = client.get("/metrics")
    assert prometheus.status_code == 200
    assert prometheus.headers["content-type"].startswith("text/plain")
    assert "cuepilot_turns_total" in prometheus.text
    assert 'cuepilot_ttft_ms_bucket{instance=' in prometheus.text
    assert 'le="+Inf"' in prometheus.text

    traces = client.get("/api/traces").json()["traces"]
    assert traces
    turn = next(trace for trace in traces if trace["name"] == "turn")
    assert turn["duration_ms"] is not None
    assert turn["attributes"]["ttft_ms"] is not None
    child_names = {child["name"] for child in turn["children"]}
    assert "retrieval" in child_names
    nested = {child["name"] for node in turn["children"] for child in node["children"]}
    assert {"retrieval.bm25", "retrieval.dense"} <= nested | child_names

    detail = client.get(f"/api/traces/{turn['trace_id']}")
    assert detail.status_code == 200


def test_auth_is_enforced_when_configured() -> None:
    app = create_app(make_settings(api_token="secret"))
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/kb/search", params={"q": "x"}).status_code == 401
        assert (
            client.get(
                "/api/kb/search",
                params={"q": "x"},
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 200
        )
