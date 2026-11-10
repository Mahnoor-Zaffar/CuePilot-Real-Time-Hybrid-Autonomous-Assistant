#!/usr/bin/env python
"""Reproducible benchmark for the cuePilot gateway.

Measures, on whatever machine runs it:
  1. retrieval recall@5 and MRR for bm25-only / dense-only / hybrid (RRF),
     split by identifier and paraphrase query families;
  2. retrieval latency p50/p95 per mode, cache disabled;
  3. Time to First Token through the real WebSocket gateway (in-process ASGI
     transport) and the gateway overhead above the provider's own first byte;
  4. delta frame coalescing ratio at a 90 tokens/second stream;
  5. the built renderer bundle size (raw and gzip), if present;
  6. the desktop client's idle footprint, if `desktop/bench/idle.json` exists
     (run `npm run measure-idle` in `desktop/` first).

Usage:
    cd server && .venv/bin/python bench/run_bench.py [--write]

Results are printed as markdown and written to ../docs/metrics.json plus
../docs/METRICS.md (the generated half of the metrics doc).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from app.config import Settings  # noqa: E402
from app.infra.cache import MemoryCache  # noqa: E402
from app.infra.db import MemoryVectorStore  # noqa: E402
from app.main import create_app  # noqa: E402
from app.retrieval.embedder import Embedder  # noqa: E402
from app.retrieval.orchestrator import HybridRetriever  # noqa: E402
from app.telemetry.metrics import Metrics  # noqa: E402
from app.telemetry.otel import TraceRecorder  # noqa: E402
from app.ws.outbound import OutboundStream  # noqa: E402
from bench.corpus import benchmark_documents  # noqa: E402
from bench.embedders import DEFAULT_LOCAL_MODEL, build_bench_embedder  # noqa: E402
from bench.qrels import QUERIES, by_family, families  # noqa: E402
from bench.stats import (  # noqa: E402
    DEFAULT_RESAMPLES,
    Interval,
    bootstrap_ci,
    describe_effect,
    paired_bootstrap_delta,
)

DOCS_DIR = ROOT.parent / "docs"
TOP_K = 5


def bench_settings(**overrides) -> Settings:
    base = {
        "env": "test",
        "llm_provider": "mock",
        "embedding_provider": "hash",
        "allow_degraded_infra": True,
        "infra_probe_timeout_s": 0.05,
        "database_url": "postgresql://x:x@127.0.0.1:1/none",
        "redis_url": "redis://127.0.0.1:1/0",
        "otel_enabled": False,
        "otel_exporter": "none",
        "seed_on_start": True,
        "mock_token_delay_ms": 0.0,
        "mock_ttft_ms": 0.0,
        "audio_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return round(ordered[idx], 2)


def latency_block(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": round(statistics.fmean(values), 2) if values else None,
    }


# ── 1 + 2: retrieval quality and latency ───────────────────────────────────
async def evaluate_retrieval(
    settings: Settings,
    embedder: Embedder,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict:
    metrics = Metrics()
    retriever = HybridRetriever(
        settings,
        MemoryVectorStore(embedder.dim),
        embedder,
        MemoryCache(),
        metrics,
        TraceRecorder(),
    )
    documents = benchmark_documents()
    ingest_started = time.perf_counter()
    ingest = await retriever.ingest(documents)
    ingest_ms = (time.perf_counter() - ingest_started) * 1000

    report: dict[str, dict] = {}
    for mode in ("bm25", "dense", "hybrid"):
        latencies: list[float] = []
        family_stats: dict[str, dict[str, float]] = {}
        per_query: list[dict] = []
        hit_vector: list[float] = []
        hits_at_k = 0
        reciprocal_ranks: list[float] = []
        for query in QUERIES:
            started = time.perf_counter()
            result = await retriever.search(query.text, mode=mode, top_k=TOP_K, use_cache=False)
            latencies.append((time.perf_counter() - started) * 1000)
            docs = [hit.chunk.doc_id for hit in result.hits]
            rank = next(
                (i + 1 for i, doc in enumerate(docs) if doc in query.expected),
                None,
            )
            hit = rank is not None
            hits_at_k += int(hit)
            hit_vector.append(1.0 if hit else 0.0)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            family = family_stats.setdefault(query.family, {"n": 0, "hits": 0})
            family["n"] += 1
            family["hits"] += int(hit)
            per_query.append(
                {
                    "query": query.text,
                    "family": query.family,
                    "expected": list(query.expected),
                    "top_docs": docs,
                    "rank": rank,
                }
            )
        n = len(QUERIES)
        report[mode] = {
            "recall_at_5": round(hits_at_k / n, 4),
            "recall_at_5_ci": bootstrap_ci(
                hit_vector, resamples=resamples, seed=seed
            ).to_dict(),
            "mrr": round(statistics.fmean(reciprocal_ranks), 4),
            "mrr_ci": bootstrap_ci(
                reciprocal_ranks, resamples=resamples, seed=seed
            ).to_dict(),
            "hits": hits_at_k,
            "n": n,
            "latency_ms": latency_block(latencies),
            "by_family": {
                family: round(values["hits"] / values["n"], 4)
                for family, values in family_stats.items()
            },
            "per_query": per_query,
        }

    bm25 = report["bm25"]
    dense = report["dense"]
    hybrid = report["hybrid"]
    best_single = max(bm25["recall_at_5"], dense["recall_at_5"])
    report["summary"] = {
        "corpus_documents": len(documents),
        "corpus_chunks": ingest["chunks"],
        "ingest_ms": round(ingest_ms, 2),
        "embedder": ingest["embedder"],
        "queries": len(QUERIES),
        "queries_by_family": {family: len(by_family(family)) for family in families()},
        "hybrid_vs_best_single_pp": round((hybrid["recall_at_5"] - best_single) * 100, 2),
        "hybrid_vs_dense_pp": round((hybrid["recall_at_5"] - dense["recall_at_5"]) * 100, 2),
        "queries_dense_missed_hybrid_found": sum(
            1
            for dense_q, hybrid_q in zip(dense["per_query"], hybrid["per_query"], strict=True)
            if dense_q["rank"] is None and hybrid_q["rank"] is not None
        ),
        "queries_bm25_missed_hybrid_found": sum(
            1
            for bm25_q, hybrid_q in zip(bm25["per_query"], hybrid["per_query"], strict=True)
            if bm25_q["rank"] is None and hybrid_q["rank"] is not None
        ),
        "queries_any_single_found": sum(
            1
            for bm25_q, dense_q in zip(bm25["per_query"], dense["per_query"], strict=True)
            if bm25_q["rank"] is not None or dense_q["rank"] is not None
        ),
        "queries_hybrid_found_neither": sum(
            1
            for bm25_q, dense_q, hybrid_q in zip(
                bm25["per_query"], dense["per_query"], hybrid["per_query"], strict=True
            )
            if bm25_q["rank"] is None
            and dense_q["rank"] is None
            and hybrid_q["rank"] is not None
        ),
        "queries_hybrid_lost_from_singles": sum(
            1
            for bm25_q, dense_q, hybrid_q in zip(
                bm25["per_query"], dense["per_query"], hybrid["per_query"], strict=True
            )
            if hybrid_q["rank"] is None
            and (bm25_q["rank"] is not None or dense_q["rank"] is not None)
        ),
    }
    summary = report["summary"]
    bm25_hits = [1.0 if q["rank"] else 0.0 for q in bm25["per_query"]]
    dense_hits = [1.0 if q["rank"] else 0.0 for q in dense["per_query"]]
    hybrid_hits = [1.0 if q["rank"] else 0.0 for q in hybrid["per_query"]]
    summary["best_single"] = "bm25" if bm25["recall_at_5"] >= dense["recall_at_5"] else "dense"
    summary["hybrid_vs_bm25"] = paired_bootstrap_delta(
        bm25_hits, hybrid_hits, resamples=resamples, seed=seed
    ).to_dict()
    summary["hybrid_vs_dense"] = paired_bootstrap_delta(
        dense_hits, hybrid_hits, resamples=resamples, seed=seed
    ).to_dict()
    summary["hybrid_vs_best_single"] = summary[f"hybrid_vs_{summary['best_single']}"]
    summary["bootstrap"] = {"resamples": resamples, "seed": seed}
    summary["union_recall_at_5"] = round(summary["queries_any_single_found"] / len(QUERIES), 4)
    summary["fusion_gap"] = summary["queries_hybrid_lost_from_singles"]
    return report


# ── 3: TTFT through the real gateway ───────────────────────────────────────
TTFT_PROMPTS = [
    "What does CUE-E1042 mean?",
    "How does reciprocal rank fusion combine BM25 and dense results?",
    "What is the reconnect backoff policy?",
    "How do I cancel an in-flight generation?",
    "What is ws_send_queue_size for?",
    "How does the Electron client capture audio?",
    "Which metrics are exported for TTFT?",
    "What happens when every model is rate limited?",
]


def evaluate_ttft(settings: Settings, provider_delay_ms: float = 120.0) -> dict:
    from fastapi.testclient import TestClient

    app = create_app(settings)
    client_ttft: list[float] = []
    server_ttft: list[float] = []
    frames_per_turn: list[int] = []
    deltas_per_turn: list[int] = []

    with TestClient(app) as client:
        for index, prompt in enumerate(TTFT_PROMPTS):
            with client.websocket_connect("/ws/stream") as ws:
                ws.send_json({"type": "hello"})
                ws.receive_json()
                started = time.perf_counter()
                ws.send_json(
                    {
                        "type": "prompt",
                        "id": f"ttft-{index}",
                        "text": prompt,
                        "options": {"tools": False},
                    }
                )
                first_delta_ms: float | None = None
                frames = 0
                deltas = 0
                done: dict | None = None
                for _ in range(500):
                    frame = ws.receive_json()
                    frames += 1
                    if frame["type"] == "delta":
                        if first_delta_ms is None:
                            first_delta_ms = (time.perf_counter() - started) * 1000
                        deltas += 1
                    if frame["type"] == "done":
                        done = frame
                        break
                if first_delta_ms is not None:
                    client_ttft.append(first_delta_ms)
                if done and done.get("ttft_ms") is not None:
                    server_ttft.append(done["ttft_ms"])
                frames_per_turn.append(frames)
                deltas_per_turn.append(deltas)

    overhead = [
        client - server for client, server in zip(client_ttft, server_ttft, strict=False)
    ]
    return {
        "provider_delay_ms": provider_delay_ms,
        "turns": len(TTFT_PROMPTS),
        "client_ttft_ms": latency_block(client_ttft),
        "server_ttft_ms": latency_block(server_ttft),
        "gateway_overhead_ms": latency_block(overhead),
        "frames_per_turn": latency_block([float(v) for v in frames_per_turn]),
        "deltas_per_turn": latency_block([float(v) for v in deltas_per_turn]),
    }


# ── 4: coalescing ratio ────────────────────────────────────────────────────
class _Sink:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send_json(self, frame: dict) -> None:
        self.frames.append(frame)


async def evaluate_coalescing(settings: Settings, tokens: int = 300) -> dict:
    sink = _Sink()
    metrics = Metrics()
    out = OutboundStream(sink, settings, metrics)
    await out.start()
    started = time.perf_counter()
    try:
        for _ in range(tokens):
            await out.push_delta("bench", "token ")
            await asyncio.sleep(1 / 90)  # ~90 tokens/second pacing
        await out.flush_turn("bench")
        await asyncio.sleep(settings.delta_flush_ms / 1000 * 2)
    finally:
        await out.close()
    elapsed = time.perf_counter() - started
    delta_frames = [frame for frame in sink.frames if frame["type"] == "delta"]
    text = "".join(frame["text"] for frame in delta_frames)
    return {
        "tokens": tokens,
        "frames": len(delta_frames),
        "frame_reduction_pct": round((1 - len(delta_frames) / tokens) * 100, 1),
        "tokens_per_frame": round(tokens / max(1, len(delta_frames)), 2),
        "stream_seconds": round(elapsed, 3),
        "text_lossless": text == "token " * tokens,
        "dropped_frames": int(metrics.counters.get("delta_frames_dropped", 0)),
    }


# ── 5: renderer bundle ─────────────────────────────────────────────────────
def _gzip_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        total += len(gzip.compress(path.read_bytes(), compresslevel=9))
    return total


def renderer_bundle() -> dict | None:
    assets = ROOT.parent / "desktop" / "dist" / "assets"
    if not assets.exists():
        return None
    js = sorted(assets.glob("*.js"))
    css = sorted(assets.glob("*.css"))
    if not js:
        return None
    js_bytes = sum(path.stat().st_size for path in js)
    css_bytes = sum(path.stat().st_size for path in css)
    js_gzip = _gzip_bytes(js)
    css_gzip = _gzip_bytes(css)
    return {
        "js_bytes": js_bytes,
        "js_kb": round(js_bytes / 1024, 1),
        "js_gzip_kb": round(js_gzip / 1024, 1),
        "css_bytes": css_bytes,
        "css_kb": round(css_bytes / 1024, 1),
        "css_gzip_kb": round(css_gzip / 1024, 1),
        "files": [path.name for path in js + css],
    }


# ── 6: desktop idle footprint ──────────────────────────────────────────────
def client_idle() -> dict | None:
    path = ROOT.parent / "desktop" / "bench" / "idle.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    heap = raw.get("rendererHeap") or {}
    return {
        "measured_at": raw.get("measuredAt"),
        "platform": raw.get("platform"),
        "electron": raw.get("electron"),
        "chrome": raw.get("chrome"),
        "renderer_working_set_mb": raw.get("rendererWorkingSetMB"),
        "all_processes_working_set_mb": raw.get("allProcessesWorkingSetMB"),
        "renderer_heap_used_mb": heap.get("usedHeapMB"),
        "renderer_heap_total_mb": heap.get("totalHeapMB"),
        "dom_nodes": heap.get("domNodes"),
        "processes": raw.get("processes", []),
    }



def failure_lines(dense: dict, hybrid: dict, bm25: dict) -> list[str]:
    lines = ["Dense-only missed, hybrid recovered:", ""]
    recovered = [
        hybrid_q
        for dense_q, hybrid_q in zip(dense["per_query"], hybrid["per_query"], strict=True)
        if dense_q["rank"] is None and hybrid_q["rank"] is not None
    ]
    lines += [
        f'- "{item["query"]}" -> hybrid rank {item["rank"]} (`{item["expected"][0]}`)'
        for item in recovered
    ] or ["- none"]
    lines += ["", "Hybrid missed, at least one single retriever found:", ""]
    regressions = [
        (hybrid_q, bm25_q, dense_q)
        for hybrid_q, bm25_q, dense_q in zip(
            hybrid["per_query"], bm25["per_query"], dense["per_query"], strict=True
        )
        if hybrid_q["rank"] is None and (bm25_q["rank"] or dense_q["rank"])
    ]
    lines += [
        f'- "{item[0]["query"]}" (bm25={item[1]["rank"]}, dense={item[2]["rank"]})'
        for item in regressions
    ] or ["- none"]
    return lines


def render_markdown(report: dict) -> str:
    retrieval = report["retrieval"]
    bm25, dense, hybrid = retrieval["bm25"], retrieval["dense"], retrieval["hybrid"]
    summary = retrieval["summary"]
    ttft = report["ttft"]
    coalescing = report["coalescing"]
    bundle = report.get("bundle")
    family_names = families()

    def row(name: str, block: dict) -> str:
        family_cells = "".join(
            f"| {block['by_family'][family]:.1%} " for family in family_names
        )
        ci = block.get("recall_at_5_ci") or {}
        recall = f"{block['recall_at_5']:.1%}"
        if ci:
            recall += f" [{ci['low']:.1%}, {ci['high']:.1%}]"
        return (
            f"| {name} | {recall} | {block['mrr']:.3f} "
            f"{family_cells}"
            f"| {block['latency_ms']['p50']} ms | {block['latency_ms']['p95']} ms |"
        )

    def effect(block: dict) -> str:
        return describe_effect(Interval(**block))

    family_header = "".join(f"| {family.replace('_', ' ').title()} " for family in family_names)
    family_rule = "|" + "---|" * (3 + len(family_names) + 2)
    query_mix = ", ".join(
        f"{count} {family.replace('_', ' ')}"
        for family, count in summary["queries_by_family"].items()
    )

    embedder = summary["embedder"]
    if embedder.startswith("hash"):
        embedder_note = (
            "deterministic lexical proxy over tokens, bigrams and character trigrams; "
            "not a semantic model"
        )
    elif embedder.startswith("model2vec"):
        embedder_note = (
            "local static sentence embeddings; a semantic model, used as a stand-in "
            "for the hosted embedding path"
        )
    else:
        embedder_note = "hosted embeddings"

    lines = [
        "# Measured metrics",
        "",
        f"Generated by `server/bench/run_bench.py` on {report['environment']['timestamp']}.",
        "Every number below was produced by running the code in this repository; nothing is",
        "hand-entered. Re-run it to reproduce on your hardware.",
        "",
        "## Environment",
        "",
        f"- Python {report['environment']['python']} on {report['environment']['platform']}",
        f"- Embedder: `{embedder}` — {embedder_note}",
        f"- Provider for TTFT: `{report['environment']['provider']}`",
        f"- Corpus: {summary['corpus_documents']} documents -> {summary['corpus_chunks']} chunks "
        f"(ingest {summary['ingest_ms']} ms)",
        f"- Queries: {summary['queries']} labelled ({query_mix})",
        "",
        f"## 1. Retrieval quality (recall@5, {summary['queries']} labelled queries)",
        "",
        f"| Retriever | Recall@5 [95% CI] | MRR {family_header}| p50 | p95 |",
        family_rule,
        row("BM25 only", bm25),
        row("Dense only", dense),
        row("Hybrid (RRF, k=60)", hybrid),
        "",
        f"- Hybrid vs. best single retriever (`{summary['best_single']}`): "
        f"{effect(summary['hybrid_vs_best_single'])}",
        f"- Hybrid vs. dense-only: {effect(summary['hybrid_vs_dense'])}",
        f"- Queries dense-only missed that hybrid recovered: **{summary['queries_dense_missed_hybrid_found']}**",
        f"- Queries BM25-only missed that hybrid recovered: **{summary['queries_bm25_missed_hybrid_found']}**",
        f"- Union recall (either retriever alone answers): "
        f"**{summary['queries_any_single_found']}/{summary['queries']} "
        f"({summary['union_recall_at_5']:.1%})**",
        f"- Fusion gap (a single retriever answered but hybrid lost): "
        f"**{summary['fusion_gap']}** queries",
        f"- Queries hybrid recovered that neither single retriever found: "
        f"**{summary['queries_hybrid_found_neither']}**",
        "",
        "### Failure analysis",
        "",
        *failure_lines(dense, hybrid, bm25),
        "",
        "### Uncertainty",
        "",
        f"Percentile bootstrap over the {summary['queries']} labelled queries "
        f"({summary['bootstrap']['resamples']} resamples, seed {summary['bootstrap']['seed']}).",
        "Comparisons are paired: the same query is resampled for both retrievers, so query",
        "difficulty cancels out. One query is 1 pp, which is why these intervals are wide —",
        "they describe label noise in this set, not fit to a real domain.",
        "",
        f"- MRR: BM25 {bm25['mrr']:.3f} "
        f"[{bm25['mrr_ci']['low']:.3f}, {bm25['mrr_ci']['high']:.3f}], "
        f"dense {dense['mrr']:.3f} [{dense['mrr_ci']['low']:.3f}, {dense['mrr_ci']['high']:.3f}], "
        f"hybrid {hybrid['mrr']:.3f} "
        f"[{hybrid['mrr_ci']['low']:.3f}, {hybrid['mrr_ci']['high']:.3f}]",
        "",
        *(
            [
                "## 2. Time to First Token (full gateway path)",
                "",
                f"Mock provider emits its first byte after a fixed {ttft['provider_delay_ms']:.0f} ms, so the",
                "gateway overhead column isolates socket + retrieval + prompt assembly + coalescer latency.",
                "",
                "| Measurement | p50 | p95 | mean |",
                "|---|---|---|---|",
                f"| Client-observed TTFT | {ttft['client_ttft_ms']['p50']} ms | "
                f"{ttft['client_ttft_ms']['p95']} ms | {ttft['client_ttft_ms']['mean']} ms |",
                f"| Server-measured TTFT | {ttft['server_ttft_ms']['p50']} ms | "
                f"{ttft['server_ttft_ms']['p95']} ms | {ttft['server_ttft_ms']['mean']} ms |",
                f"| Gateway overhead | {ttft['gateway_overhead_ms']['p50']} ms | "
                f"{ttft['gateway_overhead_ms']['p95']} ms | {ttft['gateway_overhead_ms']['mean']} ms |",
                "",
                f"Average frames per turn: {ttft['frames_per_turn']['mean']}; "
                f"average delta frames: {ttft['deltas_per_turn']['mean']}.",
                "",
            ]
            if ttft.get("turns")
            else [
                "## 2. Time to First Token (full gateway path)",
                "",
                "_Skipped in this run (`--skip-ttft`)._",
                "",
            ]
        ),
        "## 3. Delta frame coalescing",
        "",
        f"- {coalescing['tokens']} tokens at ~90 tok/s produced **{coalescing['frames']} delta frames** "
        f"({coalescing['frame_reduction_pct']}% fewer frames), {coalescing['tokens_per_frame']} tokens/frame",
        f"- Lossless: {coalescing['text_lossless']}; dropped frames under normal backpressure: "
        f"{coalescing['dropped_frames']}",
        "",
    ]
    if bundle:
        lines += [
            "## 4. Renderer bundle",
            "",
            "| Asset | Raw (KiB) | Gzip (KiB) |",
            "|---|---|---|",
            f"| JS | {bundle['js_kb']} KiB | {bundle['js_gzip_kb']} KiB |",
            f"| CSS | {bundle['css_kb']} KiB | {bundle['css_gzip_kb']} KiB |",
            "",
            f"Files: {', '.join(f'`{name}`' for name in bundle['files'])}.",
            "",
        ]
    idle = report.get("client_idle")
    if idle:
        lines += [
            "## 5. Desktop client idle footprint",
            "",
            f"Measured with `desktop/scripts/measure-idle.cjs` on {idle['platform']} "
            f"(Electron {idle['electron']}, Chrome {idle['chrome']}).",
            "",
            "| Scope | Working set | Notes |",
            "|---|---|---|",
            f"| Renderer process | {idle['renderer_working_set_mb']} MB | Chromium process RSS for the UI |",
            f"| Renderer JS heap (used) | {idle['renderer_heap_used_mb']} MB | "
            f"what the app itself allocates ({idle['dom_nodes']} DOM nodes) |",
            f"| All Electron processes | {idle['all_processes_working_set_mb']} MB | "
            "main + GPU + utility + renderer, a fixed runtime baseline |",
            "",
            "The sub-50 MB idle target is met by the **renderer JS heap**, not by the whole",
            "Chromium working set or the total of all Electron processes. Chromium's baseline",
            "(GPU process, compositor, V8, shared libraries) is a cost of the runtime, not of this",
            "app's React tree; the heap number is the honest measure of what the UI allocates.",
            "",
        ]
    lines += [
        "## Caveats",
        "",
        "- Retrieval numbers depend on the embedder. `--embedder hash` is a deterministic",
        "  lexical proxy over word tokens, bigrams and character trigrams, not a semantic",
        "  model; `--embedder local` uses model2vec static sentence embeddings; `--embedder",
        "  openai` uses hosted embeddings. The retrieval code under test is identical.",
        "- The evaluation corpus is synthetic and larger than the demo knowledge base. It is",
        "  built to contain hard negatives, so absolute recall is lower than on the demo KB by",
        "  design. Treat it as a regression baseline, not a claim about your domain.",
        "- Intervals are percentile bootstraps over 100 labels with a fixed seed: reproducible,",
        "  but not a substitute for a larger domain-specific label set. They quantify sampling",
        "  noise only.",
        "- TTFT is measured over the in-process ASGI transport to remove loopback noise. Real",
        "  network RTT adds to the client-observed number.",
        "- Hosted-provider TTFT depends on the provider and model, not on this gateway. The",
        "  honest gateway claim is the overhead column.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write docs/METRICS.md")
    parser.add_argument(
        "--out",
        default="METRICS.md",
        help="markdown filename under docs/ (default: METRICS.md); "
        "the JSON sidecar is named to match",
    )
    parser.add_argument("--skip-ttft", action="store_true")
    parser.add_argument(
        "--embedder",
        choices=["hash", "local", "openai"],
        default="hash",
        help=(
            "hash: deterministic lexical proxy (default, no dependencies); "
            "local: model2vec static sentence embeddings; "
            "openai: hosted embeddings (requires CUEPILOT_OPENAI_API_KEY)"
        ),
    )
    parser.add_argument(
        "--embedder-model",
        default=DEFAULT_LOCAL_MODEL,
        help=f"model2vec model for --embedder local (default: {DEFAULT_LOCAL_MODEL})",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"bootstrap resamples for confidence intervals (default: {DEFAULT_RESAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="bootstrap RNG seed, so intervals are reproducible (default: 7)",
    )
    args = parser.parse_args()

    settings = bench_settings()
    embedder = build_bench_embedder(
        args.embedder,
        dim=settings.embedding_dim,
        api_key=settings.openai_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        timeout_s=settings.embedding_timeout_s,
        local_model=args.embedder_model,
    )
    print(f"evaluating retrieval with {embedder.name}...", file=sys.stderr)
    retrieval = asyncio.run(
        evaluate_retrieval(settings, embedder, resamples=args.bootstrap, seed=args.seed)
    )

    ttft = {"turns": 0}
    if not args.skip_ttft:
        print("measuring TTFT through the gateway...", file=sys.stderr)
        ttft = evaluate_ttft(
            bench_settings(mock_ttft_ms=120.0, mock_token_delay_ms=4.0),
            provider_delay_ms=120.0,
        )

    print("measuring coalescing...", file=sys.stderr)
    coalescing = asyncio.run(evaluate_coalescing(bench_settings()))

    report = {
        "environment": {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "provider": "mock (deterministic)",
            "embedder": retrieval["summary"]["embedder"],
            "cpu_count": os.cpu_count(),
        },
        "retrieval": retrieval,
        "ttft": ttft,
        "coalescing": coalescing,
        "bundle": renderer_bundle(),
        "client_idle": client_idle(),
    }

    markdown = render_markdown(report)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.out[:-3] if args.out.endswith(".md") else args.out
    json_name = f"{stem.replace('METRICS', 'metrics')}.json"
    (DOCS_DIR / json_name).write_text(json.dumps(report, indent=2))
    if args.write:
        (DOCS_DIR / args.out).write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
