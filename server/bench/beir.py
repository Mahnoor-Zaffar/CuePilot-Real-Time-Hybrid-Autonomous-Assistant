#!/usr/bin/env python
"""Opt-in BEIR subset benchmark for cuePilot.

Runs the same retrieval-quality evaluation as bench/run_bench.py, but against
an external BEIR dataset instead of the synthetic corpus. Nothing here runs in
CI or `make check`: it needs network access to download the dataset and can
take a minute or two.

Usage:
    cd server && .venv/bin/python bench/beir.py [--dataset scifact] [--write]

    --dataset     BEIR dataset name (default: scifact). Small ones: scifact,
                  nfcorpus, fiqa, arguana, trec-covid.
    --split       qrels split (default: test).
    --max-queries Cap on queries evaluated (default: 100).
    --max-docs    Cap on corpus documents ingested (default: 5000).
    --embedder    hash (default) or local (model2vec, needs the local extra).
    --top-k       Retrieval depth (default: 5).
    --write       Write docs/METRICS.beir.md and docs/metrics.beir.json.

The dataset is cached under bench/data/ after the first download.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from app.config import Settings  # noqa: E402
from app.infra.cache import MemoryCache  # noqa: E402
from app.infra.db import MemoryVectorStore  # noqa: E402
from app.retrieval.orchestrator import HybridRetriever  # noqa: E402
from app.telemetry.metrics import Metrics  # noqa: E402
from app.telemetry.otel import TraceRecorder  # noqa: E402
from bench.embedders import DEFAULT_LOCAL_MODEL, build_bench_embedder  # noqa: E402
from bench.stats import (  # noqa: E402
    DEFAULT_RESAMPLES,
    bootstrap_ci,
    paired_bootstrap_delta,
)

DATA_DIR = ROOT / "bench" / "data"
DOCS_DIR = ROOT.parent / "docs"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def bench_settings(embedder_name: str) -> Settings:
    provider = "local" if embedder_name == "local" else "hash"
    return Settings(
        env="test",
        llm_provider="mock",
        embedding_provider=provider,
        local_embedding_model=DEFAULT_LOCAL_MODEL,
        allow_degraded_infra=True,
        infra_probe_timeout_s=0.05,
        database_url="postgresql://x:x@127.0.0.1:1/none",
        redis_url="redis://127.0.0.1:1/0",
        otel_enabled=False,
        otel_exporter="none",
        seed_on_start=True,
        mock_token_delay_ms=0.0,
        mock_ttft_ms=0.0,
        audio_enabled=False,
    )


def download_dataset(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name
    if (target / "corpus.jsonl").exists():
        return target
    url = BEIR_URL.format(name=name)
    archive = DATA_DIR / f"{name}.zip"
    print(f"downloading {url} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)
    print("extracting ...", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)
    archive.unlink(missing_ok=True)
    # BEIR zips extract to a folder named after the dataset, but sometimes
    # with a version suffix; normalise.
    if not (target / "corpus.jsonl").exists():
        candidates = [p for p in DATA_DIR.iterdir() if p.is_dir() and (p / "corpus.jsonl").exists()]
        if candidates:
            candidates[0].rename(target)
    if not (target / "corpus.jsonl").exists():
        raise RuntimeError(f"dataset {name!r} did not extract as expected under {DATA_DIR}")
    return target


def load_beir(dataset_dir: Path, split: str, max_queries: int, max_docs: int, seed: int):
    corpus: dict[str, dict] = {}
    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            corpus[row["_id"]] = row
            if len(corpus) >= max_docs:
                break

    queries: dict[str, str] = {}
    with open(dataset_dir / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    with open(qrels_path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[cid] = score

    # keep only queries that have qrels and whose relevant docs survived the cap
    usable = []
    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        relevant = {cid for cid, s in rels.items() if s >= 1 and cid in corpus}
        if not relevant:
            continue
        usable.append((qid, relevant))

    rng = random.Random(seed)
    rng.shuffle(usable)
    usable = usable[:max_queries]

    return corpus, queries, usable


def make_documents(corpus: dict[str, dict]) -> list[dict]:
    docs = []
    for doc_id, row in corpus.items():
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        content = (title + "\n\n" + text).strip() if title else text
        docs.append({"doc_id": doc_id, "title": title, "text": content})
    return docs


async def evaluate(dataset: str, split: str, max_queries: int, max_docs: int,
                  embedder_name: str, top_k: int, resamples: int, seed: int):
    dataset_dir = download_dataset(dataset)
    corpus, queries, usable = load_beir(dataset_dir, split, max_queries, max_docs, seed)
    documents = make_documents(corpus)

    settings = bench_settings(embedder_name)
    embedder = build_bench_embedder(
        embedder_name,
        dim=settings.embedding_dim,
        api_key=settings.openai_api_key,
        base_url="https://api.openai.com/v1",
        model=settings.embedding_model,
        timeout_s=30.0,
        local_model=settings.local_embedding_model,
    )
    metrics = Metrics()
    retriever = HybridRetriever(
        settings, MemoryVectorStore(embedder.dim), embedder, MemoryCache(), metrics, TraceRecorder()
    )
    ingest_started = time.perf_counter()
    ingest = await retriever.ingest(documents)
    ingest_ms = (time.perf_counter() - ingest_started) * 1000

    report: dict[str, dict] = {}
    for mode in ("bm25", "dense", "hybrid"):
        latencies: list[float] = []
        hit_vector: list[float] = []
        reciprocal_ranks: list[float] = []
        hits = 0
        per_query = []
        for qid, relevant in usable:
            started = time.perf_counter()
            result = await retriever.search(queries[qid], mode=mode, top_k=top_k, use_cache=False)
            latencies.append((time.perf_counter() - started) * 1000)
            docs = [hit.chunk.doc_id for hit in result.hits]
            rank = next((i + 1 for i, d in enumerate(docs) if d in relevant), None)
            hit = rank is not None
            hits += int(hit)
            hit_vector.append(1.0 if hit else 0.0)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            per_query.append({"qid": qid, "query": queries[qid], "rank": rank, "top_docs": docs})
        n = len(usable)
        report[mode] = {
            "recall_at_k": round(hits / n, 4) if n else 0.0,
            "recall_at_k_ci": bootstrap_ci(hit_vector, resamples=resamples, seed=seed).to_dict() if n else None,
            "mrr": round(statistics.fmean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            "mrr_ci": bootstrap_ci(reciprocal_ranks, resamples=resamples, seed=seed).to_dict() if reciprocal_ranks else None,
            "hits": hits,
            "n": n,
            "latency_ms": {
                "count": len(latencies),
                "p50": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else None,
            },
            "per_query": per_query,
        }

    bm25 = report["bm25"]
    dense = report["dense"]
    hybrid = report["hybrid"]
    best_single = max(bm25["recall_at_k"], dense["recall_at_k"])
    report["summary"] = {
        "dataset": dataset,
        "split": split,
        "corpus_documents": len(documents),
        "corpus_chunks": ingest["chunks"],
        "ingest_ms": round(ingest_ms, 2),
        "embedder": ingest["embedder"],
        "queries": len(usable),
        "top_k": top_k,
        "hybrid_vs_best_single_pp": round((hybrid["recall_at_k"] - best_single) * 100, 2),
        "hybrid_vs_dense_pp": round((hybrid["recall_at_k"] - dense["recall_at_k"]) * 100, 2),
        "hybrid_vs_bm25_pp": round((hybrid["recall_at_k"] - bm25["recall_at_k"]) * 100, 2),
        "paired_hybrid_vs_dense": paired_bootstrap_delta(
            [1.0 if q["rank"] else 0.0 for q in dense["per_query"]],
            [1.0 if q["rank"] else 0.0 for q in hybrid["per_query"]],
            resamples=resamples, seed=seed,
        ).to_dict(),
        "paired_hybrid_vs_bm25": paired_bootstrap_delta(
            [1.0 if q["rank"] else 0.0 for q in bm25["per_query"]],
            [1.0 if q["rank"] else 0.0 for q in hybrid["per_query"]],
            resamples=resamples, seed=seed,
        ).to_dict(),
    }
    return report


def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = []
    lines.append("# BEIR subset — retrieval quality")
    lines.append("")
    lines.append(f"Dataset: **{s['dataset']}** ({s['split']} split), {s['corpus_documents']} documents, "
                f"{s['queries']} queries, top-{s['top_k']}. Embedder: `{s['embedder']}`.")
    lines.append("")
    lines.append("| Mode | recall@k | 95% CI | MRR | 95% CI | hits |")
    lines.append("|---|---|---|---|---|---|")
    for mode in ("bm25", "dense", "hybrid"):
        r = report[mode]
        ci = r["recall_at_k_ci"] or {}
        mci = r["mrr_ci"] or {}
        lines.append(
            f"| {mode} | {r['recall_at_k']:.4f} | [{ci.get('low', 0):.4f}, {ci.get('high', 0):.4f}] | "
            f"{r['mrr']:.4f} | [{mci.get('low', 0):.4f}, {mci.get('high', 0):.4f}] | {r['hits']}/{r['n']} |"
        )
    lines.append("")
    lines.append(f"Hybrid vs best single: {s['hybrid_vs_best_single_pp']:+.2f} pp.  "
                 f"Hybrid vs dense: {s['hybrid_vs_dense_pp']:+.2f} pp.  "
                 f"Hybrid vs BM25: {s['hybrid_vs_bm25_pp']:+.2f} pp.")
    lines.append("")
    lines.append("Paired bootstrap deltas (recall@k, pp):")
    for key in ("paired_hybrid_vs_dense", "paired_hybrid_vs_bm25"):
        d = s[key]
        lines.append(f"- {key}: {d['point']*100:+.2f} pp [{d['low']*100:+.2f}, {d['high']*100:+.2f}]")
    lines.append("")
    lines.append("Notes: queries and documents are sampled (see --max-queries/--max-docs). "
                 "Bootstrap CIs use 2000 resamples, seed 7. This is an external reference "
                 "point, not a fit to the cuePilot corpus; the synthetic-corpus numbers in "
                 "METRICS.md are the headline.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in BEIR subset benchmark.")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--embedder", choices=("hash", "local"), default="hash")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(evaluate(
        args.dataset, args.split, args.max_queries, args.max_docs,
        args.embedder, args.top_k, args.resamples, args.seed,
    ))

    md = render_markdown(report)
    print(md)

    if args.write:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        (DOCS_DIR / "METRICS.beir.md").write_text(md, encoding="utf-8")
        (DOCS_DIR / "metrics.beir.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {DOCS_DIR / 'METRICS.beir.md'} and metrics.beir.json", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())