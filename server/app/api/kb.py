from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_runtime, require_auth
from app.retrieval.corpus import all_documents
from app.runtime import Runtime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["knowledge-base"], dependencies=[Depends(require_auth)])


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    title: str = Field(default="", max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    tags: list[str] = Field(default_factory=list, max_length=32)
    source: str = Field(default="api", max_length=64)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(default_factory=list, max_length=500)
    replace: bool = False
    seed: bool = False


@router.post("/ingest")
async def ingest(payload: IngestRequest, runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    documents: list[dict[str, Any]] = [doc.model_dump() for doc in payload.documents]
    if payload.seed:
        documents.extend(all_documents())
    if not documents:
        raise HTTPException(status_code=422, detail="provide `documents` or set `seed: true`")

    result = await runtime.retriever.ingest(documents, replace=payload.replace)
    # Other replicas must drop their cached results and rebuild their BM25 index.
    await runtime.bus.publish(
        {"kind": "kb.invalidate", "origin": runtime.instance_id, "kb_version": result["kb_version"]}
    )
    return {"ok": True, **result}


@router.get("/search")
async def search(
    q: str = Query(min_length=1, max_length=1000),
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid",
    top_k: int = Query(default=8, ge=1, le=20),
    cache: bool = True,
    runtime: Runtime = Depends(get_runtime),
) -> dict[str, Any]:
    result = await runtime.retriever.search(q, top_k=top_k, mode=mode, use_cache=cache)
    return result.to_public()


@router.get("/stats")
async def stats(runtime: Runtime = Depends(get_runtime)) -> dict[str, Any]:
    return await runtime.retriever.stats()
