"""Vector store: pgvector in production, numpy cosine in-process fallback.

Both backends share one interface so the retrieval orchestrator never branches
on infrastructure. The fallback keeps the full hybrid pipeline (and its
benchmarks) runnable on a laptop with no Docker.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import asyncpg
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    content: str
    embedding: list[float]
    meta: dict[str, Any]

    def to_public(self, score: float | None = None) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "meta": self.meta,
        }
        if score is not None:
            payload["score"] = round(score, 6)
        return payload


class VectorStore(Protocol):
    backend: str
    dim: int

    async def upsert(self, chunks: list[Chunk]) -> int: ...
    async def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]: ...
    async def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    async def all_chunks(self) -> list[Chunk]: ...
    async def count(self) -> int: ...
    async def clear(self) -> None: ...
    async def bump_version(self) -> int: ...
    async def version(self) -> int: ...
    async def close(self) -> None: ...


class MemoryVectorStore:
    """Exact cosine search. Deterministic and fast below ~100k chunks."""

    backend = "memory"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: dict[str, Chunk] = {}
        self._matrix: np.ndarray | None = None
        self._index: list[str] = []
        self._version = 1
        self._dirty = True
        self._lock = asyncio.Lock()

    async def upsert(self, chunks: list[Chunk]) -> int:
        async with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
            self._dirty = True
            return len(chunks)

    def _rebuild(self) -> None:
        self._index = list(self._chunks)
        if not self._index:
            self._matrix = None
        else:
            self._matrix = np.asarray(
                [self._chunks[cid].embedding for cid in self._index], dtype=np.float32
            )
        self._dirty = False

    async def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        if self._dirty:
            self._rebuild()
        if self._matrix is None or not self._index:
            return []
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query)) or 1.0
        sims = self._matrix @ (query / norm)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._chunks[self._index[i]], float(sims[i])) for i in top]

    async def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    async def count(self) -> int:
        return len(self._chunks)

    async def clear(self) -> None:
        async with self._lock:
            self._chunks.clear()
            self._dirty = True

    async def bump_version(self) -> int:
        self._version += 1
        return self._version

    async def version(self) -> int:
        return self._version

    async def close(self) -> None:
        return None


class PgVectorStore:
    backend = "pgvector"

    def __init__(self, pool: asyncpg.Pool, dim: int) -> None:
        self._pool = pool
        self.dim = dim

    @staticmethod
    def _literal(vector: list[float]) -> str:
        return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

    async def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        version = await self.version()
        rows = [
            (
                c.id,
                c.doc_id,
                c.title,
                c.content,
                self._literal(c.embedding),
                json.dumps(c.meta),
                version,
            )
            for c in chunks
        ]
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                INSERT INTO kb_chunks (id, doc_id, title, content, embedding, meta, kb_version)
                VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb, $7)
                ON CONFLICT (id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    meta = EXCLUDED.meta,
                    kb_version = EXCLUDED.kb_version
                """,
                rows,
            )
        return len(rows)

    async def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        literal = self._literal(vector)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, doc_id, title, content, meta,
                       1 - (embedding <=> $1::vector) AS score
                FROM kb_chunks
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                literal,
                k,
            )
        return [
            (
                Chunk(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    title=row["title"],
                    content=row["content"],
                    embedding=[],
                    meta=json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"],
                ),
                float(row["score"]),
            )
            for row in rows
        ]

    async def all_chunks(self) -> list[Chunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, doc_id, title, content, meta FROM kb_chunks ORDER BY doc_id, id"
            )
        return [
            Chunk(
                id=row["id"],
                doc_id=row["doc_id"],
                title=row["title"],
                content=row["content"],
                embedding=[],
                meta=json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"],
            )
            for row in rows
        ]

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, doc_id, title, content, meta FROM kb_chunks WHERE id = $1", chunk_id
            )
        if row is None:
            return None
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            title=row["title"],
            content=row["content"],
            embedding=[],
            meta=json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"],
        )

    async def count(self) -> int:
        async with self._pool.acquire() as conn:
            return int(await conn.fetchval("SELECT count(*) FROM kb_chunks"))

    async def clear(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE kb_chunks")

    async def bump_version(self) -> int:
        async with self._pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    "UPDATE kb_meta SET version = version + 1, updated_at = now() "
                    "WHERE id = 1 RETURNING version"
                )
            )

    async def version(self) -> int:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval("SELECT version FROM kb_meta WHERE id = 1")
            return int(value or 1)

    async def close(self) -> None:
        await self._pool.close()


async def build_store(settings: Settings) -> tuple[VectorStore, str]:
    try:
        pool = await asyncio.wait_for(
            asyncpg.create_pool(
                settings.database_url,
                min_size=1,
                max_size=10,
                command_timeout=10,
                server_settings={"application_name": "cuepilot"},
            ),
            timeout=settings.infra_probe_timeout_s * 3,
        )
        assert pool is not None
        store = PgVectorStore(pool, settings.embedding_dim)
        await store.version()  # verifies schema exists
        return store, ""
    except Exception as exc:  # noqa: BLE001
        if not settings.allow_degraded_infra:
            raise
        logger.info("vector store degraded to in-memory: %s", exc)
        return MemoryVectorStore(settings.embedding_dim), (
            f"postgres unavailable ({type(exc).__name__})"
        )
