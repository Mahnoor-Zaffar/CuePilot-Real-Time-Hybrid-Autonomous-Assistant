from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime knobs. Every field is overridable with CUEPILOT_<NAME>."""

    model_config = SettingsConfigDict(
        env_prefix="CUEPILOT_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── app ────────────────────────────────────────────────────────────────
    app_name: str = "cuepilot"
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api_token: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ── LLM (OpenRouter speaks the OpenAI chat-completions dialect) ────────
    llm_provider: Literal["auto", "openrouter", "mock"] = "auto"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_name: str = "cuePilot"
    openrouter_app_url: str = "https://github.com/cuepilot/cuepilot"
    llm_model: str = "openai/gpt-4o-mini"
    llm_fallback_models: list[str] = Field(
        default_factory=lambda: ["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"]
    )
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700
    llm_timeout_s: float = 60.0
    llm_connect_timeout_s: float = 10.0
    llm_max_retries: int = 3
    llm_retry_base_s: float = 0.4
    max_tool_iterations: int = 4
    tool_timeout_s: float = 8.0
    mock_token_delay_ms: float = 3.0
    mock_ttft_ms: float = 20.0

    # ── embeddings ─────────────────────────────────────────────────────────
    # auto: hosted embeddings when an OpenAI key is set, otherwise the hash proxy.
    # local: model2vec static sentence embeddings, no network at query time.
    embedding_provider: Literal["auto", "openai", "hash", "local"] = "auto"
    openai_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    # Requested width for the hosted/hash paths. A local model declares its own
    # width, which becomes the schema's `VECTOR(n)` (a change triggers a rebuild).
    embedding_dim: int = 1536
    local_embedding_model: str = "minishlab/potion-base-8M"
    embedding_timeout_s: float = 20.0

    # ── infrastructure ─────────────────────────────────────────────────────
    database_url: str = "postgresql://cuepilot:cuepilot@127.0.0.1:5433/cuepilot"
    redis_url: str = "redis://127.0.0.1:6380/0"
    allow_degraded_infra: bool = True
    infra_probe_timeout_s: float = 1.5

    # ── retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 8
    retrieval_candidate_k: int = 50
    retrieval_cache_ttl_s: int = 120
    retrieval_timeout_s: float = 6.0
    rrf_k: int = 60
    # Cap how many chunks a single document may contribute to the fused top-k.
    # Without this, one long document fills every slot and crowds out the answer.
    retrieval_max_chunks_per_doc: int = 1
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 120
    seed_on_start: bool = True

    # ── websocket ──────────────────────────────────────────────────────────
    ws_send_queue_size: int = 64
    delta_flush_ms: int = 16
    ws_heartbeat_interval_s: float = 20.0
    ws_hello_timeout_s: float = 10.0
    ws_max_message_bytes: int = 2_000_000
    turn_history_limit: int = 24
    session_ttl_s: int = 3600

    # ── client limits ──────────────────────────────────────────────────────
    # A fixed window keyed by client identity, shared through the cache. The
    # connection cap is per replica because `ws_active` is a process gauge.
    # Behind a trusted proxy set rate_limit_trust_forwarded_for to use the
    # left-most X-Forwarded-For address instead of the socket peer.
    rate_limit_enabled: bool = True
    rate_limit_window_s: int = 60
    rate_limit_connect_attempts: int = 30
    rate_limit_max_connections: int = 200
    rate_limit_trust_forwarded_for: bool = False

    # ── durable sessions ───────────────────────────────────────────────────
    # Redis mirrors sessions; the journal is the record. Writes are queued so a
    # slow Postgres costs transcript lines, never stream latency.
    session_journal_enabled: bool = True
    session_journal_queue: int = 256
    session_journal_timeout_s: float = 5.0
    session_journal_drain_s: float = 3.0
    # Partial answers are checkpointed while streaming. A hard crash loses at
    # most one interval of text. Checkpoints are droppable; terminal writes are
    # not, so those spool to local disk when the queue or Postgres is unavailable.
    turn_checkpoint_interval_ms: int = 250
    turn_checkpoint_stale_s: float = 60.0
    session_journal_spool_path: str = "data/journal-spool.ndjson"
    session_journal_spool_max_bytes: int = 8_000_000
    session_journal_spool_retry_s: float = 30.0

    # ── audio ingest ───────────────────────────────────────────────────────
    audio_enabled: bool = True
    audio_window_ms: int = 3000
    audio_sample_rate: int = 16000
    transcription_model: str = "whisper-1"
    transcription_base_url: str = "https://api.openai.com/v1"

    # ── telemetry ──────────────────────────────────────────────────────────
    otel_enabled: bool = True
    otel_exporter: Literal["console", "otlp", "none"] = "none"
    otel_endpoint: str = "http://127.0.0.1:4318"
    otel_service_name: str = "cuepilot-gateway"
    trace_buffer_turns: int = 200
    # Each replica publishes a sampled snapshot to the shared cache; /api/metrics
    # merges every live snapshot. Percentiles are recomputed from the most recent
    # `metrics_sample_limit` observations per worker, and say so when approximate.
    metrics_publish_interval_s: float = 10.0
    metrics_publish_ttl_s: int = 30
    metrics_sample_limit: int = 512

    @field_validator("cors_origins", "llm_fallback_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept both JSON arrays and comma-separated strings from env."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @property
    def provider_mode(self) -> Literal["openrouter", "mock"]:
        if self.llm_provider == "mock":
            return "mock"
        if self.llm_provider == "openrouter":
            return "openrouter"
        return "openrouter" if self.openrouter_api_key else "mock"

    @property
    def embedding_mode(self) -> Literal["openai", "hash", "local"]:
        if self.embedding_provider == "local":
            return "local"
        if self.embedding_provider == "hash":
            return "hash"
        if self.embedding_provider == "openai":
            return "openai"
        return "openai" if self.openai_api_key else "hash"

    @property
    def model_chain(self) -> list[str]:
        chain = [self.llm_model]
        chain.extend(m for m in self.llm_fallback_models if m != self.llm_model)
        return chain


@lru_cache
def get_settings() -> Settings:
    return Settings()
