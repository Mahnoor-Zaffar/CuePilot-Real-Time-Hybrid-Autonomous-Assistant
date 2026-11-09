# syntax=docker/dockerfile:1.7

# ── build ────────────────────────────────────────────────────────────────────
# Dependencies are installed from the committed lockfile into a self-contained
# venv. Two sync passes keep the expensive layer cached when only app code
# changes: first the third-party deps, then the project itself.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
RUN uv sync --frozen --no-dev

# ── runtime ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="cuePilot Gateway" \
      org.opencontainers.image.description="Bi-directional WebSocket gateway with hybrid BM25 + pgvector retrieval and a streaming tool-calling agent" \
      org.opencontainers.image.source="https://github.com/Mahnoor-Zaffar/CuePilot-Real-Time-Hybrid-Autonomous-Assistant" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Run unprivileged. The image never writes to disk, so no volumes are required.
RUN groupadd --system --gid 10001 cuepilot \
    && useradd --system --uid 10001 --gid cuepilot --create-home cuepilot

WORKDIR /app
COPY --from=builder --chown=cuepilot:cuepilot /app /app

USER cuepilot
EXPOSE 8000

# Starts degraded (in-process vector store, cache and bus) if Postgres/Redis are
# unreachable, so the healthcheck is meaningful with or without infrastructure.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
