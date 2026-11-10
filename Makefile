# cuePilot — developer entry points.
#
# Everything runs without cloud credentials: the server falls back to the mock
# LLM, the hash embedder, and in-process cache/vector stores. Set the keys in
# .env (see .env.example) to exercise the hosted paths.

SHELL := /bin/bash
PY := server/.venv/bin/python
PIP := server/.venv/bin/pip
RUFF := server/.venv/bin/ruff
UVICORN := server/.venv/bin/uvicorn

.PHONY: help setup setup-server setup-desktop setup-tools up down logs migrate dev server desktop test \
        test-local lint typecheck build docs-check bench bench-semantic seed metrics measure-idle \
        check clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── setup ────────────────────────────────────────────────────────────────────
setup: setup-server setup-desktop setup-tools ## Install server, desktop, and tooling dependencies
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

setup-server: ## Create server/.venv and install deps
	@command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
	cd server && uv venv --python 3.12 && uv pip install -e ".[dev]"
	@echo "tip: 'cd server && uv sync --extra local' adds the offline semantic embedder"

setup-desktop: ## Install desktop dependencies
	cd desktop && npm install

setup-tools: ## Install repository tooling (documentation gate)
	npm install

# ── infrastructure ───────────────────────────────────────────────────────────
up: ## Start Postgres (pgvector) and Redis
	docker compose up -d
	@docker compose ps

down: ## Stop infrastructure (volumes are kept)
	docker compose down

logs: ## Follow infrastructure logs
	docker compose logs -f

migrate: ## Apply SQL migrations without booting the gateway
	cd server && .venv/bin/python scripts/migrate.py

# ── run ──────────────────────────────────────────────────────────────────────
dev: ## Run gateway + desktop app together
	@trap 'kill 0' EXIT INT TERM; \
	$(MAKE) server & \
	(cd desktop && npm run dev) & \
	wait

server: ## Run the FastAPI gateway with reload on :8000
	cd server && .venv/bin/uvicorn app.main:app --reload --port 8000

desktop: ## Run the Electron client against a running gateway
	cd desktop && npm run dev

seed: ## Re-ingest the knowledge base into the configured vector store
	curl -fsS -X POST http://127.0.0.1:8000/api/kb/ingest \
		-H 'Content-Type: application/json' \
		-d '{"seed": true, "replace": true}' | $(PY) -m json.tool

# ── quality ──────────────────────────────────────────────────────────────────
test: ## Run the server test suite
	cd server && .venv/bin/python -m pytest -q

test-local: ## Run the local semantic embedder tests (needs: uv sync --extra local)
	cd server && .venv/bin/python -m pytest -q -m local

lint: ## Ruff lint
	cd server && .venv/bin/ruff check .

typecheck: ## TypeScript typecheck for the desktop client
	cd desktop && npm run typecheck

build: ## Build the desktop production bundle
	cd desktop && npm run build

docs-check: ## Validate mermaid diagrams and relative links in docs
	npm run check:docs

check: lint typecheck test docs-check ## Lint, typecheck, test, and check docs

# ── metrics ──────────────────────────────────────────────────────────────────
bench: ## Regenerate docs/metrics.json (retrieval, TTFT, coalescing, bundle, idle)
	cd server && .venv/bin/python bench/run_bench.py --write

bench-semantic: ## Retrieval benchmark with a real semantic embedder (needs the bench extra)
	cd server && .venv/bin/python bench/run_bench.py --write --embedder local --out METRICS.semantic.md

bench-quick: ## Retrieval benchmark, fewer bootstrap resamples (faster, wider intervals)
	cd server && .venv/bin/python bench/run_bench.py --skip-ttft --bootstrap 400

measure-idle: ## Measure desktop idle footprint into desktop/bench/idle.json
	cd desktop && npm run measure-idle

metrics: measure-idle bench ## Measure idle memory, then regenerate all metrics

clean: ## Remove build artifacts (keeps .venv and node_modules)
	rm -rf desktop/dist desktop/dist-electron desktop/bench/idle.json
