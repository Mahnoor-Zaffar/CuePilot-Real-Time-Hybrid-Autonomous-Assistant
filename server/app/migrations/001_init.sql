-- 001 — core vector schema.
--
-- Templated: the runner substitutes {{embedding_dim}} from Settings before this
-- file is applied, so the column width always matches the configured embedder.
-- Changing that parameter is a *rebuild*, not an in-place edit: the runner
-- applies 001_init.down.sql first. See app/infra/migrations.py.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT        NOT NULL,
    title       TEXT        NOT NULL DEFAULT '',
    content     TEXT        NOT NULL,
    embedding   VECTOR({{embedding_dim}}) NOT NULL,
    meta        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    kb_version  INTEGER     NOT NULL DEFAULT 1
);

-- HNSW gives us sub-linear ANN search with cosine distance (<=>).
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
    ON kb_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS kb_chunks_doc_idx ON kb_chunks (doc_id);
CREATE INDEX IF NOT EXISTS kb_chunks_meta_idx ON kb_chunks USING gin (meta);

CREATE TABLE IF NOT EXISTS kb_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT kb_meta_single_row CHECK (id = 1)
);

INSERT INTO kb_meta (id, version) VALUES (1, 1)
ON CONFLICT (id) DO NOTHING;
