-- 002 — durable session and turn journal.
--
-- Redis stays the fast path (a hot session is a single GET), but it is a cache,
-- not the record. Eviction or a Redis restart used to end resume and lose the
-- transcript; with these tables the worst case is a cache miss.
--
-- No template parameters here on purpose: nothing about this schema depends on
-- the embedding width, so a width change rebuilds 001 and leaves this data alone.
-- No `-- requires:` header for the same reason.

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    client      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    kb_version  INTEGER     NOT NULL DEFAULT 1,
    turns       INTEGER     NOT NULL DEFAULT 0,
    scratchpad  JSONB       NOT NULL DEFAULT '{}'::jsonb
);

-- One row per finished turn. This is the transcript: `sessions` is the header,
-- `turns` is what a user would expect to still be there tomorrow.
CREATE TABLE IF NOT EXISTS turns (
    session_id      TEXT        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id         TEXT        NOT NULL,
    prompt          TEXT        NOT NULL,
    answer          TEXT        NOT NULL DEFAULT '',
    status          TEXT        NOT NULL CHECK (status IN ('done', 'aborted', 'error')),
    finish_reason   TEXT,
    ttft_ms         DOUBLE PRECISION,
    total_ms        DOUBLE PRECISION,
    tool_iterations INTEGER     NOT NULL DEFAULT 0,
    usage           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    model           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_id)
);

-- Resume reads the newest turns for one session; this index is that read.
CREATE INDEX IF NOT EXISTS turns_session_created_idx
    ON turns (session_id, created_at DESC);
