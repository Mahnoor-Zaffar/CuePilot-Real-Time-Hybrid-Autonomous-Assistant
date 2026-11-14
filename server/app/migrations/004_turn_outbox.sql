-- 004 — durable turn outbox.
--
-- A terminal turn is the transcript, so it gets more than one fallback. The
-- local NDJSON spool survives a process crash but not losing the container
-- filesystem; the shared-cache spool depends on Redis persistence. This table
-- is the durable tier: an append-only outbox in Postgres, written when the
-- local spool is unavailable or full, and drained into `turns` by the journal
-- worker.
--
-- No foreign key to `sessions` on purpose. Under backpressure the session
-- header upsert can be dropped while a terminal turn survives, and the FK would
-- reject the transcript line. The relay ensures the header before the turn.
--
-- `payload` is the serialised `TurnRecord`, so the outbox schema does not have
-- to track every turn column. The relay deletes a row only after the `turns`
-- upsert commits.
--
-- This table depends on nothing but its own schema, so it declares no
-- parameters and does not rebuild when the embedding width changes.

-- requires: 002

CREATE TABLE IF NOT EXISTS turn_outbox (
    session_id  TEXT        NOT NULL,
    turn_id     TEXT        NOT NULL,
    payload     JSONB       NOT NULL,
    attempts    INTEGER     NOT NULL DEFAULT 0,
    last_error  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_id)
);

-- The relay reads oldest-first in bounded batches; this index is that read.
CREATE INDEX IF NOT EXISTS turn_outbox_created_idx
    ON turn_outbox (created_at ASC);
