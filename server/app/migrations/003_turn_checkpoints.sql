-- 003 — partial-turn checkpoints.
--
-- 002 only allowed terminal turn statuses, so a hard crash mid-turn lost the
-- partial answer: the journal was written on `done` and `abort`, and a process
-- that never reached either left nothing behind. A checkpoint needs two things:
--
--   1. a non-terminal `streaming` status, so a partial row is not mistaken for
--      a finished one; and
--   2. `updated_at`, so recovery can distinguish a live turn on another replica
--      from a turn whose replica stopped checkpointing.
--
-- Checkpoints are best-effort. The terminal write still replaces the checkpoint
-- and is the row that matters.
--
-- This alters `turns`, which 002 owns, so the dependency is declared. 002 has
-- no template parameters and never rebuilds automatically; the header keeps the
-- rebuild graph honest if an operator ever drops it explicitly.

-- requires: 002

ALTER TABLE turns
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();

ALTER TABLE turns DROP CONSTRAINT IF EXISTS turns_status_check;
ALTER TABLE turns
    ADD CONSTRAINT turns_status_check
    CHECK (status IN ('streaming', 'done', 'aborted', 'error'));

-- Stale-turn recovery scans only streaming rows; the partial index is that read.
CREATE INDEX IF NOT EXISTS turns_streaming_updated_idx
    ON turns (updated_at) WHERE status = 'streaming';
