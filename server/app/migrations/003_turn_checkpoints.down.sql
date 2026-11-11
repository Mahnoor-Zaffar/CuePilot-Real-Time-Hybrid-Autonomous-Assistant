-- Rebuild path for 003. A streaming row cannot satisfy 002's terminal-only
-- CHECK, so normalise it to the status recovery would have assigned before
-- removing the column that identifies it as interrupted.

UPDATE turns
SET status = 'aborted', finish_reason = COALESCE(finish_reason, 'interrupted')
WHERE status = 'streaming';

DROP INDEX IF EXISTS turns_streaming_updated_idx;

ALTER TABLE turns DROP CONSTRAINT IF EXISTS turns_status_check;
ALTER TABLE turns
    ADD CONSTRAINT turns_status_check
    CHECK (status IN ('done', 'aborted', 'error'));

ALTER TABLE turns DROP COLUMN IF EXISTS updated_at;
