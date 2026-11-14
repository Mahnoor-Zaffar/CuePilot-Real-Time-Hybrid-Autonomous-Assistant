-- Down file for 004. Dropping the outbox is safe only once it is empty: an
-- un-replayed row is a transcript that has not reached `turns` yet. Operators
-- should drain first (`replay_spool` on boot does this) and inspect before a
-- rebuild.
DROP TABLE IF EXISTS turn_outbox;
