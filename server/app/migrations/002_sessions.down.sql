-- Rebuild path for 002. Unlike 001 this one is destructive: the journal is the
-- only copy of the transcript, so a rebuild here loses history that Redis may
-- already have evicted.
--
-- Nothing rebuilds this migration automatically: it declares no parameters and
-- nothing `-- requires:` it. Dropping it is an explicit operator action.

DROP TABLE IF EXISTS turns;
DROP TABLE IF EXISTS sessions;
