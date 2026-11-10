-- Rebuild path for 001. The embedding dimension changed, so every stored vector
-- is now the wrong width and cannot be cast. This is derived data: kb_chunks is
-- re-ingested from the corpus on the next boot, so dropping it loses nothing.
--
-- Bumping kb_meta.version invalidates retrieval cache entries that were keyed
-- against the old vectors.

UPDATE kb_meta SET version = version + 1, updated_at = now() WHERE id = 1;

DROP TABLE IF EXISTS kb_chunks;
