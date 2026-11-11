from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from app.infra.migrations import (
    AppliedMigration,
    MigrationDriftError,
    MigrationError,
    MigrationParamError,
    checksum,
    discover,
    plan_migrations,
    render,
    run_migrations,
)

PG_URL = os.environ.get("CUEPILOT_TEST_DATABASE_URL")


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def applied_for(migrations, params) -> dict[str, AppliedMigration]:
    return {
        migration.version: AppliedMigration(
            version=migration.version,
            name=migration.name,
            checksum=migration.checksum,
            params=dict(params),
        )
        for migration in migrations
    }


# ── discovery ──────────────────────────────────────────────────────────────
def test_discover_orders_by_version_and_pairs_down_files(tmp_path: Path) -> None:
    write(tmp_path / "002_second.sql", "SELECT 2;")
    write(tmp_path / "001_first.sql", "SELECT 1;")
    write(tmp_path / "001_first.down.sql", "SELECT 0;")

    migrations = discover(tmp_path)

    assert [m.version for m in migrations] == ["001", "002"]
    assert migrations[0].down_path is not None
    assert migrations[0].down_template == "SELECT 0;"
    assert migrations[1].down_path is None


def test_discover_rejects_a_file_without_a_version(tmp_path: Path) -> None:
    write(tmp_path / "nope.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="bad migration filename"):
        discover(tmp_path)


def test_discover_rejects_duplicate_versions(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "SELECT 1;")
    write(tmp_path / "001_b.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="duplicate migration version"):
        discover(tmp_path)


def test_discover_requires_the_directory() -> None:
    with pytest.raises(MigrationError, match="not found"):
        discover(Path("/definitely/not/a/migrations/dir"))


def test_the_shipped_migrations_are_discoverable() -> None:
    migrations = discover()
    assert [m.version for m in migrations] == ["001", "002", "003"]
    first = migrations[0]
    assert first.down_path is not None
    assert "{{embedding_dim}}" in first.template
    # 002 is the journal: it owns no parameters, so a width change must not
    # rebuild it and delete the transcript.
    journal = migrations[1]
    assert journal.parameters == set()
    assert journal.requires == ()
    assert journal.down_path is not None
    # 003 alters `turns`, which 002 owns, and must say so.
    checkpoints = migrations[2]
    assert checkpoints.parameters == set()
    assert checkpoints.requires == ("002",)
    assert checkpoints.down_path is not None


def test_discover_parses_requires_headers(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "SELECT 1;")
    write(tmp_path / "002_b.sql", "-- requires: 001\nALTER TABLE t ADD x INT;")
    write(tmp_path / "003_c.sql", "-- requires: 001, 002\nSELECT 3;")

    migrations = discover(tmp_path)

    assert migrations[0].requires == ()
    assert migrations[1].requires == ("001",)
    assert migrations[2].requires == ("001", "002")


def test_discover_rejects_a_requirement_on_a_missing_migration(tmp_path: Path) -> None:
    write(tmp_path / "002_b.sql", "-- requires: 001\nSELECT 2;")
    with pytest.raises(MigrationError, match="does not exist"):
        discover(tmp_path)


def test_discover_rejects_a_forward_requirement(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "-- requires: 002\nSELECT 1;")
    write(tmp_path / "002_b.sql", "SELECT 2;")
    with pytest.raises(MigrationError, match="earlier one"):
        discover(tmp_path)


# ── rendering ──────────────────────────────────────────────────────────────
def test_render_substitutes_scalars() -> None:
    assert render("VECTOR({{ embedding_dim }})", {"embedding_dim": 256}) == "VECTOR(256)"


def test_render_rejects_an_unknown_parameter() -> None:
    with pytest.raises(MigrationParamError, match="needs parameter 'nope'"):
        render("VECTOR({{nope}})", {"embedding_dim": 1})


def test_render_rejects_a_non_scalar_parameter() -> None:
    with pytest.raises(MigrationParamError, match="must be scalar"):
        render("{{x}}", {"x": ["a"]})


def test_checksum_ignores_line_endings_but_not_content() -> None:
    assert checksum("a\r\nb") == checksum("a\nb")
    assert checksum("a\nb") != checksum("a\nc")


def test_a_migration_declares_the_parameters_it_uses(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "VECTOR({{embedding_dim}}); -- {{other}}")
    assert discover(tmp_path)[0].parameters == {"embedding_dim", "other"}


# ── planning ───────────────────────────────────────────────────────────────
def test_plan_applies_everything_on_a_fresh_database(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "SELECT 1;")
    write(tmp_path / "002_b.sql", "SELECT 2;")
    plan = plan_migrations(discover(tmp_path), {}, {"embedding_dim": 1536})

    assert [a.kind for a in plan.actions] == ["apply", "apply"]
    assert plan.changed
    assert plan.missing == []


def test_plan_is_a_noop_when_the_schema_is_current(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "SELECT 1;")
    migrations = discover(tmp_path)
    plan = plan_migrations(migrations, applied_for(migrations, {"embedding_dim": 1536}),
                          {"embedding_dim": 1536})

    assert not plan.changed
    assert plan.up_to_date == ["001"]


def test_a_migration_with_no_parameters_ignores_an_unrelated_change(tmp_path: Path) -> None:
    # Rows written by an older runner recorded the whole settings dict. Only the
    # keys a file declares may trigger a rebuild, or 002 would be dropped on
    # every embedding-width change.
    write(tmp_path / "001_a.sql", "VECTOR({{embedding_dim}})")
    write(tmp_path / "001_a.down.sql", "DROP TABLE t;")
    write(tmp_path / "002_b.sql", "CREATE TABLE journal (id TEXT);")
    write(tmp_path / "002_b.down.sql", "DROP TABLE journal;")
    migrations = discover(tmp_path)
    applied = applied_for(migrations, {"embedding_dim": 1536})

    plan = plan_migrations(migrations, applied, {"embedding_dim": 256})

    assert [(a.kind, a.migration.version) for a in plan.actions] == [("rebuild", "001")]


def test_plan_detects_an_edited_migration(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "SELECT 1;")
    migrations = discover(tmp_path)
    applied = applied_for(migrations, {"embedding_dim": 1536})

    write(tmp_path / "001_a.sql", "SELECT 1; -- sneaky edit")
    with pytest.raises(MigrationDriftError, match="changed after it was applied"):
        plan_migrations(discover(tmp_path), applied, {"embedding_dim": 1536})


def test_plan_rebuilds_when_a_parameter_changes(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "VECTOR({{embedding_dim}})")
    write(tmp_path / "001_a.down.sql", "DROP TABLE t;")
    migrations = discover(tmp_path)
    plan = plan_migrations(
        migrations, applied_for(migrations, {"embedding_dim": 1536}), {"embedding_dim": 256}
    )

    assert [a.kind for a in plan.actions] == ["rebuild"]
    assert plan.actions[0].migration.version == "001"


def test_plan_refuses_a_parameter_change_without_a_down_file(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "VECTOR({{embedding_dim}})")
    migrations = discover(tmp_path)
    with pytest.raises(MigrationParamError, match="does not exist"):
        plan_migrations(
            migrations, applied_for(migrations, {"embedding_dim": 1536}), {"embedding_dim": 256}
        )


def test_a_rebuild_cascades_through_declared_dependencies(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "VECTOR({{embedding_dim}})")
    write(tmp_path / "001_a.down.sql", "DROP TABLE t;")
    write(tmp_path / "002_b.sql", "-- requires: 001\nALTER TABLE t ADD COLUMN x INT;")
    write(tmp_path / "002_b.down.sql", "ALTER TABLE t DROP COLUMN x;")
    migrations = discover(tmp_path)
    plan = plan_migrations(
        migrations, applied_for(migrations, {"embedding_dim": 1536}), {"embedding_dim": 256}
    )

    assert [a.kind for a in plan.actions] == ["rebuild", "rebuild"]


def test_a_dependent_without_a_down_file_is_a_hard_error(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "VECTOR({{embedding_dim}})")
    write(tmp_path / "001_a.down.sql", "DROP TABLE t;")
    write(tmp_path / "002_b.sql", "-- requires: 001\nALTER TABLE t ADD COLUMN x INT;")
    migrations = discover(tmp_path)
    with pytest.raises(MigrationParamError, match="does not exist"):
        plan_migrations(
            migrations, applied_for(migrations, {"embedding_dim": 1536}), {"embedding_dim": 256}
        )


def test_a_new_migration_after_a_rebuild_is_applied(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "VECTOR({{embedding_dim}})")
    write(tmp_path / "001_a.down.sql", "DROP TABLE t;")
    migrations = discover(tmp_path)
    applied = applied_for(migrations, {"embedding_dim": 1536})

    write(tmp_path / "002_b.sql", "ALTER TABLE t ADD COLUMN x INT;")
    plan = plan_migrations(discover(tmp_path), applied, {"embedding_dim": 256})

    assert [(a.kind, a.migration.version) for a in plan.actions] == [
        ("rebuild", "001"),
        ("apply", "002"),
    ]


def test_plan_reports_applied_versions_with_no_file(tmp_path: Path) -> None:
    write(tmp_path / "001_a.sql", "SELECT 1;")
    migrations = discover(tmp_path)
    applied = applied_for(migrations, {"embedding_dim": 1536})
    applied["009"] = AppliedMigration("009", "gone", "abc", {"embedding_dim": 1536})

    plan = plan_migrations(migrations, applied, {"embedding_dim": 1536})
    assert plan.missing == ["009"]


# ── execution (needs a real Postgres with pgvector) ────────────────────────
pytestmark_pg = pytest.mark.skipif(
    not PG_URL, reason="set CUEPILOT_TEST_DATABASE_URL to run migration integration tests"
)


async def _column_type(pool) -> str:
    return await pool.fetchval(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'kb_chunks'::regclass AND attname = 'embedding'"
    )


async def _reset(pool) -> None:
    await pool.execute(
        "DROP TABLE IF EXISTS turns; "
        "DROP TABLE IF EXISTS sessions; "
        "DROP TABLE IF EXISTS kb_chunks; "
        "DROP TABLE IF EXISTS kb_meta; "
        "DROP TABLE IF EXISTS schema_migrations"
    )


@pytestmark_pg
async def test_migrations_apply_then_rebuild_on_dimension_change() -> None:
    import asyncpg

    pool = await asyncpg.create_pool(PG_URL, min_size=1, max_size=2)
    assert pool is not None
    try:
        await _reset(pool)

        first = await run_migrations(pool, params={"embedding_dim": 1536})
        assert first.applied == ["001", "002", "003"]
        assert await _column_type(pool) == "vector(1536)"

        again = await run_migrations(pool, params={"embedding_dim": 1536})
        assert not again.changed
        assert again.up_to_date == ["001", "002", "003"]

        # The journal is the only copy of a transcript, so prove a width change
        # does not rebuild it.
        await pool.execute(
            "INSERT INTO sessions (id) VALUES ('s1'); "
            "INSERT INTO turns (session_id, turn_id, prompt, answer, status) "
            "VALUES ('s1', 't1', 'why did the socket drop', 'backpressure', 'done')"
        )

        rebuilt = await run_migrations(pool, params={"embedding_dim": 256})
        assert rebuilt.rebuilt == ["001"]
        assert await _column_type(pool) == "vector(256)"
        # Derived vectors are gone, and the version bump invalidates caches.
        assert await pool.fetchval("SELECT count(*) FROM kb_chunks") == 0
        assert await pool.fetchval("SELECT version FROM kb_meta WHERE id = 1") == 2
        # ...but the transcript survived.
        assert await pool.fetchval("SELECT count(*) FROM turns") == 1
        assert await pool.fetchval("SELECT count(*) FROM sessions") == 1
    finally:
        await _reset(pool)
        await pool.close()


@pytestmark_pg
async def test_a_table_created_outside_the_runner_is_repaired() -> None:
    import asyncpg

    pool = await asyncpg.create_pool(PG_URL, min_size=1, max_size=2)
    assert pool is not None
    try:
        await _reset(pool)
        # The old `init.sql` bootstrap: schema exists, but the runner has no row
        # for it, so the recorded parameters cannot be trusted. Verify the shape.
        await pool.execute(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE kb_chunks (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT        NOT NULL,
                title       TEXT        NOT NULL DEFAULT '',
                content     TEXT        NOT NULL,
                embedding   VECTOR(1536) NOT NULL,
                meta        JSONB       NOT NULL DEFAULT '{}'::jsonb,
                kb_version  INTEGER     NOT NULL DEFAULT 1
            );
            CREATE TABLE kb_meta (
                id      INTEGER PRIMARY KEY DEFAULT 1,
                version INTEGER NOT NULL DEFAULT 1,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            INSERT INTO kb_meta (id, version) VALUES (1, 1)
            """
        )

        result = await run_migrations(pool, params={"embedding_dim": 256})

        assert result.applied == ["001", "002", "003"]
        assert result.rebuilt == ["001"]
        assert await _column_type(pool) == "vector(256)"
    finally:
        await _reset(pool)
        await pool.close()


@pytestmark_pg
async def test_concurrent_boots_apply_a_migration_once() -> None:
    import asyncpg

    pool = await asyncpg.create_pool(PG_URL, min_size=1, max_size=2)
    assert pool is not None
    try:
        await _reset(pool)
        results = await asyncio.gather(
            run_migrations(pool, params={"embedding_dim": 1536}),
            run_migrations(pool, params={"embedding_dim": 1536}),
        )
        applied = [version for result in results for version in result.applied]
        assert applied == ["001", "002", "003"]
        assert await pool.fetchval("SELECT count(*) FROM schema_migrations") == 3
    finally:
        await _reset(pool)
        await pool.close()
