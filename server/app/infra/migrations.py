"""Forward-only SQL migrations with a cross-replica advisory lock.

Why not Alembic: the store is deliberately raw `asyncpg` with no ORM, so Alembic
would drag SQLAlchemy in for autogenerate we never use. The whole runner is one
file, and the migration files stay readable SQL that a DBA can review.

Guarantees:

* **Ordered.** Files apply in filename order (`001_`, `002_`, ...).
* **Exactly once.** Applied versions are recorded in `schema_migrations` with a
  checksum of the file that was run.
* **One writer.** A session-level advisory lock serialises replicas booting at
  the same time; the losers wait, then see the work already done.
* **No silent edits.** If an applied file changes, startup fails with a drift
  error. Editing history is how schemas diverge between environments.
* **Parameters are part of the schema.** A file may contain `{{name}}`; the
  rendered values are recorded. When a parameter changes, that migration is
  rebuilt: its `.down.sql` runs, then the file re-applies. A parameter change
  without a down file is a hard error, never a silent no-op.
* **Only the owning migration rebuilds.** A file records the parameters *it*
  declares, not the whole settings dict, so `002` (which declares none) is not
  dragged into a rebuild of `001`.
* **Dependencies are declared, not guessed.** A file that modifies objects
  created by an earlier migration says so with a header line:

      -- requires: 001

  A rebuild cascades through that graph, so `002_alter_kb_chunks.sql` is
  re-applied when `001` rebuilds, while an independent `002_sessions.sql` keeps
  its rows. Undeclared dependencies are the one thing this cannot detect, which
  is why the header is part of the review checklist.

The one real parameter today is `embedding_dim`. It has to live in the DDL
(`VECTOR(n)`), and pgvector cannot cast between widths, so the honest model is
"change the width -> rebuild the derived table". That is exactly what the down
file does.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import asyncpg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
LOCK_KEY = "cuepilot:migrations"

_NAME = re.compile(r"^(?P<version>\d{3,})[_-](?P<name>[A-Za-z0-9_.-]+)\.sql$")
_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
_REQUIRES = re.compile(r"^\s*--\s*requires:\s*(?P<versions>[0-9][0-9,\s]*)\s*$", re.MULTILINE)

_SCHEMA_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    name       TEXT        NOT NULL,
    checksum   TEXT        NOT NULL,
    params     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    """Base class: anything that must stop startup rather than degrade."""


class MigrationDriftError(MigrationError):
    """An already-applied migration file changed on disk."""


class MigrationParamError(MigrationError):
    """A migration's parameters changed but it cannot be rebuilt."""


# ── discovery ──────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    name: str
    path: Path
    template: str
    checksum: str
    down_path: Path | None = None
    down_template: str | None = None
    requires: tuple[str, ...] = ()

    @property
    def down_name(self) -> str:
        return f"{self.path.stem}.down.sql"

    @property
    def parameters(self) -> set[str]:
        """Template placeholders this file declares, e.g. {"embedding_dim"}."""
        return set(_TEMPLATE.findall(self.template))

    def render_up(self, params: Mapping[str, Any]) -> str:
        return render(self.template, params, source=self.path.name)

    def render_down(self, params: Mapping[str, Any]) -> str:
        if self.down_template is None:
            raise MigrationParamError(f"{self.version} has no {self.down_name}")
        return render(self.down_template, params, source=self.down_name)


def checksum(text: str) -> str:
    """Hash the *template*, so an edit is detected even though params vary."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def render(text: str, params: Mapping[str, Any], *, source: str = "<sql>") -> str:
    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            raise MigrationParamError(f"{source} needs parameter {key!r}; got {sorted(params)}")
        value = params[key]
        if not isinstance(value, (str, int, float, bool)):
            raise MigrationParamError(f"{source} parameter {key!r} must be scalar, got {value!r}")
        return str(value)

    return _TEMPLATE.sub(substitute, text)


def parse_requires(template: str) -> tuple[str, ...]:
    """Read `-- requires: 001, 002` header lines. No header means independent."""
    versions: list[str] = []
    for match in _REQUIRES.finditer(template):
        for part in re.split(r"[,\s]+", match.group("versions").strip()):
            if part and part not in versions:
                versions.append(part)
    return tuple(versions)


def discover(directory: Path | None = None) -> list[Migration]:
    """Find up-migrations, sorted by version. `.down.sql` files are paired."""
    directory = directory or MIGRATIONS_DIR
    if not directory.is_dir():
        raise MigrationError(f"migrations directory not found: {directory}")

    migrations: list[Migration] = []
    seen: dict[str, str] = {}
    for path in sorted(directory.glob("*.sql")):
        if path.name.endswith(".down.sql"):
            continue
        match = _NAME.match(path.name)
        if match is None:
            raise MigrationError(
                f"bad migration filename {path.name!r}: expected NNN_name.sql "
                "(three or more digits, then an underscore or dash)"
            )
        version = match.group("version")
        if version in seen:
            raise MigrationError(
                f"duplicate migration version {version}: {seen[version]} and {path.name}"
            )
        seen[version] = path.name
        template = path.read_text(encoding="utf-8")
        down_path = path.with_name(f"{path.stem}.down.sql")
        migrations.append(
            Migration(
                version=version,
                name=match.group("name"),
                path=path,
                template=template,
                checksum=checksum(template),
                down_path=down_path if down_path.exists() else None,
                down_template=(
                    down_path.read_text(encoding="utf-8") if down_path.exists() else None
                ),
                requires=parse_requires(template),
            )
        )
    migrations.sort(key=lambda m: m.version)

    known = {migration.version for migration in migrations}
    for migration in migrations:
        for required in migration.requires:
            if required not in known:
                raise MigrationError(
                    f"{migration.path.name} requires {required}, which does not exist"
                )
            if required >= migration.version:
                raise MigrationError(
                    f"{migration.path.name} requires {required}, but a migration can only "
                    "depend on an earlier one"
                )
    return migrations


# ── planning (pure, so it can be tested without a database) ────────────────
@dataclass(frozen=True, slots=True)
class AppliedMigration:
    version: str
    name: str
    checksum: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Action:
    kind: Literal["apply", "rebuild"]
    migration: Migration


@dataclass
class MigrationPlan:
    actions: list[Action] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.actions)


def plan_migrations(
    migrations: list[Migration],
    applied: Mapping[str, AppliedMigration],
    params: Mapping[str, Any],
) -> MigrationPlan:
    """Decide what to run. Raises on drift, and cascades a rebuild forward.

    A migration rebuilds when a parameter *it declares* changed, or when a
    migration it `-- requires:` rebuilds. Everything else is left alone: a
    width change to `001` must not drop the turn journal created by `002`.
    """
    plan = MigrationPlan()
    params = dict(params)
    rebuilding: set[str] = set()

    for migration in migrations:
        row = applied.get(migration.version)
        if row is None:
            plan.actions.append(Action("apply", migration))
            continue

        if row.checksum != migration.checksum:
            raise MigrationDriftError(
                f"migration {migration.version} ({migration.name}) changed after it was "
                f"applied: recorded {row.checksum[:12]}, file {migration.checksum[:12]}. "
                "Add a new migration instead of editing an applied one."
            )

        declared = {key: params[key] for key in migration.parameters if key in params}
        changed = {
            key: (row.params.get(key), value)
            for key, value in declared.items()
            if row.params.get(key) != value
        }
        # Only declared parameters count: rows written by older runners recorded
        # the whole settings dict, and that must not force a spurious rebuild.
        depends_on_rebuild = bool(rebuilding.intersection(migration.requires))

        if changed or depends_on_rebuild:
            if migration.down_path is None:
                reason = (
                    f"parameters {changed}"
                    if changed
                    else f"a migration it requires rebuilt ({sorted(rebuilding)})"
                )
                raise MigrationParamError(
                    f"migration {migration.version} must be rebuilt for {reason} but "
                    f"{migration.down_name} does not exist. Add the down file, or restore "
                    "the previous parameters."
                )
            plan.actions.append(Action("rebuild", migration))
            rebuilding.add(migration.version)
        else:
            plan.up_to_date.append(migration.version)

    known = {migration.version for migration in migrations}
    plan.missing = sorted(set(applied) - known)
    return plan


@dataclass
class MigrationResult:
    applied: list[str] = field(default_factory=list)
    rebuilt: list[str] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.applied or self.rebuilt)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "rebuilt": self.rebuilt,
            "up_to_date": self.up_to_date,
            "missing": self.missing,
        }


# ── execution ──────────────────────────────────────────────────────────────
async def _read_applied(conn: asyncpg.Connection) -> dict[str, AppliedMigration]:
    rows = await conn.fetch("SELECT version, name, checksum, params FROM schema_migrations")
    applied: dict[str, AppliedMigration] = {}
    for row in rows:
        raw = row["params"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        applied[row["version"]] = AppliedMigration(
            version=row["version"],
            name=row["name"],
            checksum=row["checksum"],
            params=dict(raw or {}),
        )
    return applied


async def _apply(conn: asyncpg.Connection, migration: Migration, params: Mapping[str, Any]) -> None:
    # No bind parameters, so asyncpg uses the simple protocol and the file may
    # contain several statements (extension, tables, indexes) in one round trip.
    await conn.execute(migration.render_up(params))
    # Record only what this file declares. Storing the whole settings dict makes
    # every migration look parameterised, so an unrelated change rebuilds it.
    recorded = {key: params[key] for key in migration.parameters if key in params}
    await conn.execute(
        """
        INSERT INTO schema_migrations (version, name, checksum, params)
        VALUES ($1, $2, $3, $4::jsonb)
        ON CONFLICT (version) DO UPDATE SET
            name = EXCLUDED.name,
            checksum = EXCLUDED.checksum,
            params = EXCLUDED.params,
            applied_at = now()
        """,
        migration.version,
        migration.name,
        migration.checksum,
        json.dumps(recorded),
    )


async def _rebuild(
    conn: asyncpg.Connection,
    migration: Migration,
    old_params: Mapping[str, Any],
    new_params: Mapping[str, Any],
) -> None:
    logger.warning(
        "migration %s rebuild: parameters changed %s -> %s",
        migration.version,
        dict(old_params),
        dict(new_params),
    )
    await conn.execute(migration.render_down(old_params))
    await _apply(conn, migration, new_params)


async def _column_type(conn: asyncpg.Connection, table: str, column: str) -> str | None:
    return await conn.fetchval(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = to_regclass($1) AND attname = $2",
        table,
        column,
    )


async def _enforce_parameters(
    conn: asyncpg.Connection,
    migrations: list[Migration],
    params: Mapping[str, Any],
) -> list[str]:
    """Make the live schema match the recorded parameters.

    `schema_migrations` records what the runner applied, but a table can predate
    the runner (the old `init.sql` bootstrap) or be changed out of band. In that
    case the row says `embedding_dim=256` while the column is still `vector(1536)`,
    and the first ingest fails on a width mismatch. So verify the postcondition,
    not just the bookkeeping: rebuild through the migration that owns the
    parameter. Returns every version that was rebuilt.
    """
    dim = params.get("embedding_dim")
    if dim is None:
        return []
    actual = await _column_type(conn, "kb_chunks", "embedding")
    if actual is None or actual == f"vector({dim})":
        return []

    for index, migration in enumerate(reversed(migrations)):
        if "embedding_dim" in migration.parameters and migration.down_path is not None:
            logger.warning(
                "kb_chunks.embedding is %s but embedding_dim=%s; rebuilding through %s",
                actual,
                dim,
                migration.version,
            )
            # Cascade exactly like the planner: only declared dependents rebuild.
            position = len(migrations) - 1 - index
            cascade = {migration.version}
            await _rebuild(conn, migration, params, params)
            for dependent in migrations[position + 1 :]:
                if not cascade.intersection(dependent.requires):
                    continue
                if dependent.down_path is None:
                    raise MigrationParamError(
                        f"{dependent.path.name} depends on {sorted(cascade)} and must be "
                        f"rebuilt, but {dependent.down_name} does not exist."
                    )
                await _rebuild(conn, dependent, params, params)
                cascade.add(dependent.version)
            return sorted(cascade)
    raise MigrationParamError(
        f"kb_chunks.embedding is {actual} but embedding_dim={dim}, and no migration "
        "with a down file owns that parameter. Add one."
    )


async def run_migrations(
    pool: asyncpg.Pool,
    directory: Path | None = None,
    *,
    params: Mapping[str, Any] | None = None,
) -> MigrationResult:
    """Apply pending migrations. Safe to call from every replica at boot."""
    params = dict(params or {})
    migrations = discover(directory)
    result = MigrationResult()

    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA_MIGRATIONS)
        # Session-level lock: released explicitly, and by the server if we die.
        await conn.execute("SELECT pg_advisory_lock(hashtext($1))", LOCK_KEY)
        try:
            applied = await _read_applied(conn)
            plan = plan_migrations(migrations, applied, params)
            result.up_to_date = plan.up_to_date
            result.missing = plan.missing

            if plan.missing:
                logger.warning(
                    "schema_migrations contains versions with no file on disk: %s",
                    ", ".join(plan.missing),
                )

            for action in plan.actions:
                migration = action.migration
                if action.kind == "rebuild":
                    recorded = applied[migration.version].params
                    await _rebuild(conn, migration, recorded, params)
                    result.rebuilt.append(migration.version)
                else:
                    await _apply(conn, migration, params)
                    result.applied.append(migration.version)

            repaired = await _enforce_parameters(conn, migrations, params)
            for version in repaired:
                if version not in result.rebuilt:
                    result.rebuilt.append(version)
        finally:
            await conn.execute("SELECT pg_advisory_unlock(hashtext($1))", LOCK_KEY)

    if result.changed:
        logger.info(
            "schema migrated applied=%s rebuilt=%s",
            result.applied or "none",
            result.rebuilt or "none",
        )
    return result


async def current_version(pool: asyncpg.Pool) -> str | None:
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT max(version) FROM schema_migrations")


# ── operator entry point ───────────────────────────────────────────────────
def main() -> int:
    from app.config import Settings

    parser = argparse.ArgumentParser(description="Apply cuePilot SQL migrations.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--dir", default=None, help=f"default: {MIGRATIONS_DIR}")
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="embedding dimension (default: CUEPILOT_EMBEDDING_DIM)",
    )
    args = parser.parse_args()

    settings = Settings()
    database_url = args.database_url or settings.database_url
    dim = args.dim if args.dim is not None else settings.embedding_dim
    directory = Path(args.dir) if args.dir else None

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    async def run() -> MigrationResult:
        pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
        assert pool is not None
        try:
            return await run_migrations(pool, directory, params={"embedding_dim": dim})
        finally:
            await pool.close()

    try:
        result = asyncio.run(run())
    except MigrationError as exc:
        print(f"migration failed: {exc}")
        return 1
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
