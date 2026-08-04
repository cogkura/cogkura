"""PostgreSQL schema migration helpers."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

MIGRATION_VERSION = "001_initial"


def _migration_sql() -> str:
    path = Path(__file__).parent / "postgres" / "001_initial.sql"
    return path.read_text(encoding="utf-8")


async def apply_migrations(engine: AsyncEngine, *, schema: str = "cognema") -> None:
    """Apply pending Cognema PostgreSQL migrations."""
    if schema != "cognema":
        raise ValueError("Only the cognema schema is supported in this release.")
    sql = _migration_sql()
    async with engine.begin() as conn:
        applied = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                      AND table_name = 'schema_migrations'
                )
                """
            ),
            {"schema": schema},
        )
        if applied.scalar():
            version = await conn.execute(
                text(f"SELECT version FROM {schema}.schema_migrations WHERE version = :version"),
                {"version": MIGRATION_VERSION},
            )
            if version.scalar() is not None:
                return

        schema_exists = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.schemata
                    WHERE schema_name = :schema
                )
                """
            ),
            {"schema": schema},
        )
        if not schema_exists.scalar():
            await conn.execute(text("CREATE SCHEMA cognema"))

        for statement in _split_statements(sql):
            await conn.execute(text(statement))
        await conn.execute(
            text(
                f"""
                INSERT INTO {schema}.schema_migrations (version)
                VALUES (:version)
                ON CONFLICT (version) DO NOTHING
                """
            ),
            {"version": MIGRATION_VERSION},
        )


def _split_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current))
            current = []
    if current:
        statements.append("\n".join(current))
    return statements


def migration_files() -> list[Path]:
    """Return packaged migration SQL paths."""
    base = Path(__file__).parent / "postgres"
    return sorted(base.glob("*.sql"))
