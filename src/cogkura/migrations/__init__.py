"""PostgreSQL schema migration helpers."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


async def apply_migrations(engine: AsyncEngine, *, schema: str = "cogkura") -> None:
    """Apply pending Cogkura PostgreSQL migrations."""
    if schema != "cogkura":
        raise ValueError("Only the cogkura schema is supported in this release.")

    async with engine.begin() as conn:
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
            await conn.execute(text("CREATE SCHEMA cogkura"))

        for path in migration_files():
            version = path.stem
            if await _is_applied(conn, schema=schema, version=version):
                continue
            sql = path.read_text(encoding="utf-8")
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
                {"version": version},
            )


async def _is_applied(conn: AsyncConnection, *, schema: str, version: str) -> bool:
    table_exists = await conn.execute(
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
    if not table_exists.scalar():
        return False
    result = await conn.execute(
        text(f"SELECT 1 FROM {schema}.schema_migrations WHERE version = :version"),
        {"version": version},
    )
    return result.scalar() is not None


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
