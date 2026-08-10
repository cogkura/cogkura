"""Migration runner tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura.migrations import apply_migrations, migration_files

pytestmark = pytest.mark.postgres


def _migration_url() -> str | None:
    """Prefer a table-owner URL so ALTER migrations can run."""
    return os.environ.get("COGKURA_POSTGRES_MEMORY_ADMIN_URL") or os.environ.get(
        "COGKURA_POSTGRES_MEMORY_URL"
    )


@pytest.fixture
async def memory_engine() -> AsyncIterator[AsyncEngine]:
    url = _migration_url()
    if url is None:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    engine = create_async_engine(url)
    yield engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_migration_files_are_ordered() -> None:
    files = migration_files()
    assert [path.name for path in files] == [
        "001_initial.sql",
        "002_episodic_memory.sql",
        "003_semantic_consolidation.sql",
        "004_declarative_activation.sql",
        "005_forgetting_dynamics.sql",
    ]


@pytest.mark.asyncio
async def test_apply_migrations_is_idempotent(memory_engine: AsyncEngine) -> None:
    await apply_migrations(memory_engine)
    await apply_migrations(memory_engine)
    async with memory_engine.connect() as conn:
        result = await conn.execute(
            text("SELECT version FROM cogkura.schema_migrations ORDER BY version")
        )
        versions = [row[0] for row in result.all()]
    assert versions == [
        "001_initial",
        "002_episodic_memory",
        "003_semantic_consolidation",
        "004_declarative_activation",
        "005_forgetting_dynamics",
    ]


@pytest.mark.asyncio
async def test_episodic_schema_objects_exist(memory_engine: AsyncEngine) -> None:
    await apply_migrations(memory_engine)
    async with memory_engine.connect() as conn:
        columns = await conn.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'cogkura'
                  AND table_name = 'observations'
                  AND column_name IN (
                      'attention_score', 'retention_class', 'policy_reasons'
                  )
                """
            )
        )
        assert {row[0] for row in columns.all()} == {
            "attention_score",
            "retention_class",
            "policy_reasons",
        }
        entities = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'memory_entities'
                )
                """
            )
        )
        assert entities.scalar() is True


@pytest.mark.asyncio
async def test_semantic_schema_objects_exist(memory_engine: AsyncEngine) -> None:
    await apply_migrations(memory_engine)
    async with memory_engine.connect() as conn:
        claims = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'semantic_claims'
                )
                """
            )
        )
        derivations = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'memory_derivations'
                )
                """
            )
        )
        tenant_id_index = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_indexes
                    WHERE schemaname = 'cogkura'
                      AND indexname = 'memories_tenant_id_memory_id_idx'
                )
                """
            )
        )
        assert claims.scalar() is True
        assert derivations.scalar() is True
        assert tenant_id_index.scalar() is True


@pytest.mark.asyncio
async def test_activation_schema_objects_exist(memory_engine: AsyncEngine) -> None:
    await apply_migrations(memory_engine)
    async with memory_engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'memory_activation_references'
                )
                """
            )
        )
        assert result.scalar() is True


@pytest.mark.asyncio
async def test_forgetting_schema_objects_exist(memory_engine: AsyncEngine) -> None:
    await apply_migrations(memory_engine)
    async with memory_engine.connect() as conn:
        dynamics = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'memory_dynamics'
                )
                """
            )
        )
        weight_column = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'memory_activation_references'
                      AND column_name = 'weight'
                )
                """
            )
        )
        assert dynamics.scalar() is True
        assert weight_column.scalar() is True


@pytest.mark.asyncio
async def test_upgrade_from_001_only_applies_002(memory_engine: AsyncEngine) -> None:
    """Simulate a DB that has 001 recorded but not yet 002."""
    admin_url = os.environ.get("COGKURA_POSTGRES_MEMORY_ADMIN_URL")
    if admin_url is None:
        pytest.skip("COGKURA_POSTGRES_MEMORY_ADMIN_URL required for upgrade simulation")

    await apply_migrations(memory_engine)
    async with memory_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM cogkura.schema_migrations WHERE version = '002_episodic_memory'")
        )
        await conn.execute(text("DROP TABLE IF EXISTS cogkura.memory_entities CASCADE"))
        await conn.execute(
            text(
                """
                ALTER TABLE cogkura.observations
                DROP COLUMN IF EXISTS attention_score
                """
            )
        )

    await apply_migrations(memory_engine)

    async with memory_engine.connect() as conn:
        versions = await conn.execute(
            text("SELECT version FROM cogkura.schema_migrations ORDER BY version")
        )
        assert [row[0] for row in versions.all()] == ["001_initial", "002_episodic_memory"]
        attention = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'observations'
                      AND column_name = 'attention_score'
                )
                """
            )
        )
        assert attention.scalar() is True
        entities = await conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'cogkura'
                      AND table_name = 'memory_entities'
                )
                """
            )
        )
        assert entities.scalar() is True
