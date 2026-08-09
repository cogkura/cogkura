"""PostgreSQL semantic memory store tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import MappingProxyType

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura.migrations import apply_migrations
from cogkura.models import (
    SemanticCardinality,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticWriteStatus,
)
from cogkura.storage.postgres import PostgresSemanticMemoryStore

pytestmark = pytest.mark.postgres


@pytest.fixture
async def memory_engine() -> AsyncIterator[AsyncEngine]:
    import os

    url = os.environ.get("COGKURA_POSTGRES_MEMORY_URL")
    if url is None:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    engine = create_async_engine(url)
    await apply_migrations(engine)
    yield engine
    await engine.dispose()


def _memory(
    *,
    memory_key: str = "semantic-key-1",
    fingerprint: str = "fp-1",
) -> SemanticMemoryInput:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    return SemanticMemoryInput(
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        slot_key="slot-1",
        statement="Customer prefers PostgreSQL in production.",
        subject_entity_id="customer_42",
        predicate="preferred_database",
        object_value="postgresql",
        object_entity_id="postgresql",
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers=MappingProxyType({"environment": "production"}),
        confidence=0.9,
        importance=0.7,
        status=SemanticMemoryStatus.ACTIVE,
        support_count=2,
        contradiction_count=0,
        first_supported_at=now,
        last_supported_at=now,
        derivations=(),
        observation_evidence=(),
        metadata=MappingProxyType(
            {
                "semantic": {
                    "content_fingerprint": fingerprint,
                    "consolidation_version": "cls-deterministic-v1",
                }
            }
        ),
    )


@pytest.mark.asyncio
async def test_postgres_semantic_create_update_unchanged(memory_engine: AsyncEngine) -> None:
    store = PostgresSemanticMemoryStore(memory_engine)
    memory = _memory()
    status = await store.upsert(memory)
    assert status is SemanticWriteStatus.CREATED

    status = await store.upsert(memory)
    assert status is SemanticWriteStatus.UNCHANGED

    updated = _memory(fingerprint="fp-2")
    status = await store.upsert(updated)
    assert status is SemanticWriteStatus.UPDATED

    listed = await store.list(tenant_id="company_123", subject_id="customer_42")
    assert len(listed) == 1
    assert listed[0].predicate == "preferred_database"
    assert listed[0].support_count == 2


@pytest.mark.asyncio
async def test_postgres_semantic_deactivate_missing(memory_engine: AsyncEngine) -> None:
    store = PostgresSemanticMemoryStore(memory_engine)
    await store.upsert(_memory(memory_key="k1"))
    await store.upsert(_memory(memory_key="k2", fingerprint="fp-2"))

    deactivated = await store.deactivate_missing(
        tenant_id="company_123",
        subject_id="customer_42",
        active_memory_keys={"k1"},
    )
    assert deactivated == 1
    active = await store.list(tenant_id="company_123", subject_id="customer_42")
    assert len(active) == 1
    assert active[0].memory_key == "k1"
