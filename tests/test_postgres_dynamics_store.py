"""PostgreSQL memory dynamics store tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura.migrations import apply_migrations
from cogkura.models import (
    ActivationReferenceKind,
    MemoryIdentity,
    MemoryKind,
    MemoryReference,
    MemoryRetentionState,
    StoredMemoryDynamics,
)
from cogkura.storage.postgres import PostgresActivationStore, PostgresMemoryDynamicsStore

pytestmark = pytest.mark.postgres

_T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


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


@pytest.mark.asyncio
async def test_postgres_dynamics_upsert_get_and_reactivate(memory_engine: AsyncEngine) -> None:
    store = PostgresMemoryDynamicsStore(memory_engine)
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-dynamics")
    dynamics = StoredMemoryDynamics(
        tenant_id="company_123",
        memory_kind=identity.memory_kind,
        memory_key=identity.memory_key,
        retention_state=MemoryRetentionState.FORGOTTEN,
        last_base_level=-4.0,
        last_retention_score=0.02,
        below_threshold_since=_T0 - timedelta(days=3),
        forgotten_at=_T0 - timedelta(days=1),
        evaluated_at=_T0,
        updated_at=_T0,
    )
    await store.upsert_many([dynamics])
    loaded = await store.get_many(tenant_id="company_123", identities=[identity])
    assert loaded[identity].retention_state is MemoryRetentionState.FORGOTTEN

    await store.reactivate(
        tenant_id="company_123",
        identities=[identity],
        at=_T0 + timedelta(hours=1),
    )
    reloaded = await store.get_many(tenant_id="company_123", identities=[identity])
    assert reloaded[identity].retention_state is MemoryRetentionState.ACTIVE
    assert reloaded[identity].forgotten_at is None

    await store.clear(tenant_id="company_123")
    cleared = await store.get_many(tenant_id="company_123", identities=[identity])
    assert identity not in cleared


@pytest.mark.asyncio
async def test_postgres_activation_compaction(memory_engine: AsyncEngine) -> None:
    store = PostgresActivationStore(memory_engine)
    tenant_id = "company_123"
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-compact")
    same_time = _T0 + timedelta(seconds=10)
    references = [
        MemoryReference(
            tenant_id=tenant_id,
            memory_kind=identity.memory_kind,
            memory_key=identity.memory_key,
            reference_kind=ActivationReferenceKind.RETRIEVED,
            referenced_at=same_time,
            request_id=f"req-{index}",
        )
        for index in range(5)
    ]
    await store.append_references(references)

    as_of = _T0 + timedelta(hours=1)
    result = await store.compact_references(
        tenant_id=tenant_id,
        before=as_of,
        bucket_seconds=86_400.0,
    )
    assert result.references_compacted == 5

    traces = await store.list_reference_traces(
        tenant_id=tenant_id,
        identities=[identity],
        before_or_at=as_of,
    )
    assert len(traces[identity]) == 1
    assert traces[identity][0].weight == 5

    await store.clear(tenant_id=tenant_id)
