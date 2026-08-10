"""PostgreSQL activation reference store tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura.migrations import apply_migrations
from cogkura.models import ActivationReferenceKind, MemoryIdentity, MemoryKind, MemoryReference
from cogkura.storage.postgres import PostgresActivationStore

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


@pytest.mark.asyncio
async def test_postgres_activation_append_and_list(memory_engine: AsyncEngine) -> None:
    store = PostgresActivationStore(memory_engine)
    reference = MemoryReference(
        tenant_id="company_123",
        memory_kind=MemoryKind.EPISODE,
        memory_key="episode-key",
        reference_kind=ActivationReferenceKind.RETRIEVED,
        referenced_at=datetime(2026, 8, 7, 10, 0, tzinfo=UTC),
        request_id="req-1",
    )
    await store.append_references([reference])
    await store.append_references([reference])

    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-key")
    traces = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=datetime(2026, 8, 7, 12, 0, tzinfo=UTC),
    )
    assert len(traces[identity]) == 1
