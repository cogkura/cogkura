"""Unit tests for in-memory activation store."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura.models import ActivationReferenceKind, MemoryIdentity, MemoryKind, MemoryReference
from cogkura.storage.in_memory_activation import InMemoryActivationStore


def _reference(
    *,
    memory_key: str = "episode-key",
    request_id: str | None = None,
    referenced_at: datetime | None = None,
    weight: int = 1,
) -> MemoryReference:
    return MemoryReference(
        tenant_id="company_123",
        memory_kind=MemoryKind.EPISODE,
        memory_key=memory_key,
        reference_kind=ActivationReferenceKind.RETRIEVED,
        referenced_at=referenced_at or datetime(2026, 8, 7, 10, 0, tzinfo=UTC),
        request_id=request_id,
        weight=weight,
    )


@pytest.mark.asyncio
async def test_append_and_list_reference_traces() -> None:
    store = InMemoryActivationStore()
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-key")
    await store.append_references([_reference(memory_key="episode-key")])

    traces = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=datetime(2026, 8, 7, 12, 0, tzinfo=UTC),
    )
    assert identity in traces
    assert len(traces[identity]) == 1
    assert traces[identity][0].weight == 1


@pytest.mark.asyncio
async def test_request_id_is_idempotent() -> None:
    store = InMemoryActivationStore()
    reference = _reference(request_id="req-1")
    await store.append_references([reference])
    await store.append_references([reference])

    identity = reference.identity
    traces = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=datetime(2026, 8, 7, 12, 0, tzinfo=UTC),
    )
    assert len(traces[identity]) == 1


@pytest.mark.asyncio
async def test_clear_removes_tenant_references() -> None:
    store = InMemoryActivationStore()
    await store.append_references([_reference()])
    await store.clear(tenant_id="company_123")
    traces = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-key")],
        before_or_at=datetime(2026, 8, 7, 12, 0, tzinfo=UTC),
    )
    assert traces == {}
