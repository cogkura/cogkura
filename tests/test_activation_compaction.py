"""Tests for activation reference compaction."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from cogkura.algorithms.activation import calculate_base_level
from cogkura.models import (
    ActivationReferenceKind,
    MemoryKind,
    MemoryReference,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _reference(*, referenced_at: datetime, weight: int = 1) -> MemoryReference:
    return MemoryReference(
        tenant_id="company_123",
        memory_kind=MemoryKind.EPISODE,
        memory_key="episode-key",
        reference_kind=ActivationReferenceKind.RETRIEVED,
        referenced_at=referenced_at,
        weight=weight,
    )


@pytest.mark.asyncio
async def test_same_timestamp_compaction_preserves_base_level() -> None:
    store = InMemoryActivationStore()
    same_time = _T0 + timedelta(seconds=10)
    references = [_reference(referenced_at=same_time) for _ in range(5)]
    await store.append_references(references)

    as_of = _T0 + timedelta(hours=1)
    identity = references[0].identity
    before = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=as_of,
    )
    before_level = calculate_base_level(
        before[identity],
        as_of=as_of,
        decay=0.5,
        constant=0.0,
        time_unit_seconds=1.0,
        minimum_elapsed_seconds=1.0,
    )

    result = await store.compact_references(
        tenant_id="company_123",
        before=as_of,
        bucket_seconds=86_400.0,
    )
    assert result.references_compacted == 5

    after = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=as_of,
    )
    after_level = calculate_base_level(
        after[identity],
        as_of=as_of,
        decay=0.5,
        constant=0.0,
        time_unit_seconds=1.0,
        minimum_elapsed_seconds=1.0,
    )
    assert math.isclose(before_level, after_level, abs_tol=1e-6)


@pytest.mark.asyncio
async def test_day_bucket_compaction_within_engineering_tolerance() -> None:
    store = InMemoryActivationStore()
    day = _T0
    references = [
        _reference(referenced_at=day + timedelta(hours=offset)) for offset in (1, 5, 9, 14, 20)
    ]
    await store.append_references(references)

    as_of = day + timedelta(days=2)
    identity = references[0].identity
    before = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=as_of,
    )
    before_level = calculate_base_level(
        before[identity],
        as_of=as_of,
        decay=0.5,
        constant=0.0,
        time_unit_seconds=3600.0,
        minimum_elapsed_seconds=1.0,
    )

    await store.compact_references(
        tenant_id="company_123",
        before=as_of,
        bucket_seconds=86_400.0,
    )

    after = await store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=as_of,
    )
    after_level = calculate_base_level(
        after[identity],
        as_of=as_of,
        decay=0.5,
        constant=0.0,
        time_unit_seconds=3600.0,
        minimum_elapsed_seconds=1.0,
    )
    relative_delta = abs(after_level - before_level) / max(abs(before_level), 1e-9)
    assert relative_delta < 0.05
