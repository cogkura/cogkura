"""Unit tests for in-memory semantic memory store."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura.models import (
    EpisodeEvidenceInput,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticWriteStatus,
)
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore


def _memory(
    *,
    memory_key: str = "semantic-key-1",
    fingerprint: str = "fp-1",
    subject_id: str | None = "user_1",
    episode_id: str = "ep-1",
) -> SemanticMemoryInput:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    return SemanticMemoryInput(
        tenant_id="company_123",
        subject_id=subject_id,
        memory_key=memory_key,
        slot_key="slot-1",
        revision_key=f"legacy:{memory_key}",
        revision_number=1,
        statement="Customer prefers PostgreSQL in production.",
        subject_entity_id="user_1",
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
        derivations=(
            SemanticDerivationInput(
                episode_id=episode_id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
        observation_evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-1",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
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
async def test_create_update_and_unchanged() -> None:
    store = InMemorySemanticMemoryStore()
    memory = _memory()
    status = await store.upsert(memory)
    assert status is SemanticWriteStatus.CREATED

    status = await store.upsert(memory)
    assert status is SemanticWriteStatus.UNCHANGED

    updated = _memory(fingerprint="fp-2", memory_key="semantic-key-1")
    status = await store.upsert(updated)
    assert status is SemanticWriteStatus.UPDATED


@pytest.mark.asyncio
async def test_list_filters_by_status() -> None:
    store = InMemorySemanticMemoryStore()
    await store.upsert(_memory(memory_key="k1"))
    contested_memory = SemanticMemoryInput(
        tenant_id="company_123",
        subject_id="user_1",
        memory_key="k2",
        slot_key="slot-1",
        revision_key="legacy:k2",
        revision_number=1,
        statement="Contested claim.",
        subject_entity_id="user_1",
        predicate="preferred_database",
        object_value="mysql",
        object_entity_id="mysql",
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers=MappingProxyType({}),
        confidence=0.6,
        importance=0.5,
        status=SemanticMemoryStatus.CONTESTED,
        support_count=2,
        contradiction_count=1,
        first_supported_at=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        last_supported_at=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        derivations=(),
        observation_evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-2",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        metadata=MappingProxyType(
            {"semantic": {"content_fingerprint": "fp-2", "consolidation_version": "v1"}}
        ),
    )
    await store.upsert(contested_memory)

    active = await store.list(tenant_id="company_123", status=SemanticMemoryStatus.ACTIVE)
    assert len(active) == 1
    assert active[0].memory_key == "k1"


@pytest.mark.asyncio
async def test_deactivate_missing() -> None:
    store = InMemorySemanticMemoryStore()
    await store.upsert(_memory(memory_key="k1"))
    await store.upsert(_memory(memory_key="k2", fingerprint="fp-2"))

    deactivated = await store.deactivate_missing(
        tenant_id="company_123",
        subject_id=None,
        active_memory_keys={"k1"},
    )
    assert deactivated == 1
    active = await store.list(tenant_id="company_123")
    assert len(active) == 1
    assert active[0].memory_key == "k1"


@pytest.mark.asyncio
async def test_clear_removes_tenant_memories() -> None:
    store = InMemorySemanticMemoryStore()
    await store.upsert(_memory())
    await store.clear(tenant_id="company_123")
    assert await store.list(tenant_id="company_123") == []
