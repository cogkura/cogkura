"""Deterministic evaluation fixtures for declarative activation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cognema import Memory, ObservationInput
from cognema.models import ActivationConfig, MemoryKind, SemanticMemoryStatus


def _obs(
    *,
    record_id: str,
    content: str,
    conversation_id: str,
    semantic_facts: list[dict] | None = None,
) -> ObservationInput:
    metadata: dict = {"conversation_id": conversation_id}
    if semantic_facts:
        metadata["semantic_facts"] = semantic_facts
    return ObservationInput(
        tenant_id="company_123",
        subject_id="customer_42",
        source_namespace="direct",
        source_record_id=record_id,
        content=content,
        observed_at=datetime.now(UTC),
        metadata=metadata,
    )


_SEMANTIC_FACT = {
    "predicate": "preferred_database",
    "object_value": "postgresql",
    "object_entity_id": "postgresql",
    "cardinality": "one",
    "polarity": "affirm",
}


@pytest.mark.asyncio
async def test_recent_episode_beats_old_semantic_on_text_match() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    old_time = datetime(2020, 1, 1, tzinfo=UTC)
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="customer_42",
            source_namespace="direct",
            source_record_id="old-1",
            content="PostgreSQL is preferred.",
            observed_at=old_time,
            metadata={"conversation_id": "conv-old", "semantic_facts": [_SEMANTIC_FACT]},
        )
    )
    await memory.observe(
        _obs(
            record_id="new-1",
            content="PostgreSQL incident resolved today.",
            conversation_id="conv-new",
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    await memory.consolidate_semantics(tenant_id="company_123")

    results = await memory.recall("PostgreSQL incident", tenant_id="company_123", limit=5)
    assert results
    assert results[0].memory_kind is MemoryKind.EPISODE


@pytest.mark.asyncio
async def test_reinforcement_before_recall_increases_activation() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    await memory.observe(
        _obs(
            record_id="m1",
            content="Payment incident resolved.",
            conversation_id="conv-1",
        )
    )
    await memory.observe(
        _obs(
            record_id="m2",
            content="Payment incident follow-up.",
            conversation_id="conv-2",
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    as_of = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    first = await memory.recall(
        "payment incident",
        tenant_id="company_123",
        limit=1,
        as_of=as_of,
    )
    await memory.record_access(
        first,
        tenant_id="company_123",
        referenced_at=as_of - timedelta(minutes=5),
        request_id="eval-1",
    )
    second = await memory.recall(
        "payment incident",
        tenant_id="company_123",
        limit=1,
        as_of=as_of,
    )
    assert second[0].activation > first[0].activation


@pytest.mark.asyncio
async def test_contested_semantic_memory_is_retrievable() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    for index in range(3):
        await memory.observe(
            _obs(
                record_id=f"m-{index}",
                content="Database preference discussion.",
                conversation_id=f"conv-{index}",
                semantic_facts=[
                    _SEMANTIC_FACT
                    if index < 2
                    else {
                        **_SEMANTIC_FACT,
                        "object_value": "mysql",
                        "object_entity_id": "mysql",
                    }
                ],
            )
        )
    await memory.encode_episodes(tenant_id="company_123")
    await memory.consolidate_semantics(tenant_id="company_123")
    semantics = await memory.list_semantic_memories(tenant_id="company_123")
    contested = [item for item in semantics if item.status is SemanticMemoryStatus.CONTESTED]
    if contested:
        results = await memory.recall(
            "preferred database",
            tenant_id="company_123",
            semantic_statuses=frozenset({SemanticMemoryStatus.CONTESTED}),
        )
        assert results
        assert results[0].memory_kind is MemoryKind.SEMANTIC
