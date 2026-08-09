"""Unit tests for Memory semantic consolidation APIs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import Memory, ObservationInput


def _observation(
    *,
    source_record_id: str,
    conversation_id: str,
    semantic_fact: dict,
) -> ObservationInput:
    return ObservationInput(
        tenant_id="company_123",
        subject_id="customer_42",
        actor_id="customer_42",
        source_namespace="chat.messages",
        source_record_id=source_record_id,
        event_type="message",
        content="Database preference discussion.",
        observed_at=datetime.now(UTC),
        metadata={
            "conversation_id": conversation_id,
            "entity_ids": ["customer_42"],
            "semantic_facts": [semantic_fact],
        },
    )


_SEMANTIC_FACT = {
    "predicate": "preferred_database",
    "object_value": "postgresql",
    "object_entity_id": "postgresql",
    "cardinality": "one",
    "polarity": "affirm",
    "qualifiers": {"environment": "production"},
}


@pytest.mark.asyncio
async def test_consolidate_and_list_semantic_memories() -> None:
    memory = Memory()
    await memory.observe(
        _observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.observe(
        _observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.encode_episodes(tenant_id="company_123", subject_id="customer_42")

    result = await memory.consolidate_semantics(
        tenant_id="company_123",
        subject_id="customer_42",
    )
    memories = await memory.list_semantic_memories(
        tenant_id="company_123",
        subject_id="customer_42",
    )

    assert result.promoted == 1
    assert result.created == 1
    assert len(memories) == 1
    assert memories[0].predicate == "preferred_database"
    assert memories[0].support_count == 2


@pytest.mark.asyncio
async def test_consolidate_semantics_is_idempotent() -> None:
    memory = Memory()
    await memory.observe(
        _observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.observe(
        _observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    first = await memory.consolidate_semantics(tenant_id="company_123")
    second = await memory.consolidate_semantics(tenant_id="company_123")
    assert first.created == 1
    assert second.unchanged == 1


@pytest.mark.asyncio
async def test_clear_removes_semantic_memories() -> None:
    memory = Memory()
    await memory.observe(
        _observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.observe(
        _observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    await memory.consolidate_semantics(tenant_id="company_123")
    await memory.clear(tenant_id="company_123")
    assert await memory.list_semantic_memories(tenant_id="company_123") == []
