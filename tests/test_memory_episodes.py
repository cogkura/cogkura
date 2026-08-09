"""Unit tests for Memory episodic APIs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import Memory, ObservationInput


def _observation(
    *,
    source_record_id: str,
    content: str,
    metadata: dict | None = None,
) -> ObservationInput:
    return ObservationInput(
        tenant_id="company_123",
        subject_id="customer_42",
        actor_id="customer_42",
        source_namespace="direct",
        source_record_id=source_record_id,
        event_type="message",
        content=content,
        observed_at=datetime.now(UTC),
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_encode_and_list_episodes() -> None:
    memory = Memory()
    await memory.observe(
        _observation(
            source_record_id="message_1",
            content="Redis would add too much operational complexity.",
            metadata={
                "conversation_id": "architecture_123",
                "entity_ids": ["customer_42", "redis"],
            },
        )
    )
    await memory.observe(
        _observation(
            source_record_id="message_2",
            content="The team agreed to evaluate PostgreSQL-backed caching.",
            metadata={
                "conversation_id": "architecture_123",
                "entity_ids": ["customer_42", "postgresql"],
                "terminal_event": True,
            },
        )
    )

    result = await memory.encode_episodes(
        tenant_id="company_123",
        subject_id="customer_42",
    )
    episodes = await memory.list_episodes(
        tenant_id="company_123",
        subject_id="customer_42",
    )

    assert result.created == 1
    assert len(episodes) == 1
    assert len(episodes[0].evidence) == 2


@pytest.mark.asyncio
async def test_encode_episodes_is_idempotent() -> None:
    memory = Memory()
    await memory.observe(
        _observation(
            source_record_id="message_1",
            content="A stable observation for episodic encoding.",
            metadata={"conversation_id": "conv-1"},
        )
    )
    first = await memory.encode_episodes(tenant_id="company_123")
    second = await memory.encode_episodes(tenant_id="company_123")
    assert first.created == 1
    assert second.unchanged == 1


@pytest.mark.asyncio
async def test_clear_removes_episodes_and_observations() -> None:
    memory = Memory()
    await memory.observe(
        _observation(
            source_record_id="message_1",
            content="Temporary observation for clear test.",
            metadata={"conversation_id": "conv-1"},
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    await memory.clear(tenant_id="company_123")
    assert await memory.list_episodes(tenant_id="company_123") == []
