"""Lifecycle tests ensuring encoding context is not clobbered."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    LearningFeedback,
    LearningOutcome,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
)
from cogkura.observations.encoding_context import ObservationContext

_T = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


def _observation(
    *,
    record_id: str,
    content: str,
    context: ObservationContext | None = None,
    metadata: dict | None = None,
    observed_at: datetime | None = None,
) -> ObservationInput:
    return ObservationInput(
        tenant_id="acme",
        subject_id="developer-1",
        source_namespace="github",
        source_record_id=record_id,
        content=content,
        observed_at=observed_at or _T,
        metadata=metadata or {},
        context=context,
    )


@pytest.mark.asyncio
async def test_reconsolidation_does_not_mutate_episode_encoding_context() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="1",
            content="We decided not to use Redis.",
            context=ObservationContext(
                goal="reduce-operational-complexity",
                domain="payments-api",
            ),
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("redis",),
                "semantic_facts": [
                    {
                        "predicate": "uses",
                        "object": "postgresql",
                        "subject": "payments-api",
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id="acme", as_of=_T)
    episodes = await memory.list_episodes(tenant_id="acme")
    original_signature = episodes[0].encoding_context.to_canonical_dict()

    await memory.observe(
        _observation(
            record_id="2",
            content="PostgreSQL remains the coordination store.",
            context=ObservationContext(goal="production-incident", domain="payments-api"),
            metadata={
                "conversation_id": "incident-1",
                "entity_ids": ("postgresql",),
                "semantic_facts": [
                    {
                        "predicate": "uses",
                        "object": "postgresql",
                        "subject": "payments-api",
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id="acme", as_of=_T.replace(hour=11))
    episodes_after = await memory.list_episodes(tenant_id="acme")
    original = next(ep for ep in episodes_after if ep.statement.startswith("We decided"))
    assert original.encoding_context.to_canonical_dict() == original_signature


@pytest.mark.asyncio
async def test_new_episode_receives_new_encoding_context() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="1",
            content="Architecture review started.",
            context=ObservationContext(activity="architecture-decision"),
        )
    )
    await memory.encode_episodes(tenant_id="acme", as_of=_T)
    await memory.observe(
        _observation(
            record_id="2",
            content="Production incident response started.",
            context=ObservationContext(activity="incident-response"),
            observed_at=_T.replace(hour=11),
        )
    )
    await memory.encode_episodes(tenant_id="acme", as_of=_T.replace(hour=11))
    episodes = await memory.list_episodes(tenant_id="acme")
    activities = {tuple(ep.encoding_context.activities) for ep in episodes}
    assert ("architecture-decision",) in activities
    assert ("incident-response",) in activities


@pytest.mark.asyncio
async def test_learn_and_forgetting_preserve_episode_encoding_context() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="1",
            content="Redis was rejected for operational complexity.",
            context=ObservationContext(domain="payments-api"),
            metadata={"entity_ids": ("redis",)},
        )
    )
    await memory.encode_episodes(tenant_id="acme", as_of=_T)
    episodes = await memory.list_episodes(tenant_id="acme")
    signature = episodes[0].encoding_context.to_canonical_dict()
    recall = await memory.recall("Redis operational complexity", tenant_id="acme")
    assert recall
    await memory.learn(
        LearningFeedback(
            feedback_id="fb-1",
            tenant_id="acme",
            occurred_at=_T.replace(hour=12),
            items=(
                MemoryFeedback(
                    identity=MemoryIdentity(
                        memory_kind=MemoryKind.EPISODE,
                        memory_key=episodes[0].memory_key,
                    ),
                    outcome=LearningOutcome.HELPFUL,
                ),
            ),
        )
    )
    await memory.apply_forgetting(tenant_id="acme", as_of=_T.replace(hour=12))
    episodes_after = await memory.list_episodes(tenant_id="acme")
    assert episodes_after[0].encoding_context.to_canonical_dict() == signature
