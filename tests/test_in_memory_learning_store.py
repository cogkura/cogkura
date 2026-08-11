"""In-memory learning store tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura.algorithms.learning import DeterministicLearningProcessor
from cogkura.exceptions import StorageError
from cogkura.models import (
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
)
from cogkura.storage.in_memory_learning import InMemoryLearningStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _feedback(*, feedback_id: str = "feedback-1") -> LearningFeedback:
    return LearningFeedback(
        tenant_id="company_123",
        feedback_id=feedback_id,
        occurred_at=_T0,
        items=(
            MemoryFeedback(
                identity=MemoryIdentity(
                    memory_kind=MemoryKind.EPISODE,
                    memory_key="episode-a",
                ),
                outcome=LearningOutcome.HELPFUL,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_apply_is_idempotent() -> None:
    store = InMemoryLearningStore()
    processor = DeterministicLearningProcessor()
    config = LearningConfig()
    plan = processor.plan(feedback=_feedback(), config=config)
    first = await store.apply(plan)
    second = await store.apply(plan)
    assert first.created is True
    assert second.unchanged is True


@pytest.mark.asyncio
async def test_conflicting_fingerprint_raises_storage_error() -> None:
    store = InMemoryLearningStore()
    processor = DeterministicLearningProcessor()
    config = LearningConfig()
    await store.apply(processor.plan(feedback=_feedback(), config=config))
    conflicting = LearningFeedback(
        tenant_id="company_123",
        feedback_id="feedback-1",
        occurred_at=_T0,
        items=(
            MemoryFeedback(
                identity=MemoryIdentity(
                    memory_kind=MemoryKind.EPISODE,
                    memory_key="episode-a",
                ),
                outcome=LearningOutcome.INCORRECT,
            ),
        ),
    )
    with pytest.raises(StorageError):
        await store.apply(processor.plan(feedback=conflicting, config=config))


@pytest.mark.asyncio
async def test_clear_is_tenant_scoped() -> None:
    store = InMemoryLearningStore()
    processor = DeterministicLearningProcessor()
    config = LearningConfig()
    await store.apply(processor.plan(feedback=_feedback(), config=config))
    await store.apply(
        processor.plan(
            feedback=LearningFeedback(
                tenant_id="other",
                feedback_id="feedback-1",
                occurred_at=_T0,
                items=_feedback().items,
            ),
            config=config,
        )
    )
    await store.clear(tenant_id="company_123")
    states = await store.list_states(
        tenant_id="company_123",
        identities=[MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")],
        context_keys=("global",),
    )
    other_states = await store.list_states(
        tenant_id="other",
        identities=[MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")],
        context_keys=("global",),
    )
    assert states == ()
    assert len(other_states) == 1
