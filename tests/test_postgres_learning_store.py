"""PostgreSQL learning store tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura import Memory
from cogkura.algorithms.learning import DeterministicLearningProcessor
from cogkura.migrations import apply_migrations
from cogkura.models import (
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
)
from cogkura.storage.in_memory_learning import InMemoryLearningStore
from cogkura.storage.postgres import PostgresLearningStore

pytestmark = pytest.mark.postgres

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
async def memory_engine() -> AsyncIterator[AsyncEngine]:
    url = os.environ.get("COGKURA_POSTGRES_MEMORY_URL")
    if url is None:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    engine = create_async_engine(url)
    await apply_migrations(engine)
    yield engine
    await engine.dispose()


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
            MemoryFeedback(
                identity=MemoryIdentity(
                    memory_kind=MemoryKind.EPISODE,
                    memory_key="episode-b",
                ),
                outcome=LearningOutcome.HELPFUL,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_postgres_learning_store_matches_in_memory(memory_engine: AsyncEngine) -> None:
    processor = DeterministicLearningProcessor()
    config = LearningConfig()
    feedback = _feedback()
    in_memory = InMemoryLearningStore()
    postgres = PostgresLearningStore(memory_engine)
    plan = processor.plan(feedback=feedback, config=config)

    in_memory_result = await in_memory.apply(plan)
    postgres_result = await postgres.apply(plan)
    assert in_memory_result == postgres_result

    identity_a = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    identity_b = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-b")
    in_memory_states = await in_memory.list_states(
        tenant_id="company_123",
        identities=[identity_a, identity_b],
        context_keys=("global",),
    )
    postgres_states = await postgres.list_states(
        tenant_id="company_123",
        identities=[identity_a, identity_b],
        context_keys=("global",),
    )
    assert len(in_memory_states) == len(postgres_states) == 2

    in_memory_associations = await in_memory.list_associations(
        tenant_id="company_123",
        identities=[identity_a, identity_b],
    )
    postgres_associations = await postgres.list_associations(
        tenant_id="company_123",
        identities=[identity_a, identity_b],
    )
    assert len(in_memory_associations) == len(postgres_associations) == 1

    await postgres.clear(tenant_id="company_123")


@pytest.mark.asyncio
async def test_clear_removes_all_learning_state_only_for_requested_tenant(
    memory_engine: AsyncEngine,
) -> None:
    store = PostgresLearningStore(memory_engine)
    processor = DeterministicLearningProcessor()
    feedback = _feedback(feedback_id="clear-regression")
    other = replace(feedback, tenant_id="other-clear-regression")
    identities = [item.identity for item in feedback.items]
    for tenant in (feedback.tenant_id, other.tenant_id):
        await store.clear(tenant_id=tenant)
    try:
        for item in (feedback, other):
            await store.apply(processor.plan(feedback=item, config=LearningConfig()))
        await Memory(learning_store=store).clear(tenant_id=feedback.tenant_id)
        assert not await store.list_states(
            tenant_id=feedback.tenant_id, identities=identities, context_keys=("global",)
        )
        assert not await store.list_associations(
            tenant_id=feedback.tenant_id, identities=identities
        )
        assert (
            len(
                await store.list_states(
                    tenant_id=other.tenant_id, identities=identities, context_keys=("global",)
                )
            )
            == 2
        )
        assert (
            len(await store.list_associations(tenant_id=other.tenant_id, identities=identities))
            == 1
        )
        assert (
            await store.apply(processor.plan(feedback=feedback, config=LearningConfig()))
        ).created
        states = await store.list_states(
            tenant_id=feedback.tenant_id, identities=identities, context_keys=("global",)
        )
        assert [state.helpful_count for state in states] == [1, 1]
    finally:
        for tenant in (feedback.tenant_id, other.tenant_id):
            await store.clear(tenant_id=tenant)
