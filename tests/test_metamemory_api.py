"""API tests for read-only metamemory assessment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura import Memory
from cogkura.exceptions import ValidationError
from cogkura.models import (
    ActivationConfig,
    EpisodeEvidenceInput,
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    MemoryAssessmentFlag,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
    MetamemoryConfig,
    StoredEpisode,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_dynamics import InMemoryMemoryDynamicsStore
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore
from cogkura.storage.in_memory_learning import InMemoryLearningStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
_LOW_THRESHOLD = ActivationConfig(retrieval_threshold=-10.0)


def _episode(
    *,
    tenant_id: str = "company_123",
    subject_id: str | None = "customer_42",
    memory_key: str = "episode-a",
    statement: str = "PostgreSQL selected for production.",
    confidence: float = 0.9,
) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id=tenant_id,
        subject_id=subject_id,
        memory_key=memory_key,
        statement=statement,
        started_at=_T0,
        ended_at=_T0,
        confidence=confidence,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{memory_key}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({"episode": {"content_fingerprint": memory_key}}),
        created_at=_T0,
        updated_at=_T0,
    )


async def _memory_with_episode(**kwargs: object) -> Memory:
    episode = _episode(**kwargs)
    episode_store = InMemoryEpisodeStore()
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    return Memory(
        episode_store=episode_store,
        activation_config=_LOW_THRESHOLD,
    )


@pytest.mark.asyncio
async def test_assess_memory_disabled_raises() -> None:
    memory = Memory(metamemory_config=MetamemoryConfig(enabled=False))
    with pytest.raises(ValidationError, match="disabled"):
        await memory.assess_memory("query", tenant_id="company_123")


@pytest.mark.asyncio
async def test_empty_memory_assessment() -> None:
    memory = Memory(activation_config=_LOW_THRESHOLD)
    assessment = await memory.assess_memory(
        "missing query",
        tenant_id="company_123",
        as_of=_T1,
    )
    assert assessment.retrieved_count == 0
    assert assessment.flags == (MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,)


@pytest.mark.asyncio
async def test_assess_memory_no_activation_mutation() -> None:
    activation_store = InMemoryActivationStore()
    episode = _episode()
    episode_store = InMemoryEpisodeStore()
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        activation_store=activation_store,
        activation_config=_LOW_THRESHOLD,
    )
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    refs_before = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T1,
    )
    await memory.assess_memory(
        "PostgreSQL production",
        tenant_id="company_123",
        as_of=_T1,
    )
    refs_after = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T1,
    )
    assert refs_before == refs_after


@pytest.mark.asyncio
async def test_assess_memory_no_dynamics_mutation() -> None:
    dynamics_store = InMemoryMemoryDynamicsStore()
    episode = _episode()
    episode_store = InMemoryEpisodeStore()
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        dynamics_store=dynamics_store,
        activation_config=_LOW_THRESHOLD,
    )
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    before = await dynamics_store.get_many(tenant_id="company_123", identities=[identity])
    await memory.assess_memory(
        "PostgreSQL production",
        tenant_id="company_123",
        as_of=_T1,
    )
    after = await dynamics_store.get_many(tenant_id="company_123", identities=[identity])
    assert before == after


@pytest.mark.asyncio
async def test_assess_memory_no_learning_mutation() -> None:
    learning_store = InMemoryLearningStore()
    episode = _episode()
    episode_store = InMemoryEpisodeStore()
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        learning_store=learning_store,
        activation_config=_LOW_THRESHOLD,
    )
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    await memory.learn(
        LearningFeedback(
            tenant_id="company_123",
            feedback_id="fb-1",
            subject_id="customer_42",
            occurred_at=_T1,
            items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.INCORRECT),),
        )
    )
    assess_at = _T1 + timedelta(hours=1)
    before = await learning_store.list_states(
        tenant_id="company_123",
        identities=(identity,),
        context_keys=("global",),
    )
    await memory.assess_memory("PostgreSQL", tenant_id="company_123", as_of=assess_at)
    await memory.assess_memory("PostgreSQL", tenant_id="company_123", as_of=assess_at)
    after = await learning_store.list_states(
        tenant_id="company_123",
        identities=(identity,),
        context_keys=("global",),
    )
    assert before == after


@pytest.mark.asyncio
async def test_learning_disabled_utility_none() -> None:
    memory = await _memory_with_episode()
    memory = Memory(
        episode_store=memory._episode_store,
        learning_config=LearningConfig(enabled=False),
        activation_config=_LOW_THRESHOLD,
    )
    assessment = await memory.assess_memory(
        "PostgreSQL production",
        tenant_id="company_123",
        as_of=_T1,
    )
    assert assessment.signals.learned_utility is None


@pytest.mark.asyncio
async def test_incorrect_feedback_does_not_change_confidence() -> None:
    memory = await _memory_with_episode(confidence=0.9)
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    await memory.learn(
        LearningFeedback(
            tenant_id="company_123",
            feedback_id="fb-incorrect-1",
            subject_id="customer_42",
            occurred_at=_T1,
            items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.INCORRECT),),
        )
    )
    await memory.learn(
        LearningFeedback(
            tenant_id="company_123",
            feedback_id="fb-incorrect-2",
            subject_id="customer_42",
            occurred_at=_T1,
            items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.INCORRECT),),
        )
    )
    assessment = await memory.assess_memory(
        "PostgreSQL production",
        tenant_id="company_123",
        as_of=_T1 + timedelta(hours=1),
    )
    assert assessment.signals.evidence_confidence == 0.9
    assert assessment.signals.learned_utility is not None
    assert assessment.signals.learned_utility < 0.5


@pytest.mark.asyncio
async def test_deterministic_assessment() -> None:
    memory = await _memory_with_episode()
    first = await memory.assess_memory(
        "PostgreSQL production",
        tenant_id="company_123",
        as_of=_T1,
    )
    second = await memory.assess_memory(
        "PostgreSQL production",
        tenant_id="company_123",
        as_of=_T1,
    )
    assert first == second


@pytest.mark.asyncio
async def test_tenant_isolation() -> None:
    episode_store = InMemoryEpisodeStore()
    episode_a = _episode(
        tenant_id="tenant-a", memory_key="shared-key", statement="Tenant A memory."
    )
    episode_b = _episode(
        tenant_id="tenant-b",
        memory_key="shared-key",
        statement="Tenant B memory.",
        subject_id="other",
    )
    episode_store._episodes[(episode_a.tenant_id, episode_a.memory_key)] = episode_a
    episode_store._episodes[(episode_b.tenant_id, episode_b.memory_key)] = episode_b
    memory = Memory(episode_store=episode_store, activation_config=_LOW_THRESHOLD)
    assessment_a = await memory.assess_memory(
        "memory",
        tenant_id="tenant-a",
        as_of=_T1,
    )
    assessment_b = await memory.assess_memory(
        "memory",
        tenant_id="tenant-b",
        as_of=_T1,
    )
    assert assessment_a.items[0].memory.statement == "Tenant A memory."
    assert assessment_b.items[0].memory.statement == "Tenant B memory."
