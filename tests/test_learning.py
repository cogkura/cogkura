"""Unit tests for learning and reinforcement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura import Memory
from cogkura.algorithms.learning import (
    DeterministicLearningProcessor,
    calculate_association_strength,
    calculate_utility,
    learning_context_key,
)
from cogkura.exceptions import StorageError, ValidationError
from cogkura.models import (
    EpisodeEvidenceInput,
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
    MemoryRetentionState,
    RetrievalCue,
    StoredEpisode,
    StoredMemoryDynamics,
)
from cogkura.storage.in_memory_dynamics import InMemoryMemoryDynamicsStore
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore
from cogkura.storage.in_memory_learning import InMemoryLearningStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 6, 1, tzinfo=UTC)


def _now() -> datetime:
    return datetime.now(UTC)


def _episode(*, memory_key: str = "episode-a") -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        statement=f"Episode {memory_key}.",
        started_at=_T1,
        ended_at=_T1,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-1",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({"episode": {"content_fingerprint": memory_key}}),
        created_at=_T1,
        updated_at=_T1,
    )


async def _memory_with_episode(*, memory_key: str = "episode-a") -> Memory:
    episode_store = InMemoryEpisodeStore()
    await episode_store.upsert(_episode(memory_key=memory_key))
    return Memory(episode_store=episode_store)


def _feedback(
    *,
    feedback_id: str = "feedback-1",
    outcome: LearningOutcome = LearningOutcome.HELPFUL,
    memory_key: str = "episode-a",
    goal: RetrievalCue | None = None,
    occurred_at: datetime | None = None,
) -> LearningFeedback:
    return LearningFeedback(
        tenant_id="company_123",
        feedback_id=feedback_id,
        subject_id="customer_42",
        goal=goal,
        occurred_at=occurred_at or _T1,
        items=(
            MemoryFeedback(
                identity=MemoryIdentity(
                    memory_kind=MemoryKind.EPISODE,
                    memory_key=memory_key,
                ),
                outcome=outcome,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_helpful_feedback_increments_counts_and_traces() -> None:
    memory = await _memory_with_episode()
    result = await memory.learn(_feedback())
    assert result.created is True
    assert result.helpful == 1
    assert result.memories_reinforced == 1

    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    states = await memory.list_learning_state(
        tenant_id="company_123",
        identities=[identity],
    )
    assert len(states) == 1
    assert states[0].helpful_count == 1
    assert states[0].unhelpful_count == 0
    assert states[0].incorrect_count == 0

    traces = await memory._list_activation_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T1 + timedelta(hours=1),
    )
    assert len(traces[identity]) == 1


@pytest.mark.asyncio
async def test_unhelpful_and_incorrect_do_not_create_reinforcement_traces() -> None:
    memory = await _memory_with_episode()
    await memory.learn(_feedback(outcome=LearningOutcome.UNHELPFUL))
    await memory.learn(_feedback(feedback_id="feedback-2", outcome=LearningOutcome.INCORRECT))
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    traces = await memory._list_activation_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T1 + timedelta(hours=1),
    )
    assert identity not in traces


@pytest.mark.asyncio
async def test_idempotent_feedback_id() -> None:
    memory = await _memory_with_episode()
    first = await memory.learn(_feedback())
    second = await memory.learn(_feedback())
    assert first.created is True
    assert second.unchanged is True
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    states = await memory.list_learning_state(
        tenant_id="company_123",
        identities=[identity],
    )
    assert states[0].helpful_count == 1


@pytest.mark.asyncio
async def test_conflicting_fingerprint_raises() -> None:
    store = InMemoryLearningStore()
    processor = DeterministicLearningProcessor()
    config = LearningConfig()
    plan = processor.plan(feedback=_feedback(), config=config)
    await store.apply(plan)
    conflicting = LearningFeedback(
        tenant_id="company_123",
        feedback_id="feedback-1",
        subject_id="customer_42",
        occurred_at=_T1,
        items=(
            MemoryFeedback(
                identity=MemoryIdentity(
                    memory_kind=MemoryKind.EPISODE,
                    memory_key="episode-a",
                ),
                outcome=LearningOutcome.UNHELPFUL,
            ),
        ),
    )
    with pytest.raises(StorageError):
        await store.apply(processor.plan(feedback=conflicting, config=config))


@pytest.mark.asyncio
async def test_unknown_target_fails() -> None:
    memory = Memory()
    with pytest.raises(ValidationError):
        await memory.learn(_feedback())


@pytest.mark.asyncio
async def test_helpful_raises_activation_without_record_access() -> None:
    memory = await _memory_with_episode()
    evaluation = _T1 + timedelta(hours=1)
    before = await memory.recall("Episode", tenant_id="company_123", as_of=evaluation)
    await memory.learn(_feedback(occurred_at=evaluation + timedelta(seconds=1)))
    after = await memory.recall(
        "Episode",
        tenant_id="company_123",
        as_of=evaluation + timedelta(seconds=2),
    )
    assert before, "expected recall results before learning reinforcement"
    assert after, "expected recall results after learning reinforcement"
    assert after[0].activation > before[0].activation


@pytest.mark.asyncio
async def test_helpful_reactivates_forgotten_memory() -> None:
    dynamics_store = InMemoryMemoryDynamicsStore()
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    await dynamics_store.upsert_many(
        [
            StoredMemoryDynamics(
                tenant_id="company_123",
                memory_kind=identity.memory_kind,
                memory_key=identity.memory_key,
                retention_state=MemoryRetentionState.FORGOTTEN,
                last_base_level=-4.0,
                last_retention_score=0.02,
                below_threshold_since=_T0,
                forgotten_at=_T0,
                evaluated_at=_T0,
                updated_at=_T0,
            )
        ]
    )
    memory = Memory(
        episode_store=InMemoryEpisodeStore(),
        dynamics_store=dynamics_store,
    )
    await memory._episode_store.upsert(_episode())
    result = await memory.learn(_feedback())
    assert result.reactivated == 1
    loaded = await dynamics_store.get_many(
        tenant_id="company_123",
        identities=[identity],
    )
    assert loaded[identity].retention_state is MemoryRetentionState.ACTIVE


@pytest.mark.asyncio
async def test_unhelpful_does_not_reactivate_forgotten_memory() -> None:
    dynamics_store = InMemoryMemoryDynamicsStore()
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a")
    await dynamics_store.upsert_many(
        [
            StoredMemoryDynamics(
                tenant_id="company_123",
                memory_kind=identity.memory_kind,
                memory_key=identity.memory_key,
                retention_state=MemoryRetentionState.FORGOTTEN,
                last_base_level=-4.0,
                last_retention_score=0.02,
                below_threshold_since=_T0,
                forgotten_at=_T0,
                evaluated_at=_T0,
                updated_at=_T0,
            )
        ]
    )
    memory = Memory(
        episode_store=InMemoryEpisodeStore(),
        dynamics_store=dynamics_store,
    )
    await memory._episode_store.upsert(_episode())
    result = await memory.learn(_feedback(outcome=LearningOutcome.UNHELPFUL))
    assert result.reactivated == 0


@pytest.mark.asyncio
async def test_working_memory_neutral_without_learning_state() -> None:
    episode_store = InMemoryEpisodeStore()
    for index, _score in enumerate((0.95, 0.90, 0.85)):
        await episode_store.upsert(_episode(memory_key=f"episode-{index}"))
    memory = Memory(episode_store=episode_store)
    evaluation = _now()
    baseline = await memory.select_working_memory(
        "Episode",
        tenant_id="company_123",
        as_of=evaluation,
    )
    disabled = Memory(
        episode_store=episode_store,
        learning_config=LearningConfig(enabled=False),
    )
    unchanged = await disabled.select_working_memory(
        "Episode",
        tenant_id="company_123",
        as_of=evaluation,
    )
    assert [item.identity.memory_key for item in baseline.items] == [
        item.identity.memory_key for item in unchanged.items
    ]


@pytest.mark.asyncio
async def test_contextual_utility_penalizes_goal_specific_unhelpful() -> None:
    episode_store = InMemoryEpisodeStore()
    await episode_store.upsert(_episode(memory_key="episode-a"))
    await episode_store.upsert(_episode(memory_key="episode-b"))
    memory = Memory(episode_store=episode_store)
    goal = RetrievalCue(text="database migration")
    evaluation = _T1 + timedelta(hours=1)
    await memory.learn(
        LearningFeedback(
            tenant_id="company_123",
            feedback_id="fb-a",
            subject_id="customer_42",
            goal=goal,
            occurred_at=evaluation,
            items=(
                MemoryFeedback(
                    identity=MemoryIdentity(
                        memory_kind=MemoryKind.EPISODE,
                        memory_key="episode-a",
                    ),
                    outcome=LearningOutcome.UNHELPFUL,
                ),
            ),
        )
    )
    contextual = await memory.select_working_memory(
        "Episode",
        tenant_id="company_123",
        goal=goal,
        as_of=evaluation + timedelta(hours=1),
    )
    global_only = await memory.select_working_memory(
        "Episode",
        tenant_id="company_123",
        as_of=evaluation + timedelta(hours=1),
    )
    contextual_a = next(
        item for item in contextual.items if item.identity.memory_key == "episode-a"
    )
    global_a = next(item for item in global_only.items if item.identity.memory_key == "episode-a")
    assert contextual_a.components.learned_utility < global_a.components.learned_utility


@pytest.mark.asyncio
async def test_association_pairs_require_helpful_co_use() -> None:
    episode_store = InMemoryEpisodeStore()
    await episode_store.upsert(_episode(memory_key="episode-a"))
    await episode_store.upsert(_episode(memory_key="episode-b"))
    memory = Memory(episode_store=episode_store)
    mixed = LearningFeedback(
        tenant_id="company_123",
        feedback_id="mixed",
        subject_id="customer_42",
        occurred_at=_T1,
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
                outcome=LearningOutcome.UNHELPFUL,
            ),
        ),
    )
    await memory.learn(mixed)
    associations = await memory._learning_store.list_associations(
        tenant_id="company_123",
        identities=[
            MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-a"),
            MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-b"),
        ],
    )
    assert associations == ()


def test_utility_defaults_to_neutral() -> None:
    config = LearningConfig()
    assert calculate_utility(helpful=0, unhelpful=0, incorrect=0, config=config) == pytest.approx(
        0.5
    )


def test_association_strength_respects_minimum_coactivations() -> None:
    config = LearningConfig(minimum_association_coactivations=2, association_tau=3.0)
    assert calculate_association_strength(1, config=config) == 0.0
    assert calculate_association_strength(2, config=config) > 0.0


def test_learning_context_key_is_global_without_goal() -> None:
    assert learning_context_key(None) == "global"
    assert learning_context_key(RetrievalCue(text="  Database  ")) != "global"
