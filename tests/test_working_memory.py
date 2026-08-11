"""Unit and integration tests for working-memory selection."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura.algorithms.working_memory import (
    ApproximateTokenEstimator,
    DeterministicWorkingMemorySelector,
    calculate_goal_relevance,
)
from cogkura.exceptions import ValidationError
from cogkura.memory import Memory
from cogkura.models import (
    ActivationComponents,
    ActivationConfig,
    EpisodeEvidenceInput,
    MemoryIdentity,
    MemoryKind,
    RecallResult,
    RetrievalCue,
    StoredEpisode,
    WorkingMemoryComponents,
    WorkingMemoryConfig,
    WorkingMemoryItem,
    WorkingMemorySnapshot,
)
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
_SELECTOR = DeterministicWorkingMemorySelector()
_ESTIMATOR = ApproximateTokenEstimator()


def _episode(
    *,
    memory_key: str = "episode-key",
    statement: str = "PostgreSQL incident resolved.",
    importance: float = 0.7,
    subject_id: str | None = "customer_42",
) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id=subject_id,
        memory_key=memory_key,
        statement=statement,
        started_at=_NOW,
        ended_at=_NOW,
        confidence=0.9,
        importance=importance,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-1",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({}),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _recall(
    episode: StoredEpisode,
    *,
    score: float,
    activation: float = 1.0,
) -> RecallResult:
    return RecallResult(
        memory_kind=MemoryKind.EPISODE,
        memory=episode,
        activation=activation,
        score=score,
        latency_seconds=0.1,
        components=ActivationComponents(
            base_level=activation,
            spreading=0.0,
            partial_match=0.0,
            noise=0.0,
            total=activation,
        ),
        reason="test recall",
    )


class FixedTokenEstimator:
    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = mapping

    def estimate(self, text: str) -> int:
        return self._mapping.get(text, 10)


def _select(
    candidates: list[RecallResult],
    *,
    goal: RetrievalCue | str,
    config: WorkingMemoryConfig | None = None,
    previous: WorkingMemorySnapshot | None = None,
    as_of: datetime = _NOW,
    token_estimator: object = _ESTIMATOR,
    prompt_budget_tokens: int | None = None,
) -> WorkingMemorySnapshot:
    goal_cue = RetrievalCue(text=goal) if isinstance(goal, str) else goal
    return _SELECTOR.select(
        candidates=candidates,
        goal=goal_cue,
        tenant_id="company_123",
        subject_id="customer_42",
        previous=previous,
        as_of=as_of,
        config=config or WorkingMemoryConfig(),
        token_estimator=token_estimator,
        prompt_budget_tokens=prompt_budget_tokens,
    )


def test_working_memory_config_validation() -> None:
    with pytest.raises(ValidationError):
        WorkingMemoryConfig(candidate_pool_size=0)
    with pytest.raises(ValidationError):
        WorkingMemoryConfig(activation_weight=-0.1)
    with pytest.raises(ValidationError):
        WorkingMemoryConfig(
            activation_weight=0.0,
            goal_relevance_weight=0.0,
            importance_weight=0.0,
            carryover_weight=0.0,
        )


def test_working_memory_components_validation() -> None:
    with pytest.raises(ValidationError):
        WorkingMemoryComponents(
            activation=1.5,
            goal_relevance=0.5,
            importance=0.5,
            carryover=0.0,
            base_priority=0.5,
            inhibition=0.0,
            final_score=0.5,
        )


def test_approximate_token_estimator() -> None:
    estimator = ApproximateTokenEstimator()
    assert estimator.estimate("") == 0
    assert estimator.estimate("hello") == 2
    assert estimator.estimate("café") >= 1
    assert estimator.estimate("hello") == estimator.estimate("hello")


def test_bounded_capacity() -> None:
    candidates = [
        _recall(_episode(memory_key=f"ep-{index}", statement=f"memory {index}"), score=0.9)
        for index in range(20)
    ]
    snapshot = _select(
        candidates,
        goal="memory",
        config=WorkingMemoryConfig(max_items=4, candidate_pool_size=50),
    )
    assert len(snapshot.items) == 4


def test_prompt_budget_skips_oversized_candidate() -> None:
    candidates = [
        _recall(_episode(memory_key="a", statement="alpha"), score=0.9),
        _recall(_episode(memory_key="b", statement="beta"), score=0.85),
        _recall(_episode(memory_key="c", statement="gamma"), score=0.8),
        _recall(_episode(memory_key="d", statement="delta"), score=0.75),
    ]
    estimator = FixedTokenEstimator(
        {
            "alpha": 300,
            "beta": 250,
            "gamma": 700,
            "delta": 200,
        }
    )
    snapshot = _select(
        candidates,
        goal="alpha beta gamma delta",
        config=WorkingMemoryConfig(max_items=8, max_prompt_tokens=1000),
        token_estimator=estimator,
        prompt_budget_tokens=1000,
    )
    selected_keys = {item.memory.memory_key for item in snapshot.items}
    assert "gamma" not in selected_keys
    assert snapshot.estimated_prompt_tokens <= 1000
    assert snapshot.budget_skipped_count >= 1


def test_goal_relevance_changes_ranking() -> None:
    high_activation = _recall(
        _episode(
            memory_key="high-act",
            statement="The company Christmas party was held in Manchester.",
        ),
        score=0.95,
        activation=2.0,
    )
    high_relevance = _recall(
        _episode(
            memory_key="high-goal",
            statement="Choose production database with low operational complexity.",
        ),
        score=0.55,
        activation=0.5,
    )
    snapshot = _select(
        [high_activation, high_relevance],
        goal="Choose a production database with low operational complexity.",
    )
    assert snapshot.items[0].memory.memory_key == "high-goal"


def test_redundant_memories_inhibit_one_another() -> None:
    candidates = [
        _recall(
            _episode(
                memory_key="a",
                statement="PostgreSQL reduces operational database complexity",
            ),
            score=0.9,
        ),
        _recall(
            _episode(
                memory_key="b",
                statement="PostgreSQL can reduce operational database complexity",
            ),
            score=0.88,
        ),
        _recall(
            _episode(
                memory_key="c",
                statement="Redis introduces another infrastructure dependency",
            ),
            score=0.85,
        ),
    ]
    snapshot = _select(
        candidates,
        goal="minimise operational complexity",
        config=WorkingMemoryConfig(
            max_items=2,
            redundancy_threshold=0.5,
            inhibition_strength=0.5,
        ),
    )
    selected_keys = {item.memory.memory_key for item in snapshot.items}
    assert "a" in selected_keys or "b" in selected_keys
    assert "c" in selected_keys
    assert len(selected_keys) == 2


def test_distinct_entity_facts_survive() -> None:
    candidates = [
        _recall(
            _episode(
                memory_key="ddl",
                statement="PostgreSQL supports transactional DDL.",
            ),
            score=0.9,
        ),
        _recall(
            _episode(
                memory_key="backup",
                statement="PostgreSQL requires operational backup planning.",
            ),
            score=0.88,
        ),
    ]
    snapshot = _select(
        candidates,
        goal="PostgreSQL operational planning",
        config=WorkingMemoryConfig(max_items=2),
    )
    assert len(snapshot.items) == 2


def test_deterministic_tie_breaking() -> None:
    episode_a = _episode(memory_key="aaa", statement="shared tokens here")
    episode_b = _episode(memory_key="bbb", statement="shared tokens here")
    candidates = [
        _recall(episode_a, score=0.8, activation=1.0),
        _recall(episode_b, score=0.8, activation=1.0),
    ]
    first = _select(candidates, goal="shared tokens")
    second = _select(candidates, goal="shared tokens")
    assert [item.memory.memory_key for item in first.items] == [
        item.memory.memory_key for item in second.items
    ]


def test_fast_decay() -> None:
    episode = _episode(memory_key="carry", statement="operational complexity database")
    recall = _recall(episode, score=0.8)
    previous = _select([recall], goal="operational complexity", as_of=_NOW)
    previous_item = previous.items[0]
    previous_with_strength = WorkingMemorySnapshot(
        tenant_id="company_123",
        subject_id="customer_42",
        goal=RetrievalCue(text="operational complexity"),
        items=(
            WorkingMemoryItem(
                recall=previous_item.recall,
                estimated_tokens=previous_item.estimated_tokens,
                transient_strength=1.0,
                components=previous_item.components,
                rank=1,
                reason=previous_item.reason,
            ),
        ),
        created_at=_NOW,
        candidate_count=1,
        selected_count=1,
        estimated_prompt_tokens=previous_item.estimated_tokens,
        prompt_budget_tokens=2048,
        goal_filtered_count=0,
        inhibited_count=0,
        budget_skipped_count=0,
    )
    config = WorkingMemoryConfig(decay_half_life_seconds=300.0)
    at_five_minutes = _NOW + timedelta(seconds=300)
    snapshot = _select(
        [recall],
        goal="operational complexity",
        config=config,
        previous=previous_with_strength,
        as_of=at_five_minutes,
    )
    assert math.isclose(snapshot.items[0].components.carryover, 0.5, rel_tol=1e-9)

    at_ten_minutes = _NOW + timedelta(seconds=600)
    snapshot_ten = _select(
        [recall],
        goal="operational complexity",
        config=config,
        previous=previous_with_strength,
        as_of=at_ten_minutes,
    )
    assert math.isclose(snapshot_ten.items[0].components.carryover, 0.25, rel_tol=1e-9)


def test_refresh_boosts_repeated_candidate() -> None:
    episode = _episode(memory_key="refresh", statement="operational complexity database")
    recall = _recall(episode, score=0.7)
    peer = _recall(
        _episode(memory_key="peer", statement="operational complexity database peer"),
        score=0.7,
    )
    previous = _select([recall, peer], goal="operational complexity", as_of=_NOW)
    refreshed = _select(
        [recall, peer],
        goal="operational complexity",
        previous=previous,
        as_of=_NOW + timedelta(seconds=60),
    )
    refreshed_carry = next(
        item.components.carryover for item in refreshed.items if item.memory.memory_key == "refresh"
    )
    new_only = _select([peer], goal="operational complexity", as_of=_NOW + timedelta(seconds=60))
    assert refreshed_carry > new_only.items[0].components.carryover


def test_missing_previous_candidate_not_reinserted() -> None:
    episode = _episode(memory_key="stale", statement="operational complexity")
    recall = _recall(episode, score=0.9)
    previous = _select([recall], goal="operational complexity")
    snapshot = _select([], goal="operational complexity", previous=previous)
    assert snapshot.items == ()


def test_previous_tenant_isolation() -> None:
    episode = _episode(memory_key="iso", statement="operational complexity")
    previous = WorkingMemorySnapshot(
        tenant_id="tenant-a",
        subject_id="customer_42",
        goal=RetrievalCue(text="operational complexity"),
        items=(),
        created_at=_NOW,
        candidate_count=0,
        selected_count=0,
        estimated_prompt_tokens=0,
        prompt_budget_tokens=2048,
        goal_filtered_count=0,
        inhibited_count=0,
        budget_skipped_count=0,
    )
    with pytest.raises(ValidationError):
        _select([_recall(episode, score=0.8)], goal="operational", previous=previous)


def test_previous_subject_isolation() -> None:
    episode = _episode(memory_key="iso", statement="operational complexity")
    previous = WorkingMemorySnapshot(
        tenant_id="company_123",
        subject_id="customer-99",
        goal=RetrievalCue(text="operational complexity"),
        items=(),
        created_at=_NOW,
        candidate_count=0,
        selected_count=0,
        estimated_prompt_tokens=0,
        prompt_budget_tokens=2048,
        goal_filtered_count=0,
        inhibited_count=0,
        budget_skipped_count=0,
    )
    with pytest.raises(ValidationError):
        _select([_recall(episode, score=0.8)], goal="operational", previous=previous)


def test_goal_relevance_text_coverage() -> None:
    episode = _episode(statement="PostgreSQL reduces operational complexity")
    recall = _recall(episode, score=0.8)
    relevance = calculate_goal_relevance(
        recall,
        RetrievalCue(text="operational complexity database"),
    )
    assert relevance > 0.5


@pytest.mark.asyncio
async def test_memory_select_working_memory_integration() -> None:
    episode_store = InMemoryEpisodeStore()
    episode = _episode(
        memory_key="wm-1",
        statement="PostgreSQL reduces operational complexity.",
    )
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        activation_config=ActivationConfig(retrieval_threshold=-10.0),
    )
    snapshot = await memory.select_working_memory(
        "database operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        goal="minimise operational complexity",
    )
    assert snapshot.selected_count >= 1


@pytest.mark.asyncio
async def test_select_working_memory_does_not_mutate_durable_state() -> None:
    episode_store = InMemoryEpisodeStore()
    episode = _episode(memory_key="durability", statement="operational complexity")
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        activation_config=ActivationConfig(retrieval_threshold=-10.0),
    )
    refs_before = await memory._activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="durability")],
        before_or_at=_NOW,
    )
    await memory.select_working_memory(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
    )
    refs_after = await memory._activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="durability")],
        before_or_at=_NOW,
    )
    assert refs_before == refs_after


@pytest.mark.asyncio
async def test_record_access_after_selection() -> None:
    episode_store = InMemoryEpisodeStore()
    episode = _episode(memory_key="access", statement="operational complexity")
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        activation_config=ActivationConfig(retrieval_threshold=-10.0),
    )
    snapshot = await memory.select_working_memory(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
    )
    await memory.record_access(
        snapshot.recall_results,
        tenant_id="company_123",
        referenced_at=_NOW,
    )
    refs = await memory._activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="access")],
        before_or_at=_NOW + timedelta(seconds=1),
    )
    assert len(refs[MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="access")]) >= 1


def test_architecture_decision_evaluation_fixture() -> None:
    memories = [
        ("team", "PostgreSQL is already operated by the team."),
        ("redis-dep-1", "Redis would introduce an additional operational dependency."),
        ("redis-dep-2", "Redis would add another operational dependency."),
        ("postgres-workload", "PostgreSQL supports the required transactional workload."),
        ("party", "The company Christmas party was held in Manchester."),
        ("postgres-mentioned", "PostgreSQL was mentioned repeatedly in earlier discussions."),
        ("redis-latency", "Redis has lower latency for some key-value workloads."),
        ("goal-memory", "The production goal is to minimise operational complexity."),
    ]
    candidates = [
        _recall(_episode(memory_key=key, statement=statement), score=score)
        for key, statement, score in [
            ("team", memories[0][1], 0.85),
            ("redis-dep-1", memories[1][1], 0.82),
            ("redis-dep-2", memories[2][1], 0.81),
            ("postgres-workload", memories[3][1], 0.88),
            ("party", memories[4][1], 0.6),
            ("postgres-mentioned", memories[5][1], 0.93),
            ("redis-latency", memories[6][1], 0.8),
            ("goal-memory", memories[7][1], 0.7),
        ]
    ]
    snapshot = _select(
        candidates,
        goal="Choose a production data store while minimising operational complexity.",
        config=WorkingMemoryConfig(
            max_items=5,
            max_prompt_tokens=500,
            goal_relevance_weight=0.5,
            activation_weight=0.3,
        ),
        token_estimator=FixedTokenEstimator({statement: 40 for _, statement in memories}),
    )
    selected_keys = {item.memory.memory_key for item in snapshot.items}
    assert "party" not in selected_keys
    assert "postgres-workload" in selected_keys or "team" in selected_keys
    assert len(selected_keys) <= 5
    assert snapshot.estimated_prompt_tokens <= 500
