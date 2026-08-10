"""Integration tests for memory forgetting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura import Memory
from cogkura.algorithms.activation import activation_candidate_from_episode
from cogkura.algorithms.spreading import calculate_spreading_activation
from cogkura.models import (
    ActivationConfig,
    EpisodeEvidenceInput,
    ForgettingConfig,
    MemoryKind,
    MemoryRetentionState,
    RetrievalCue,
    StoredEpisode,
    StoredMemoryDynamics,
)
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore

_OLD = datetime.now(UTC) - timedelta(days=120)
_RECENT = datetime.now(UTC) - timedelta(seconds=30)
_NOW = datetime.now(UTC)
_FORGETTING_CONFIG = ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0)
_AGGRESSIVE_FORGETTING = ForgettingConfig(
    fading_retention_threshold=0.9,
    forgotten_retention_threshold=0.8,
    grace_period_seconds=86_400.0,
)


def _episode(
    *,
    memory_key: str,
    created_at: datetime,
    entity_ids: tuple[str, ...] = (),
    tenant_id: str = "company_123",
) -> StoredEpisode:
    from cogkura.models import EpisodeEntity

    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id=tenant_id,
        subject_id="customer_42",
        memory_key=memory_key,
        statement=f"Statement for {memory_key}",
        started_at=created_at,
        ended_at=created_at,
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
        entities=tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids),
        metadata=MappingProxyType({}),
        created_at=created_at,
        updated_at=created_at,
    )


def _memory(
    *,
    episodes: list[StoredEpisode],
    activation_config: ActivationConfig = _FORGETTING_CONFIG,
) -> Memory:
    episode_store = InMemoryEpisodeStore()
    for episode in episodes:
        episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    return Memory(
        episode_store=episode_store,
        activation_config=activation_config,
        forgetting_config=_AGGRESSIVE_FORGETTING,
    )


def _memory_with_old_episode(*, memory_key: str = "episode-key") -> Memory:
    return _memory(episodes=[_episode(memory_key=memory_key, created_at=_OLD)])


def _memory_with_recent_episode(*, memory_key: str = "episode-key") -> Memory:
    return _memory(episodes=[_episode(memory_key=memory_key, created_at=_RECENT)])


async def _forget_episode(memory: Memory, *, tenant_id: str = "company_123") -> None:
    await memory.apply_forgetting(tenant_id=tenant_id, as_of=_NOW)
    await memory.apply_forgetting(
        tenant_id=tenant_id,
        as_of=_NOW + timedelta(days=2),
    )


@pytest.mark.asyncio
async def test_apply_forgetting_transitions_states() -> None:
    memory = _memory_with_old_episode()
    first = await memory.apply_forgetting(tenant_id="company_123", as_of=_NOW)
    assert first.evaluated == 1
    assert first.fading == 1
    second = await memory.apply_forgetting(
        tenant_id="company_123",
        as_of=_NOW + timedelta(days=2),
    )
    assert second.forgotten == 1


@pytest.mark.asyncio
async def test_forgotten_memories_excluded_from_recall() -> None:
    memory = _memory_with_recent_episode()
    candidate = activation_candidate_from_episode(
        _episode(memory_key="episode-key", created_at=_RECENT)
    )
    await memory._dynamics_store.upsert_many(
        [
            StoredMemoryDynamics(
                tenant_id="company_123",
                memory_kind=MemoryKind.EPISODE,
                memory_key="episode-key",
                retention_state=MemoryRetentionState.FORGOTTEN,
                last_base_level=-1.0,
                last_retention_score=0.01,
                below_threshold_since=_NOW - timedelta(days=3),
                forgotten_at=_NOW - timedelta(days=1),
                evaluated_at=_NOW,
                updated_at=_NOW,
            )
        ]
    )
    results = await memory.recall("Statement", tenant_id="company_123", as_of=_NOW)
    assert results == []
    recovered = await memory.recall(
        "Statement",
        tenant_id="company_123",
        as_of=_NOW,
        include_forgotten=True,
    )
    assert recovered
    assert recovered[0].memory.memory_key == candidate.memory_key


@pytest.mark.asyncio
async def test_record_access_reactivates_forgotten_memory() -> None:
    memory = _memory_with_recent_episode()
    await memory._dynamics_store.upsert_many(
        [
            StoredMemoryDynamics(
                tenant_id="company_123",
                memory_kind=MemoryKind.EPISODE,
                memory_key="episode-key",
                retention_state=MemoryRetentionState.FORGOTTEN,
                last_base_level=-1.0,
                last_retention_score=0.01,
                below_threshold_since=_NOW - timedelta(days=3),
                forgotten_at=_NOW - timedelta(days=1),
                evaluated_at=_NOW,
                updated_at=_NOW,
            )
        ]
    )
    forgotten = await memory.recall(
        "Statement",
        tenant_id="company_123",
        as_of=_NOW,
        include_forgotten=True,
        limit=1,
    )
    await memory.record_access(forgotten, tenant_id="company_123", referenced_at=_NOW)
    await memory.apply_forgetting(tenant_id="company_123", as_of=_NOW + timedelta(hours=1))
    results = await memory.recall("Statement", tenant_id="company_123", as_of=_NOW)
    assert results


def test_fading_memory_can_receive_spreading_activation() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="a",
                created_at=_OLD,
                entity_ids=("alice", "project-kura"),
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="b",
                created_at=_OLD,
                entity_ids=("project-kura",),
            )
        ),
    ]
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    assert result.scores[candidates[1].identity] > 0.0


@pytest.mark.asyncio
async def test_tenant_isolation_for_forgetting() -> None:
    memory = _memory(
        episodes=[
            _episode(memory_key="episode-a", created_at=_OLD, tenant_id="tenant-a"),
            _episode(memory_key="episode-b", created_at=_RECENT, tenant_id="tenant-b"),
        ],
    )
    await _forget_episode(memory, tenant_id="tenant-a")
    results_b = await memory.recall(
        "Statement for episode-b",
        tenant_id="tenant-b",
        as_of=_NOW,
    )
    assert results_b
