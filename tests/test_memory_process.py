"""Tests for Memory.process() orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.exceptions import ValidationError
from cogkura.models import EpisodeEvidenceInput, SemanticMemoryStatus, StoredEpisode
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 6, 1, tzinfo=UTC)
_T2 = datetime(2027, 1, 1, tzinfo=UTC)
_FIXED_TIME = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _semantic_observation(
    *,
    source_record_id: str,
    conversation_id: str,
    semantic_fact: dict,
    subject_id: str = "customer_42",
    content: str = "Database preference discussion.",
    observed_at: datetime = _FIXED_TIME,
) -> ObservationInput:
    return ObservationInput(
        tenant_id="company_123",
        subject_id=subject_id,
        actor_id=subject_id,
        source_namespace="chat.messages",
        source_record_id=source_record_id,
        event_type="message",
        content=content,
        observed_at=observed_at,
        metadata={
            "conversation_id": conversation_id,
            "entity_ids": [subject_id],
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


def _stale_episode(
    *,
    subject_id: str = "customer_42",
    memory_key: str = "stale-episode",
) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id=subject_id,
        memory_key=memory_key,
        statement="Stale episode with no backing observations.",
        started_at=_FIXED_TIME,
        ended_at=_FIXED_TIME,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-stale",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({"episode": {"content_fingerprint": memory_key}}),
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )


@pytest.mark.asyncio
async def test_process_empty_state() -> None:
    memory = Memory()
    result = await memory.process(tenant_id="company_123", as_of=_FIXED_TIME)
    assert result.tenant_id == "company_123"
    assert result.subject_id is None
    assert result.processed_at == _FIXED_TIME
    assert result.episodes.observations == 0
    assert result.episodes.created == 0
    assert result.semantics.promoted == 0


@pytest.mark.asyncio
async def test_process_episodic() -> None:
    memory = Memory()
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="customer_42",
            source_namespace="direct",
            source_record_id="1",
            content="Customer discussed PostgreSQL for production.",
            observed_at=_FIXED_TIME,
        )
    )
    result = await memory.process(
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_FIXED_TIME,
    )
    assert result.episodes.created >= 1
    episodes = await memory.list_episodes(tenant_id="company_123", subject_id="customer_42")
    assert len(episodes) >= 1
    assert episodes[0].updated_at == _FIXED_TIME


@pytest.mark.asyncio
async def test_process_semantic() -> None:
    memory = Memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    result = await memory.process(tenant_id="company_123", subject_id="customer_42")
    memories = await memory.list_semantic_memories(
        tenant_id="company_123",
        subject_id="customer_42",
    )
    assert result.semantics.promoted == 1
    assert result.semantics.created == 1
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_process_is_idempotent() -> None:
    memory = Memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact=_SEMANTIC_FACT,
        )
    )
    first = await memory.process(tenant_id="company_123")
    second = await memory.process(tenant_id="company_123")
    assert first.semantics.created == 1
    assert second.semantics.unchanged == 1
    memories = await memory.list_semantic_memories(tenant_id="company_123")
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_process_uses_single_as_of() -> None:
    memory = Memory()
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            source_namespace="direct",
            source_record_id="1",
            content="Observation for shared timestamp.",
            observed_at=_FIXED_TIME,
        )
    )
    result = await memory.process(tenant_id="company_123", as_of=_FIXED_TIME)
    assert result.processed_at == _FIXED_TIME
    episodes = await memory.list_episodes(tenant_id="company_123")
    assert len(episodes) == 1
    assert episodes[0].updated_at == _FIXED_TIME


@pytest.mark.asyncio
async def test_process_rejects_naive_as_of() -> None:
    memory = Memory()
    with pytest.raises(ValidationError, match="timezone-aware"):
        await memory.process(
            tenant_id="company_123",
            as_of=datetime(2026, 3, 1, 12, 0),
        )


@pytest.mark.asyncio
async def test_process_empty_tenant_raises() -> None:
    memory = Memory()
    with pytest.raises(ValidationError, match="tenant_id"):
        await memory.process(tenant_id="")


@pytest.mark.asyncio
async def test_process_subject_id_scopes_encoding() -> None:
    memory = Memory()
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="customer_a",
            source_namespace="direct",
            source_record_id="a",
            content="Customer A discussed PostgreSQL.",
            observed_at=_FIXED_TIME,
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="customer_b",
            source_namespace="direct",
            source_record_id="b",
            content="Customer B discussed Redis.",
            observed_at=_FIXED_TIME,
        )
    )
    result = await memory.process(
        tenant_id="company_123",
        subject_id="customer_a",
        as_of=_FIXED_TIME,
    )
    assert result.subject_id == "customer_a"
    episodes_a = await memory.list_episodes(tenant_id="company_123", subject_id="customer_a")
    episodes_b = await memory.list_episodes(tenant_id="company_123", subject_id="customer_b")
    assert len(episodes_a) >= 1
    assert episodes_b == []


@pytest.mark.asyncio
async def test_process_deactivates_stale_episodes_for_tenant() -> None:
    episode_store = InMemoryEpisodeStore()
    stale = _stale_episode()
    episode_store._episodes[(stale.tenant_id, stale.memory_key)] = stale
    memory = Memory(episode_store=episode_store)
    result = await memory.process(tenant_id="company_123", as_of=_FIXED_TIME)
    assert result.episodes.deactivated >= 1
    episodes = await memory.list_episodes(tenant_id="company_123", include_inactive=True)
    assert all(not episode.is_active for episode in episodes)


@pytest.mark.asyncio
async def test_process_subject_scope_does_not_deactivate_other_subjects() -> None:
    episode_store = InMemoryEpisodeStore()
    episode_a = _stale_episode(subject_id="customer_a", memory_key="a-episode")
    episode_b = _stale_episode(subject_id="customer_b", memory_key="b-episode")
    episode_store._episodes[(episode_a.tenant_id, episode_a.memory_key)] = episode_a
    episode_store._episodes[(episode_b.tenant_id, episode_b.memory_key)] = episode_b
    memory = Memory(episode_store=episode_store)
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="customer_a",
            source_namespace="direct",
            source_record_id="a",
            content="Customer A observation.",
            observed_at=_FIXED_TIME,
        )
    )
    result = await memory.process(
        tenant_id="company_123",
        subject_id="customer_a",
        as_of=_FIXED_TIME,
    )
    assert result.episodes.deactivated >= 1
    active = await memory.list_episodes(
        tenant_id="company_123",
        subject_id="customer_b",
        include_inactive=False,
    )
    assert len(active) == 1
    assert active[0].memory_key == "b-episode"


@pytest.mark.asyncio
async def test_process_empty_subject_scope_deactivates_only_that_subject() -> None:
    episode_store = InMemoryEpisodeStore()
    episode_a = _stale_episode(subject_id="customer_a", memory_key="a-episode")
    episode_b = _stale_episode(subject_id="customer_b", memory_key="b-episode")
    episode_store._episodes[(episode_a.tenant_id, episode_a.memory_key)] = episode_a
    episode_store._episodes[(episode_b.tenant_id, episode_b.memory_key)] = episode_b
    memory = Memory(episode_store=episode_store)
    result = await memory.process(
        tenant_id="company_123",
        subject_id="customer_a",
        as_of=_FIXED_TIME,
    )
    assert result.episodes.deactivated >= 1
    scoped = await memory.list_episodes(
        tenant_id="company_123",
        subject_id="customer_a",
        include_inactive=True,
    )
    assert scoped
    assert all(not episode.is_active for episode in scoped)
    other = await memory.list_episodes(
        tenant_id="company_123",
        subject_id="customer_b",
        include_inactive=False,
    )
    assert len(other) == 1
    assert other[0].memory_key == "b-episode"


@pytest.mark.asyncio
async def test_process_reconsolidation_supersedes_with_shared_as_of() -> None:
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact={
                "predicate": "preferred_vendor",
                "object_value": "Acme",
                "object_entity_id": "acme",
                "cardinality": "one",
                "polarity": "affirm",
                "valid_from": _T0.isoformat(),
                "valid_until": _T1.isoformat(),
            },
            content="Preferred Acme.",
            observed_at=_T0,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact={
                "predicate": "preferred_vendor",
                "object_value": "Beta",
                "object_entity_id": "beta",
                "cardinality": "one",
                "polarity": "affirm",
                "valid_from": _T1.isoformat(),
                "valid_until": _T2.isoformat(),
            },
            content="Preferred Beta.",
            observed_at=_T1,
        )
    )
    result = await memory.process(tenant_id="company_123", as_of=_T2)
    assert result.processed_at == _T2
    assert result.semantics.superseded >= 1
    episodes = await memory.list_episodes(tenant_id="company_123", include_inactive=True)
    assert episodes
    assert all(episode.updated_at == _T2 for episode in episodes)
    current = await memory.list_semantic_memories(tenant_id="company_123")
    historical = await memory.list_semantic_memories(tenant_id="company_123", valid_at=_T0)
    assert any(item.object_value.lower() == "beta" for item in current)
    assert any(item.object_value.lower() == "acme" for item in historical)
    assert all(item.updated_at == _T2 for item in current)
    revisions = await memory.list_semantic_revisions(tenant_id="company_123")
    assert any(revision.status is SemanticMemoryStatus.SUPERSEDED for revision in revisions)
    assert all(revision.updated_at == _T2 for revision in revisions)
