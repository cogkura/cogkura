"""Deterministic evaluation fixtures for semantic consolidation."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cognema import Memory, ObservationInput
from cognema.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cognema.models import (
    EpisodeEvidenceInput,
    SemanticCardinality,
    SemanticFactCandidate,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
)


def _episode(
    *,
    episode_id: str,
    source_namespaces: tuple[str, ...] = ("chat.messages",),
) -> StoredEpisode:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    return StoredEpisode(
        id=episode_id,
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=f"key-{episode_id}",
        statement="Episode statement.",
        started_at=now,
        ended_at=now,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{episode_id}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({"episode": {"source_namespaces": list(source_namespaces)}}),
        created_at=now,
        updated_at=now,
    )


def _fact(
    *,
    episode_id: str,
    object_value: str = "postgresql",
    object_entity_id: str = "postgresql",
    polarity: SemanticPolarity = SemanticPolarity.AFFIRM,
    cardinality: SemanticCardinality = SemanticCardinality.ONE,
) -> SemanticFactCandidate:
    return SemanticFactCandidate(
        tenant_id="company_123",
        source_episode_id=episode_id,
        subject_entity_id="customer_42",
        predicate="preferred_database",
        object_value=object_value,
        object_entity_id=object_entity_id,
        polarity=polarity,
        cardinality=cardinality,
        confidence=0.9,
        observed_at=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        qualifiers=MappingProxyType({"environment": "production"}),
    )


def test_fixture_stable_preference_promotes_with_two_episodes() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator()
    episodes = [_episode(episode_id="ep-1"), _episode(episode_id="ep-2")]
    candidates = [_fact(episode_id="ep-1"), _fact(episode_id="ep-2")]
    memories = consolidator.consolidate(episodes, candidates)
    assert len(memories) == 1
    assert memories[0].status is SemanticMemoryStatus.ACTIVE
    assert memories[0].support_count == 2
    assert 0.5 <= memories[0].confidence <= 1.0


def test_fixture_changing_preference_produces_distinct_claims() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [_episode(episode_id="ep-1"), _episode(episode_id="ep-2")]
    candidates = [
        _fact(episode_id="ep-1", object_value="postgresql", object_entity_id="postgresql"),
        _fact(episode_id="ep-2", object_value="mysql", object_entity_id="mysql"),
    ]
    memories = consolidator.consolidate(episodes, candidates)
    values = {memory.object_value for memory in memories}
    assert values == {"postgresql", "mysql"}


def test_fixture_negation_opposes_affirmation() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [_episode(episode_id="ep-1"), _episode(episode_id="ep-2")]
    candidates = [
        _fact(episode_id="ep-1", polarity=SemanticPolarity.AFFIRM),
        _fact(episode_id="ep-2", polarity=SemanticPolarity.DENY),
    ]
    memories = consolidator.consolidate(episodes, candidates)
    affirm = next(m for m in memories if m.polarity is SemanticPolarity.AFFIRM)
    assert affirm.contradiction_count >= 1


def test_fixture_multi_valued_cardinality_coexists() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [_episode(episode_id="ep-1"), _episode(episode_id="ep-2")]
    candidates = [
        _fact(
            episode_id="ep-1",
            object_value="postgresql",
            cardinality=SemanticCardinality.MANY,
        ),
        _fact(
            episode_id="ep-2",
            object_value="redis",
            object_entity_id="redis",
            cardinality=SemanticCardinality.MANY,
        ),
    ]
    memories = consolidator.consolidate(episodes, candidates)
    assert len(memories) == 2


def test_fixture_source_disagreement_increases_contestation() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [
        _episode(episode_id="ep-1", source_namespaces=("chat.messages",)),
        _episode(episode_id="ep-2", source_namespaces=("crm.notes",)),
        _episode(episode_id="ep-3", source_namespaces=("chat.messages",)),
    ]
    candidates = [
        _fact(episode_id="ep-1"),
        _fact(episode_id="ep-2"),
        _fact(episode_id="ep-3", object_value="mysql", object_entity_id="mysql"),
    ]
    memories = consolidator.consolidate(episodes, candidates)
    postgres = next(m for m in memories if m.object_value == "postgresql")
    assert postgres.contradiction_count >= 1


def _observation(
    *,
    source_record_id: str,
    conversation_id: str,
    semantic_fact: dict,
) -> ObservationInput:
    return ObservationInput(
        tenant_id="company_123",
        subject_id="customer_42",
        source_namespace="chat.messages",
        source_record_id=source_record_id,
        event_type="message",
        content="Database preference.",
        observed_at=datetime.now(UTC),
        metadata={
            "conversation_id": conversation_id,
            "semantic_facts": [semantic_fact],
        },
    )


@pytest.mark.asyncio
async def test_fixture_observation_revision_updates_semantic_memory() -> None:
    memory = Memory()
    fact = {
        "predicate": "preferred_database",
        "object_value": "postgresql",
        "object_entity_id": "postgresql",
        "cardinality": "one",
        "polarity": "affirm",
    }
    await memory.observe(
        _observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact=fact,
        )
    )
    await memory.observe(
        _observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact=fact,
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    first = await memory.consolidate_semantics(tenant_id="company_123")
    second = await memory.consolidate_semantics(tenant_id="company_123")
    assert first.created == 1
    assert second.unchanged == 1


@pytest.mark.asyncio
async def test_fixture_deletion_deactivates_semantic_memory() -> None:
    memory = Memory()
    fact = {
        "predicate": "preferred_database",
        "object_value": "postgresql",
        "object_entity_id": "postgresql",
        "cardinality": "one",
        "polarity": "affirm",
    }
    await memory.observe(
        _observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact=fact,
        )
    )
    await memory.observe(
        _observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact=fact,
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    await memory.consolidate_semantics(tenant_id="company_123")
    await memory.clear(tenant_id="company_123")
    assert await memory.list_semantic_memories(tenant_id="company_123") == []
