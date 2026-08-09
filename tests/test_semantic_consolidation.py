"""Unit tests for semantic consolidation algorithm."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
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
    confidence: float = 0.9,
    importance: float = 0.7,
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
        confidence=confidence,
        importance=importance,
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
    confidence: float = 0.95,
    qualifiers: dict | None = None,
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
        confidence=confidence,
        observed_at=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        qualifiers=MappingProxyType(qualifiers or {"environment": "production"}),
    )


def test_one_episode_does_not_promote() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator()
    episodes = [_episode(episode_id="ep-1")]
    candidates = [_fact(episode_id="ep-1")]
    assert consolidator.consolidate(episodes, candidates) == []


def test_two_episodes_promote() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator()
    episodes = [_episode(episode_id="ep-1"), _episode(episode_id="ep-2")]
    candidates = [_fact(episode_id="ep-1"), _fact(episode_id="ep-2")]
    memories = consolidator.consolidate(episodes, candidates)
    assert len(memories) == 1
    assert memories[0].support_count == 2
    assert memories[0].status is SemanticMemoryStatus.ACTIVE


def test_duplicate_facts_within_episode_count_once() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [_episode(episode_id="ep-1")]
    candidates = [
        _fact(episode_id="ep-1", confidence=0.5),
        _fact(episode_id="ep-1", confidence=0.9),
    ]
    memories = consolidator.consolidate(episodes, candidates)
    assert len(memories) == 1
    assert memories[0].support_count == 1


def test_cardinality_one_contradiction() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [
        _episode(episode_id="ep-1"),
        _episode(episode_id="ep-2"),
        _episode(episode_id="ep-3"),
    ]
    candidates = [
        _fact(episode_id="ep-1"),
        _fact(episode_id="ep-2"),
        _fact(episode_id="ep-3", object_value="mysql", object_entity_id="mysql"),
    ]
    memories = consolidator.consolidate(episodes, candidates)
    postgres = next(m for m in memories if m.object_value == "postgresql")
    assert postgres.contradiction_count >= 1


def test_cardinality_many_coexists() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [_episode(episode_id="ep-1"), _episode(episode_id="ep-2")]
    candidates = [
        _fact(episode_id="ep-1", object_value="postgresql", cardinality=SemanticCardinality.MANY),
        _fact(
            episode_id="ep-2",
            object_value="redis",
            object_entity_id="redis",
            cardinality=SemanticCardinality.MANY,
        ),
    ]
    memories = consolidator.consolidate(episodes, candidates)
    assert len(memories) == 2


def test_consolidation_is_deterministic() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator()
    episodes = [_episode(episode_id="ep-1"), _episode(episode_id="ep-2")]
    candidates = [_fact(episode_id="ep-1"), _fact(episode_id="ep-2")]
    first = consolidator.consolidate(episodes, candidates)
    second = consolidator.consolidate(list(reversed(episodes)), list(reversed(candidates)))
    assert first[0].memory_key == second[0].memory_key
    assert first[0].statement == second[0].statement
