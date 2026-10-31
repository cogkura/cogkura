"""Unit tests for semantic consolidation algorithm."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType

from cogkura.algorithms.reconsolidation import DeterministicSemanticReconciler
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    EpisodeEvidenceInput,
    SemanticAuthorityBasis,
    SemanticCardinality,
    SemanticFactAuthority,
    SemanticFactCandidate,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticPromotionBasis,
    StoredEpisode,
)

_AS_OF = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def _reconcile(candidates):
    plan = DeterministicSemanticReconciler().reconcile(
        candidates=candidates,
        existing_memories=(),
        existing_revisions=(),
        as_of=_AS_OF,
    )
    return plan.current_memories


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
    revisions = consolidator.consolidate(episodes, candidates)
    assert len(revisions) == 1
    assert revisions[0].support_count == 2
    memories = _reconcile(revisions)
    assert len(memories) == 1
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
    memories = _reconcile(consolidator.consolidate(episodes, candidates))
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


def _authorised(fact: SemanticFactCandidate, *, observation_id: str) -> SemanticFactCandidate:
    return replace(
        fact,
        source_observation_id=observation_id,
        source_observation_revision=1,
        authority=SemanticFactAuthority(basis=SemanticAuthorityBasis.APPLICATION_ASSERTED),
    )


def test_authority_flag_off_does_not_promote_one_episode() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator()
    episodes = [_episode(episode_id="ep-1")]
    candidates = [_authorised(_fact(episode_id="ep-1"), observation_id="obs-1")]
    assert consolidator.consolidate(episodes, candidates) == []


def test_verified_authority_promotes_one_episode_without_raising_confidence() -> None:
    episode = _episode(episode_id="ep-1")
    ordinary = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    authorised = ComplementaryLearningSemanticConsolidator(allow_authoritative_single_episode=True)
    fact = _fact(episode_id="ep-1", confidence=0.8)
    baseline = ordinary.consolidate([episode], [fact])[0]
    promoted = authorised.consolidate(
        [episode],
        [_authorised(fact, observation_id="obs-authority")],
    )[0]
    assert promoted.support_count == 1
    assert promoted.support_confidence == baseline.support_confidence
    assert promoted.support_confidence < 1.0
    promotion = promoted.metadata["semantic"]["promotion"]
    assert promotion["basis"] == SemanticPromotionBasis.EXPLICIT_SOURCE_AUTHORITY.value
    assert promotion["observation_id"] == "obs-authority"
    assert promotion["support_count"] == 1
    assert baseline.metadata["semantic"]["promotion"]["basis"] == (
        SemanticPromotionBasis.RECURRENCE.value
    )


def test_lower_confidence_authority_is_kept_for_the_same_claim() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True
    )
    episodes = [_episode(episode_id="ep-1")]
    ordinary = _fact(episode_id="ep-1", confidence=1.0)
    authorised = _authorised(
        _fact(episode_id="ep-1", confidence=0.4),
        observation_id="obs-low",
    )
    first = consolidator.consolidate(episodes, [ordinary, authorised])
    second = consolidator.consolidate(episodes, [authorised, ordinary])
    assert first[0].metadata["semantic"]["promotion"]["observation_id"] == "obs-low"
    assert second[0].metadata["semantic"]["promotion"]["observation_id"] == "obs-low"
    assert first[0].support_confidence == second[0].support_confidence
