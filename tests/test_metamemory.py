"""Unit tests for deterministic metamemory monitoring."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from cogkura.algorithms.metamemory import DeterministicMemoryMonitor
from cogkura.models import (
    ActivationComponents,
    ActivationConfig,
    EpisodeEvidenceInput,
    MemoryAssessmentFlag,
    MemoryIdentity,
    MemoryKind,
    MetamemoryConfig,
    RecallResult,
    RetrievalCue,
    SemanticCardinality,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
)

_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
_MONITOR = DeterministicMemoryMonitor()
_ACTIVATION_CONFIG = ActivationConfig()


def _episode(
    *,
    memory_key: str = "episode-a",
    statement: str = "Episode statement.",
    confidence: float = 0.9,
    observation_id: str = "obs-1",
    ended_at: datetime = _NOW,
) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        statement=statement,
        started_at=ended_at,
        ended_at=ended_at,
        confidence=confidence,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=observation_id,
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({}),
        created_at=ended_at,
        updated_at=ended_at,
    )


def _semantic(
    *,
    memory_key: str = "semantic-a",
    statement: str = "Semantic statement.",
    confidence: float = 0.9,
    status: SemanticMemoryStatus = SemanticMemoryStatus.ACTIVE,
    support_count: int = 9,
    contradiction_count: int = 0,
    observation_id: str = "obs-semantic",
    last_supported_at: datetime = _NOW,
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        slot_key="slot",
        revision_key="rev",
        revision_number=1,
        statement=statement,
        subject_entity_id=None,
        predicate="predicate",
        object_value="object",
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers=MappingProxyType({}),
        confidence=confidence,
        importance=0.7,
        status=status,
        support_count=support_count,
        contradiction_count=contradiction_count,
        first_supported_at=last_supported_at,
        last_supported_at=last_supported_at,
        valid_from=None,
        valid_until=None,
        is_active=True,
        derivations=(),
        observation_evidence=(
            EpisodeEvidenceInput(
                observation_id=observation_id,
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({}),
        created_at=last_supported_at,
        updated_at=last_supported_at,
    )


def _recall(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    score: float,
    base_level: float = 0.0,
) -> RecallResult:
    memory_kind = MemoryKind.EPISODE if isinstance(memory, StoredEpisode) else MemoryKind.SEMANTIC
    return RecallResult(
        memory_kind=memory_kind,
        memory=memory,
        activation=score,
        score=score,
        latency_seconds=0.1,
        components=ActivationComponents(
            base_level=base_level,
            spreading=0.0,
            partial_match=0.0,
            noise=0.0,
            total=score,
        ),
        reason="test",
    )


def _assess(
    candidates: list[RecallResult],
    *,
    config: MetamemoryConfig | None = None,
    learning_utilities: dict[MemoryIdentity, float] | None = None,
    as_of: datetime = _NOW,
) -> object:
    return _MONITOR.assess(
        candidates=candidates,
        query=RetrievalCue(text="test query"),
        goal=RetrievalCue(text="test goal"),
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=as_of,
        valid_at=None,
        config=config or MetamemoryConfig(),
        activation_config=_ACTIVATION_CONFIG,
        learning_utilities=learning_utilities,
    )


def test_empty_assessment() -> None:
    assessment = _assess([])
    assert assessment.retrieved_count == 0
    assert assessment.signals.cue_coverage == 0.0
    assert assessment.signals.top_retrieval_strength == 0.0
    assert assessment.signals.evidence_confidence is None
    assert assessment.flags == (MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,)


def test_retrieval_strength_aggregates() -> None:
    candidates = [
        _recall(_episode(memory_key="a"), score=0.9),
        _recall(_episode(memory_key="b"), score=0.7),
        _recall(_episode(memory_key="c"), score=0.5),
    ]
    assessment = _assess(candidates)
    assert assessment.signals.top_retrieval_strength == 0.9
    assert math.isclose(assessment.signals.mean_retrieval_strength, 0.7)


def test_retrieval_strength_not_confidence() -> None:
    episode = _episode(confidence=0.3)
    assessment = _assess([_recall(episode, score=0.95)])
    assert assessment.signals.top_retrieval_strength == 0.95
    assert assessment.signals.evidence_confidence == 0.3


def test_weighted_evidence_confidence() -> None:
    high = _recall(_episode(memory_key="high", confidence=0.9), score=0.9)
    low = _recall(_episode(memory_key="low", confidence=0.1), score=0.1)
    assessment = _assess([high, low])
    assert assessment.signals.evidence_confidence is not None
    assert assessment.signals.evidence_confidence > 0.8


def test_contested_semantic_conflict() -> None:
    semantic = _semantic(status=SemanticMemoryStatus.CONTESTED)
    assessment = _assess([_recall(semantic, score=0.9)])
    assert assessment.signals.semantic_conflict >= 0.9
    config = MetamemoryConfig(semantic_conflict_threshold=0.25)
    flagged = _assess([_recall(semantic, score=0.9)], config=config)
    assert MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY in flagged.flags


def test_low_level_contradiction() -> None:
    semantic = _semantic(
        status=SemanticMemoryStatus.ACTIVE,
        support_count=9,
        contradiction_count=1,
    )
    item = _assess([_recall(semantic, score=0.5)]).items[0]
    assert 0.0 < item.semantic_conflict < 1.0


def test_superseded_not_conflict() -> None:
    semantic = _semantic(
        status=SemanticMemoryStatus.SUPERSEDED,
        contradiction_count=0,
    )
    item = _assess([_recall(semantic, score=0.8)]).items[0]
    assert item.semantic_conflict == 0.0


def test_provenance_diversity() -> None:
    one = _assess([_recall(_episode(memory_key="a", observation_id="obs-1"), score=0.8)])
    two = _assess(
        [
            _recall(_episode(memory_key="a", observation_id="obs-1"), score=0.8),
            _recall(_episode(memory_key="b", observation_id="obs-2"), score=0.7),
        ]
    )
    assert one.signals.provenance_diversity == 0.0
    assert two.signals.provenance_diversity > one.signals.provenance_diversity


def test_duplicate_provenance_not_inflated() -> None:
    assessment = _assess(
        [
            _recall(_episode(memory_key="a", observation_id="obs-1"), score=0.8),
            _recall(_episode(memory_key="b", observation_id="obs-1"), score=0.7),
        ]
    )
    assert assessment.distinct_observation_count == 1


def test_freshness_disabled_by_default() -> None:
    old = _NOW - timedelta(days=30)
    assessment = _assess([_recall(_episode(ended_at=old), score=0.8)], as_of=_NOW)
    assert assessment.signals.freshness is None
    assert MemoryAssessmentFlag.STALE_EVIDENCE not in assessment.flags


def test_freshness_enabled_half_life() -> None:
    old = _NOW - timedelta(days=1)
    config = MetamemoryConfig(freshness_half_life_seconds=86400.0)
    assessment = _assess(
        [_recall(_episode(ended_at=old), score=0.8)],
        config=config,
        as_of=_NOW,
    )
    assert assessment.signals.freshness is not None
    assert math.isclose(assessment.signals.freshness, 0.5, rel_tol=0.01)


def test_forgetting_pressure_from_base_level() -> None:
    high_retention = _recall(_episode(memory_key="high"), score=0.8, base_level=2.0)
    low_retention = _recall(_episode(memory_key="low"), score=0.8, base_level=-2.0)
    high_item = _assess([high_retention]).items[0]
    low_item = _assess([low_retention]).items[0]
    assert low_item.forgetting_pressure > high_item.forgetting_pressure


def test_learned_utility_neutral_and_disabled() -> None:
    episode = _episode()
    recall = _recall(episode, score=0.8)
    enabled = _assess([recall], learning_utilities={})
    assert enabled.signals.learned_utility == 0.5
    disabled = _assess([recall], learning_utilities=None)
    assert disabled.signals.learned_utility is None


def test_item_bound() -> None:
    candidates = [
        _recall(_episode(memory_key=f"ep-{index}"), score=0.5 + index * 0.01) for index in range(50)
    ]
    config = MetamemoryConfig(candidate_pool_size=50, max_report_items=8)
    assessment = _assess(candidates, config=config)
    assert len(assessment.items) == 8
    assert assessment.retrieved_count == 50


def test_flag_ordering() -> None:
    config = MetamemoryConfig(
        low_cue_coverage_threshold=0.0,
        low_retrieval_strength_threshold=0.0,
        low_evidence_confidence_threshold=0.0,
        semantic_conflict_threshold=0.0,
        low_provenance_diversity_threshold=1.0,
        forgetting_pressure_threshold=0.0,
        low_learned_utility_threshold=0.0,
        freshness_half_life_seconds=86400.0,
        stale_evidence_threshold=1.0,
    )
    semantic = _semantic(status=SemanticMemoryStatus.CONTESTED)
    old = _NOW - timedelta(days=10)
    candidates = [_recall(_episode(ended_at=old), score=0.1), _recall(semantic, score=0.9)]
    assessment = _assess(
        candidates,
        config=config,
        learning_utilities={},
        as_of=_NOW,
    )
    expected_prefix = [
        MemoryAssessmentFlag.LOW_CUE_COVERAGE,
        MemoryAssessmentFlag.LOW_RETRIEVAL_STRENGTH,
        MemoryAssessmentFlag.LOW_EVIDENCE_CONFIDENCE,
        MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY,
        MemoryAssessmentFlag.LOW_PROVENANCE_DIVERSITY,
        MemoryAssessmentFlag.HIGH_FORGETTING_PRESSURE,
        MemoryAssessmentFlag.LOW_LEARNED_UTILITY,
        MemoryAssessmentFlag.STALE_EVIDENCE,
    ]
    assert list(assessment.flags) == [flag for flag in expected_prefix if flag in assessment.flags]
