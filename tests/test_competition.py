"""Unit tests for observational cue-competition diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime

from cogkura.algorithms.competition import (
    CompetitionProfile,
    DeterministicCompetitionMatcher,
    apply_inspection_competition,
    competition_direction,
    effective_memory_time,
    subjects_compatible,
)
from cogkura.models import (
    ActivationComponents,
    CompetitionConfig,
    CompetitionDirection,
    MemoryIdentity,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RetrievalDiagnostics,
    SemanticCardinality,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
)

_T = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_T_EARLIER = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
_T_LATER = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _episode(
    *,
    memory_key: str = "ep-1",
    started_at: datetime = _T,
    created_at: datetime = _T,
    subject_id: str = "svc",
    statement: str = "Episode statement.",
) -> StoredEpisode:
    return StoredEpisode(
        id="episode-1",
        tenant_id="tenant",
        subject_id=subject_id,
        memory_key=memory_key,
        statement=statement,
        started_at=started_at,
        ended_at=started_at,
        confidence=0.9,
        importance=0.5,
        is_active=True,
        evidence=(),
        entities=(),
        metadata={},
        created_at=created_at,
        updated_at=created_at,
    )


def _semantic(
    *,
    memory_key: str = "sem-1",
    slot_key: str = "slot-1",
    predicate: str = "deployment_system",
    object_value: str = "jenkins",
    subject_entity_id: str = "payments-api",
    valid_from: datetime | None = _T,
    last_supported_at: datetime = _T,
    created_at: datetime = _T,
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id="semantic-1",
        tenant_id="tenant",
        subject_id="operator",
        memory_key=memory_key,
        slot_key=slot_key,
        revision_key="rev-1",
        revision_number=1,
        statement=f"{subject_entity_id} {predicate} {object_value}",
        subject_entity_id=subject_entity_id,
        predicate=predicate,
        object_value=object_value,
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers={},
        confidence=0.9,
        importance=0.5,
        status=SemanticMemoryStatus.ACTIVE,
        support_count=1,
        contradiction_count=0,
        first_supported_at=last_supported_at,
        last_supported_at=last_supported_at,
        valid_from=valid_from,
        valid_until=None,
        is_active=True,
        derivations=(),
        observation_evidence=(),
        entities=(),
        metadata={},
        created_at=created_at,
        updated_at=created_at,
    )


def _diagnostics(**overrides: object) -> RetrievalDiagnostics:
    base = {
        "rank_activation": 0.0,
        "accessibility_partial": 0.0,
        "ranking_partial": 0.0,
        "conjunction": 0.0,
        "text_coverage": 0.5,
        "text_cue_fit": 0.5,
        "temporal_mode": "current",
        "semantic_relevance": 0.5,
        "semantic_slot_key": None,
    }
    base.update(overrides)
    return RetrievalDiagnostics(**base)


def _profile(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    memory_kind: MemoryKind,
    slot_key: str | None = None,
    predicate: str | None = None,
    subject_entity_id: str | None = None,
    entity_ids: tuple[str, ...] = (),
    features: tuple[str, ...] = ("deploy",),
    cue_fit: float = 0.8,
    effective_cue_fit: float = 0.8,
    effective_time: datetime = _T,
    lineage_group: str | None = None,
) -> CompetitionProfile:
    return CompetitionProfile(
        identity=MemoryIdentity(memory_kind=memory_kind, memory_key=memory.memory_key),
        memory_kind=memory_kind,
        effective_time=effective_time,
        subject_id=memory.subject_id,
        subject_entity_id=subject_entity_id,
        semantic_slot_key=slot_key,
        predicate=predicate,
        entity_ids=entity_ids,
        retrieval_features=features,
        cue_fit=cue_fit,
        effective_cue_fit=effective_cue_fit,
        lineage_group=lineage_group,
    )


def test_effective_memory_time_episode_prefers_started_at() -> None:
    episode = _episode(started_at=_T_EARLIER, created_at=_T_LATER)
    assert effective_memory_time(episode, memory_kind=MemoryKind.EPISODE) == _T_EARLIER


def test_effective_memory_time_semantic_prefers_valid_from() -> None:
    semantic = _semantic(valid_from=_T_EARLIER, last_supported_at=_T, created_at=_T_LATER)
    assert effective_memory_time(semantic, memory_kind=MemoryKind.SEMANTIC) == _T_EARLIER


def test_competition_direction_classification() -> None:
    assert competition_direction(_T, _T_EARLIER) is CompetitionDirection.PROACTIVE
    assert competition_direction(_T, _T_LATER) is CompetitionDirection.RETROACTIVE
    assert competition_direction(_T, _T) is CompetitionDirection.CO_TEMPORAL


def test_same_slot_strength_is_symmetric() -> None:
    matcher = DeterministicCompetitionMatcher()
    config = CompetitionConfig()
    left = _profile(
        _semantic(memory_key="a", slot_key="slot-x"),
        memory_kind=MemoryKind.SEMANTIC,
        slot_key="slot-x",
        predicate="deployment_system",
        subject_entity_id="payments-api",
    )
    right = _profile(
        _semantic(memory_key="b", slot_key="slot-x", object_value="github_actions"),
        memory_kind=MemoryKind.SEMANTIC,
        slot_key="slot-x",
        predicate="deployment_system",
        subject_entity_id="payments-api",
        effective_time=_T_LATER,
    )
    ab = matcher.compare(left, right, config=config, cue_subject_id=None)
    ba = matcher.compare(right, left, config=config, cue_subject_id=None)
    assert ab is not None
    assert ba is not None
    assert ab.strength == ba.strength
    assert ab.relationship_strength == ba.relationship_strength
    assert ab.joint_cue_fit == ba.joint_cue_fit
    assert ab.direction is CompetitionDirection.RETROACTIVE
    assert ba.direction is CompetitionDirection.PROACTIVE


def test_entity_overlap_without_subject_is_rejected() -> None:
    matcher = DeterministicCompetitionMatcher()
    config = CompetitionConfig()
    left = _profile(
        _semantic(subject_entity_id="postgresql", predicate="production_database"),
        memory_kind=MemoryKind.SEMANTIC,
        predicate="production_database",
        subject_entity_id="postgresql",
        entity_ids=("postgresql",),
        features=("postgresql", "production", "database"),
    )
    right = _profile(
        _semantic(
            memory_key="sem-2",
            predicate="backup_schedule",
            object_value="six_hours",
            subject_entity_id="postgresql",
        ),
        memory_kind=MemoryKind.SEMANTIC,
        predicate="backup_schedule",
        subject_entity_id="postgresql",
        entity_ids=("postgresql",),
        features=("postgresql", "backup", "hours"),
    )
    assert matcher.compare(left, right, config=config, cue_subject_id=None) is None


def test_same_lineage_excluded() -> None:
    matcher = DeterministicCompetitionMatcher()
    config = CompetitionConfig()
    semantic = _semantic()
    left = _profile(
        semantic,
        memory_kind=MemoryKind.SEMANTIC,
        slot_key=semantic.slot_key,
        predicate=semantic.predicate,
        subject_entity_id=semantic.subject_entity_id,
        lineage_group="semantic:sem-1",
    )
    right = _profile(
        _episode(memory_key="ep-support"),
        memory_kind=MemoryKind.EPISODE,
        lineage_group="semantic:sem-1",
    )
    assert matcher.compare(left, right, config=config, cue_subject_id=None) is None


def test_subjects_compatible_requires_shared_subject() -> None:
    left = _profile(
        _semantic(subject_entity_id="payments-api"),
        memory_kind=MemoryKind.SEMANTIC,
        subject_entity_id="payments-api",
    )
    right = _profile(
        _semantic(memory_key="sem-2", subject_entity_id="identity-service"),
        memory_kind=MemoryKind.SEMANTIC,
        subject_entity_id="identity-service",
    )
    assert not subjects_compatible(left, right, cue_subject_id="payments-api")


def test_bounded_top_k_is_deterministic() -> None:
    config = CompetitionConfig(max_competitors_per_candidate=2, minimum_strength=0.1)
    matcher = DeterministicCompetitionMatcher()
    candidate = RecallInspectionCandidate(
        memory_kind=MemoryKind.SEMANTIC,
        memory=_semantic(memory_key="base"),
        disposition=RecallInspectionDisposition.RETURNED,
        activation=0.0,
        score=0.5,
        retrieval_threshold=-3.0,
        passed_threshold=True,
        soft_admitted=False,
        components=ActivationComponents(0, 0, 0, 0, 0),
        cognitive_traces=(),
        stored_traces=(),
        diagnostics=_diagnostics(
            semantic_slot_key="slot-1",
            semantic_relevance=0.9,
            text_cue_fit=0.9,
        ),
    )
    others = [
        RecallInspectionCandidate(
            memory_kind=MemoryKind.SEMANTIC,
            memory=_semantic(
                memory_key=f"sem-{index}",
                object_value=f"value-{index}",
            ),
            disposition=RecallInspectionDisposition.BELOW_THRESHOLD,
            activation=0.0,
            score=0.4,
            retrieval_threshold=-3.0,
            passed_threshold=False,
            soft_admitted=False,
            components=ActivationComponents(0, 0, 0, 0, 0),
            cognitive_traces=(),
            stored_traces=(),
            diagnostics=_diagnostics(
                semantic_slot_key="slot-1",
                semantic_relevance=0.9 - (index * 0.05),
                text_cue_fit=0.9 - (index * 0.05),
            ),
        )
        for index in range(4)
    ]
    updated, run = apply_inspection_competition(
        (candidate, *others),
        config=config,
        matcher=matcher,
        episode_slot_index={},
        cue_subject_id=None,
        retrieval_context_provided=False,
    )
    base = next(item for item in updated if item.memory.memory_key == "base")
    assert base.competition is not None
    assert base.competition.competitor_count == 2
    assert run.maximum_competitors_for_candidate == 2


def test_disabled_competition_returns_none_on_candidates() -> None:
    candidate = RecallInspectionCandidate(
        memory_kind=MemoryKind.SEMANTIC,
        memory=_semantic(),
        disposition=RecallInspectionDisposition.RETURNED,
        activation=0.0,
        score=0.5,
        retrieval_threshold=-3.0,
        passed_threshold=True,
        soft_admitted=False,
        components=ActivationComponents(0, 0, 0, 0, 0),
        cognitive_traces=(),
        stored_traces=(),
        diagnostics=_diagnostics(semantic_slot_key="slot-1"),
    )
    updated, run = apply_inspection_competition(
        (candidate,),
        config=CompetitionConfig(enabled=False),
        matcher=DeterministicCompetitionMatcher(),
        episode_slot_index={},
        cue_subject_id=None,
        retrieval_context_provided=False,
    )
    assert updated[0].competition is None
    assert run.candidate_count == 0
