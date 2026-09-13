"""Unit tests for observational cue-competition diagnostics."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from cogkura.algorithms.competition import (
    CompetitionProfile,
    DeterministicCompetitionMatcher,
    _compute_interference,
    apply_inspection_competition,
    competition_direction,
    effective_memory_time,
    subjects_compatible,
)
from cogkura.models import (
    ActivationComponents,
    CompetitionConfig,
    CompetitionDirection,
    CompetitionEvidence,
    MemoryIdentity,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RetrievalDiagnostics,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
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
    ab = matcher.compare(left, right, config=config, cue_subject_id=None, cue_entity_ids=())
    ba = matcher.compare(right, left, config=config, cue_subject_id=None, cue_entity_ids=())
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
    assert (
        matcher.compare(left, right, config=config, cue_subject_id=None, cue_entity_ids=()) is None
    )


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
    assert (
        matcher.compare(left, right, config=config, cue_subject_id=None, cue_entity_ids=()) is None
    )


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
    assert not subjects_compatible(left, right, cue_subject_id="payments-api", cue_entity_ids=())


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
        cue_entity_ids=(),
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
        cue_entity_ids=(),
        retrieval_context_provided=False,
    )
    assert updated[0].competition is None
    assert run.candidate_count == 0


def test_shared_query_anchor_enables_subject_compatibility() -> None:
    left = _profile(
        _semantic(subject_entity_id="payments-api"),
        memory_kind=MemoryKind.SEMANTIC,
        subject_entity_id="payments-api",
        entity_ids=("payments-api", "shared-anchor"),
    )
    right = _profile(
        _semantic(memory_key="sem-2", subject_entity_id="identity-service"),
        memory_kind=MemoryKind.SEMANTIC,
        subject_entity_id="identity-service",
        entity_ids=("identity-service", "shared-anchor"),
    )
    assert subjects_compatible(
        left,
        right,
        cue_subject_id=None,
        cue_entity_ids=("shared-anchor",),
    )


def test_broad_subject_id_does_not_match_conflicting_entities() -> None:
    left = _profile(
        replace(_semantic(subject_entity_id="payments-api"), subject_id="operator"),
        memory_kind=MemoryKind.SEMANTIC,
        subject_entity_id="payments-api",
    )
    right = _profile(
        replace(
            _semantic(memory_key="sem-2", subject_entity_id="identity-service"),
            subject_id="operator",
        ),
        memory_kind=MemoryKind.SEMANTIC,
        subject_entity_id="identity-service",
    )
    assert not subjects_compatible(left, right, cue_subject_id=None, cue_entity_ids=())


def test_effective_memory_time_uses_support_episode_chronology() -> None:
    episode = _episode(
        memory_key="ep-support",
        started_at=_T_EARLIER,
        created_at=_T_LATER,
    )
    semantic = _semantic(
        valid_from=None,
        last_supported_at=_T_LATER,
        created_at=_T_LATER,
    )
    semantic = replace(
        semantic,
        derivations=(
            SemanticDerivationInput(
                episode_id="episode-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=1.0,
            ),
        ),
    )
    assert (
        effective_memory_time(
            semantic,
            memory_kind=MemoryKind.SEMANTIC,
            episode_by_id={"episode-1": episode},
        )
        == _T_EARLIER
    )


def test_noisy_or_interference_formula() -> None:
    profile = _profile(_semantic(), memory_kind=MemoryKind.SEMANTIC, slot_key="slot-1")
    competitor = MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="other")
    evidence = CompetitionEvidence(
        competitor_identity=competitor,
        direction=CompetitionDirection.PROACTIVE,
        strength=0.5,
        candidate_cue_fit=0.8,
        competitor_cue_fit=0.8,
        same_subject=True,
        same_semantic_slot=True,
        same_predicate=True,
        shared_entity_ids=("payments-api",),
        shared_features=("deploy",),
        relationship_strength=0.95,
        joint_cue_fit=0.8,
    )
    config = CompetitionConfig(
        apply_interference=True,
        proactive_weight=0.2,
        retroactive_weight=0.2,
    )
    interference = _compute_interference(
        profile,
        (evidence,),
        config=config,
        accessibility_by_identity={competitor: -100.0},
        retrieval_threshold=-3.0,
    )
    assert interference.proactive_pressure == 0.0
    assert interference.total_penalty == 0.0

    interference = _compute_interference(
        profile,
        (evidence,),
        config=config,
        accessibility_by_identity={competitor: 2.0},
        retrieval_threshold=-3.0,
    )
    assert interference.proactive_pressure > 0.0
    assert interference.total_penalty < 0.0


def test_co_temporal_competition_has_zero_interference_pressure() -> None:
    profile = _profile(_semantic(), memory_kind=MemoryKind.SEMANTIC)
    competitor = MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="other")
    evidence = CompetitionEvidence(
        competitor_identity=competitor,
        direction=CompetitionDirection.CO_TEMPORAL,
        strength=0.9,
        candidate_cue_fit=0.9,
        competitor_cue_fit=0.9,
        same_subject=True,
        same_semantic_slot=True,
        same_predicate=True,
        shared_entity_ids=("payments-api",),
        shared_features=("deploy",),
        relationship_strength=0.95,
        joint_cue_fit=0.9,
    )
    interference = _compute_interference(
        profile,
        (evidence,),
        config=CompetitionConfig(apply_interference=True),
        accessibility_by_identity={competitor: 5.0},
        retrieval_threshold=-3.0,
    )
    assert interference.total_penalty == 0.0
