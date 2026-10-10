"""Unit tests for behavioural competition eligibility policy."""

from __future__ import annotations

from datetime import UTC, datetime

from cogkura.algorithms.behavioral_competition import (
    DeterministicBehavioralCompetitionPolicy,
    build_behavioral_query_scope,
)
from cogkura.algorithms.competition import CompetitionProfile
from cogkura.models import (
    BehavioralEligibilityReason,
    BehavioralQueryScope,
    BehavioralStructuralAnchor,
    CompetitionConfig,
    CompetitionDirection,
    CompetitionEvidence,
    MemoryIdentity,
    MemoryKind,
    RetrievalCue,
    SemanticCardinality,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredSemanticMemory,
)

_T = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_POLICY = DeterministicBehavioralCompetitionPolicy()
_CONFIG = CompetitionConfig(
    minimum_strength=0.3,
    minimum_behavioral_strength=0.6,
    minimum_behavioral_cue_fit=0.4,
)


def _semantic(
    *,
    memory_key: str,
    slot_key: str,
    predicate: str,
    object_value: str,
    subject_entity_id: str,
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id=f"id-{memory_key}",
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
        first_supported_at=_T,
        last_supported_at=_T,
        valid_from=_T,
        valid_until=None,
        is_active=True,
        derivations=(),
        observation_evidence=(),
        entities=(),
        metadata={},
        created_at=_T,
        updated_at=_T,
    )


def _profile(
    memory: StoredSemanticMemory,
    *,
    entity_ids: tuple[str, ...],
    features: tuple[str, ...],
    slot_key: str | None = None,
    predicate: str | None = None,
    subject_entity_id: str | None = None,
    lineage_group: str | None = None,
) -> CompetitionProfile:
    return CompetitionProfile(
        identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key=memory.memory_key),
        memory_kind=MemoryKind.SEMANTIC,
        effective_time=_T,
        subject_id=memory.subject_id,
        subject_entity_id=subject_entity_id or memory.subject_entity_id,
        semantic_slot_key=slot_key or memory.slot_key,
        predicate=predicate or memory.predicate,
        entity_ids=entity_ids,
        retrieval_features=features,
        cue_fit=0.85,
        effective_cue_fit=0.85,
        lineage_group=lineage_group,
    )


def _evidence(
    *,
    direction: CompetitionDirection = CompetitionDirection.PROACTIVE,
    strength: float = 0.8,
    relationship_strength: float = 0.9,
    candidate_cue_fit: float = 0.85,
    competitor_cue_fit: float = 0.85,
    same_subject: bool = True,
    same_semantic_slot: bool = False,
    same_predicate: bool = False,
    competitor_key: str = "competitor",
) -> CompetitionEvidence:
    return CompetitionEvidence(
        competitor_identity=MemoryIdentity(
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=competitor_key,
        ),
        direction=direction,
        strength=strength,
        candidate_cue_fit=candidate_cue_fit,
        competitor_cue_fit=competitor_cue_fit,
        same_subject=same_subject,
        same_semantic_slot=same_semantic_slot,
        same_predicate=same_predicate,
        shared_entity_ids=("payments-api",),
        shared_features=("deployment", "system"),
        relationship_strength=relationship_strength,
        joint_cue_fit=0.85,
    )


def _evaluate(
    candidate: CompetitionProfile,
    competitor: CompetitionProfile,
    evidence: CompetitionEvidence,
    query_scope: BehavioralQueryScope,
) -> BehavioralEligibilityReason:
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=evidence,
        query_scope=query_scope,
        config=_CONFIG,
    )
    return result.reason if not result.eligible else result.reason


def test_same_lineage_rejected() -> None:
    left = _profile(
        _semantic(
            memory_key="left",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
        lineage_group="lineage-1",
    )
    right = _profile(
        _semantic(
            memory_key="right",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="gha",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
        lineage_group="lineage-1",
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api deployment_system"))
    reason = _evaluate(left, right, _evidence(same_semantic_slot=True, same_predicate=True), scope)
    assert reason is BehavioralEligibilityReason.SAME_LINEAGE


def test_co_temporal_rejected() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="gha",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api deployment_system"))
    reason = _evaluate(
        candidate,
        competitor,
        _evidence(
            direction=CompetitionDirection.CO_TEMPORAL,
            same_semantic_slot=True,
            same_predicate=True,
        ),
        scope,
    )
    assert reason is BehavioralEligibilityReason.NON_BEHAVIORAL_DIRECTION
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=_evidence(
            direction=CompetitionDirection.CO_TEMPORAL,
            same_semantic_slot=True,
            same_predicate=True,
        ),
        query_scope=scope,
        config=_CONFIG,
    )
    assert result.scope_eligible
    assert not result.eligible
    assert result.structural_anchor is BehavioralStructuralAnchor.SEMANTIC_SLOT


def test_weak_strength_rejected() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="gha",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api deployment_system"))
    reason = _evaluate(
        candidate,
        competitor,
        _evidence(
            strength=0.5,
            relationship_strength=0.5,
            same_semantic_slot=True,
            same_predicate=True,
        ),
        scope,
    )
    assert reason is BehavioralEligibilityReason.COMPETITION_TOO_WEAK


def test_weak_candidate_cue_fit_rejected() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="gha",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api deployment_system"))
    reason = _evaluate(
        candidate,
        competitor,
        _evidence(
            candidate_cue_fit=0.2,
            same_semantic_slot=True,
            same_predicate=True,
        ),
        scope,
    )
    assert reason is BehavioralEligibilityReason.CANDIDATE_CUE_FIT_TOO_WEAK


def test_same_semantic_slot_eligible() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-deploy",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
        slot_key="slot-deploy",
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-deploy",
            predicate="deployment_system",
            object_value="gha",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
        slot_key="slot-deploy",
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api deployment_system"))
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=_evidence(same_semantic_slot=True, same_predicate=True),
        query_scope=scope,
        config=_CONFIG,
    )
    assert result.eligible
    assert result.reason is BehavioralEligibilityReason.SAME_SEMANTIC_SLOT
    assert result.structural_anchor is BehavioralStructuralAnchor.SEMANTIC_SLOT


def test_same_subject_predicate_eligible() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-a",
            predicate="favorite_color",
            object_value="blue",
            subject_entity_id="customer-1",
        ),
        entity_ids=("customer-1",),
        features=("customer", "favorite", "color", "blue"),
        predicate="favorite_color",
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-b",
            predicate="favorite_color",
            object_value="green",
            subject_entity_id="customer-1",
        ),
        entity_ids=("customer-1",),
        features=("customer", "favorite", "color", "green"),
        predicate="favorite_color",
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="customer-1 favorite_color"))
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=_evidence(same_predicate=True),
        query_scope=scope,
        config=_CONFIG,
    )
    assert result.eligible
    assert result.reason is BehavioralEligibilityReason.SAME_SUBJECT_PREDICATE


def test_same_predicate_different_subject_rejected() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-b",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="identity-service",
        ),
        entity_ids=("identity-service",),
        features=("identity", "service", "deployment", "system"),
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api deployment_system"))
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=_evidence(same_predicate=True, same_subject=False),
        query_scope=scope,
        config=_CONFIG,
    )
    assert not result.eligible
    assert result.reason in {
        BehavioralEligibilityReason.NO_QUERY_SCOPE_ANCHOR,
        BehavioralEligibilityReason.ENTITY_OVERLAP_ONLY,
        BehavioralEligibilityReason.FEATURE_OVERLAP_ONLY,
    }


def test_query_anchored_episode_eligible() -> None:
    candidate = _profile(
        _semantic(
            memory_key="ep-a",
            slot_key="",
            predicate="",
            object_value="",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "incident", "rollback"),
        slot_key=None,
        predicate=None,
    )
    competitor = _profile(
        _semantic(
            memory_key="ep-b",
            slot_key="",
            predicate="",
            object_value="",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "incident", "rollback", "jenkins"),
        slot_key=None,
        predicate=None,
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api incident rollback"))
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=_evidence(competitor_key="ep-b"),
        query_scope=scope,
        config=_CONFIG,
    )
    assert result.eligible
    assert result.reason is BehavioralEligibilityReason.QUERY_ANCHORED_COMPETITION


def test_entity_overlap_only_rejected() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api", "shared"),
        features=("payments", "api", "deployment"),
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-b",
            predicate="database_engine",
            object_value="postgresql",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api", "shared"),
        features=("payments", "api", "database"),
    )
    scope = build_behavioral_query_scope(
        RetrievalCue(text="payments-api database_engine", entity_ids=("payments-api",))
    )
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=_evidence(same_predicate=False, same_subject=True),
        query_scope=scope,
        config=_CONFIG,
    )
    assert not result.eligible
    assert result.reason is BehavioralEligibilityReason.NO_SHARED_QUERY_FEATURE


def test_explicit_entity_anchor_miss_rejected() -> None:
    candidate = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-a",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
    )
    competitor = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-b",
            predicate="deployment_system",
            object_value="gha",
            subject_entity_id="identity-service",
        ),
        entity_ids=("identity-service",),
        features=("identity", "service", "deployment", "system"),
    )
    scope = BehavioralQueryScope(
        subject_id=None,
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
        predicate=None,
    )
    result = _POLICY.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=_evidence(same_predicate=True),
        query_scope=scope,
        config=_CONFIG,
    )
    assert not result.eligible
    assert result.reason is BehavioralEligibilityReason.NO_QUERY_SCOPE_ANCHOR


def test_input_order_invariance() -> None:
    left = _profile(
        _semantic(
            memory_key="a",
            slot_key="slot-deploy",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
        slot_key="slot-deploy",
    )
    right = _profile(
        _semantic(
            memory_key="b",
            slot_key="slot-deploy",
            predicate="deployment_system",
            object_value="gha",
            subject_entity_id="payments-api",
        ),
        entity_ids=("payments-api",),
        features=("payments", "api", "deployment", "system"),
        slot_key="slot-deploy",
    )
    scope = build_behavioral_query_scope(RetrievalCue(text="payments-api deployment_system"))
    evidence = _evidence(same_semantic_slot=True, same_predicate=True)
    forward = _POLICY.evaluate(
        candidate=left,
        competitor=right,
        evidence=evidence,
        query_scope=scope,
        config=_CONFIG,
    )
    reverse = _POLICY.evaluate(
        candidate=right,
        competitor=left,
        evidence=replace_evidence_competitor(evidence, "a"),
        query_scope=scope,
        config=_CONFIG,
    )
    assert forward.eligible == reverse.eligible
    assert forward.reason == reverse.reason


def replace_evidence_competitor(
    evidence: CompetitionEvidence,
    competitor_key: str,
) -> CompetitionEvidence:
    return CompetitionEvidence(
        competitor_identity=MemoryIdentity(
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=competitor_key,
        ),
        direction=evidence.direction,
        strength=evidence.strength,
        candidate_cue_fit=evidence.candidate_cue_fit,
        competitor_cue_fit=evidence.competitor_cue_fit,
        same_subject=evidence.same_subject,
        same_semantic_slot=evidence.same_semantic_slot,
        same_predicate=evidence.same_predicate,
        shared_entity_ids=evidence.shared_entity_ids,
        shared_features=evidence.shared_features,
        relationship_strength=evidence.relationship_strength,
        joint_cue_fit=evidence.joint_cue_fit,
    )
