"""Behaviour tests for 0.17.4 persistent inhibition application."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from test_017_3_inhibition import _fact, _seed_deployment_pair

from cogkura import Memory
from cogkura.algorithms.behavioral_competition import build_behavioral_query_scope
from cogkura.algorithms.inhibition_application import (
    DeterministicInhibitionScopeMatcher,
    InhibitionCandidateProfile,
    noisy_or,
    remaining_strength,
)
from cogkura.algorithms.retrieval_features import canonical_content_features
from cogkura.models import (
    ActivationComponents,
    BehavioralStructuralAnchor,
    CompetitionConfig,
    CompetitionDirection,
    InhibitionConfig,
    InhibitionScopeMatchReason,
    InhibitionScopeSignature,
    InhibitoryTrace,
    MemoryIdentity,
    MemoryKind,
    RetrievalCue,
    RetrievalDiagnostics,
)
from cogkura.storage.in_memory_inhibition import InMemoryInhibitionStore

_TENANT = "apply-tenant"
_SUBJECT = "operator-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


class _CountingInhibitionStore(InMemoryInhibitionStore):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    async def list_for_memories(self, **kwargs: object) -> dict:
        self.batch_calls += 1
        return await super().list_for_memories(**kwargs)  # type: ignore[arg-type]


def _memory(store: InMemoryInhibitionStore, **kwargs: object) -> Memory:
    from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator

    defaults: dict[str, object] = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
        "competition_config": CompetitionConfig(enabled=True, apply_interference=False),
        "inhibition_store": store,
        "inhibition_config": InhibitionConfig(enabled=True, apply_to_recall=True),
    }
    defaults.update(kwargs)
    return Memory(**defaults)  # type: ignore[arg-type]


def _identity(memory_key: str) -> MemoryIdentity:
    return MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key=memory_key)


def _trace(
    *,
    trace_id: str,
    inhibited_key: str,
    selected_key: str,
    scope: InhibitionScopeSignature,
    induced_at: datetime,
    pressure: float = 0.8,
    direction: CompetitionDirection = CompetitionDirection.PROACTIVE,
) -> InhibitoryTrace:
    return InhibitoryTrace(
        id=trace_id,
        tenant_id="inhibition-tenant",
        inhibited_identity=_identity(inhibited_key),
        selected_identity=_identity(selected_key),
        direction=direction,
        scope=scope,
        competition_strength=pressure,
        competitor_accessibility=1.0,
        induction_pressure=pressure,
        retrieval_evaluated_at=induced_at,
        induced_at=induced_at,
    )


def test_recovery_half_life_and_cutoff() -> None:
    induced = _T
    half = 604_800.0
    fresh, _ = remaining_strength(
        0.8, induced_at=induced, evaluated_at=induced, half_life_seconds=half
    )
    assert fresh == pytest.approx(0.8)
    one, _ = remaining_strength(
        0.8,
        induced_at=induced,
        evaluated_at=induced + timedelta(seconds=half),
        half_life_seconds=half,
    )
    assert one == pytest.approx(0.4)
    two, _ = remaining_strength(
        0.8,
        induced_at=induced,
        evaluated_at=induced + timedelta(seconds=2 * half),
        half_life_seconds=half,
    )
    assert two == pytest.approx(0.2)
    three, _ = remaining_strength(
        0.8,
        induced_at=induced,
        evaluated_at=induced + timedelta(seconds=3 * half),
        half_life_seconds=half,
    )
    assert three == pytest.approx(0.1)
    tiny, _ = remaining_strength(
        0.8,
        induced_at=induced,
        evaluated_at=induced + timedelta(seconds=10 * half),
        half_life_seconds=half,
    )
    assert tiny < 0.01


def test_noisy_or_is_bounded() -> None:
    pressure = noisy_or([0.8, 0.8, 0.8, 0.8])
    assert pressure <= 1.0
    assert pressure > 0.8
    assert pressure < noisy_or([0.8, 0.8, 0.8, 0.8, 0.8]) or pressure == pytest.approx(1.0)


def test_query_scope_and_subject_predicate_matching() -> None:
    matcher = DeterministicInhibitionScopeMatcher()
    candidate = InhibitionCandidateProfile(
        identity=_identity("jenkins"),
        subject_id=_SUBJECT,
        subject_entity_id="customer-42",
        semantic_slot_key="slot-jacket",
        predicate="preferred_jacket",
        entity_ids=("customer-42",),
        retrieval_features=("customer", "42", "preferred", "jacket"),
    )
    jacket = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.SUBJECT_PREDICATE,
        subject_id="customer-42",
        predicate="preferred_jacket",
    )
    home = build_behavioral_query_scope(RetrievalCue(text="Where does customer-42 live?"))
    prefer = build_behavioral_query_scope(RetrievalCue(text="What jacket does customer-42 prefer?"))
    assert not matcher.matches(trace_scope=jacket, candidate=candidate, query_scope=home).matched
    assert matcher.matches(trace_scope=jacket, candidate=candidate, query_scope=prefer).matched
    deploy_scope = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.QUERY_SCOPE,
        entity_ids=("payments-api",),
        feature_ids=tuple(sorted(canonical_content_features("deploy"))),
    )
    payments = InhibitionCandidateProfile(
        identity=_identity("jenkins"),
        subject_id=_SUBJECT,
        subject_entity_id="payments-api",
        semantic_slot_key=None,
        predicate=None,
        entity_ids=("payments-api",),
        retrieval_features=("payments", "api", "deploy"),
    )
    deploy_query = build_behavioral_query_scope(RetrievalCue(text="How do we deploy payments-api?"))
    other_service = build_behavioral_query_scope(
        RetrievalCue(text="How is identity-service deployed?")
    )
    assert (
        matcher.matches(
            trace_scope=deploy_scope, candidate=payments, query_scope=deploy_query
        ).reason
        is InhibitionScopeMatchReason.QUERY_SCOPE_MATCH
    )
    assert (
        matcher.matches(
            trace_scope=deploy_scope, candidate=payments, query_scope=other_service
        ).reason
        is InhibitionScopeMatchReason.QUERY_ENTITY_MISMATCH
    )


@pytest.mark.asyncio
async def test_apply_to_recall_false_does_not_read_the_store() -> None:
    store = _CountingInhibitionStore()
    memory = _memory(
        store,
        inhibition_config=InhibitionConfig(enabled=True, apply_to_recall=False),
    )
    await _seed_deployment_pair(memory)
    await memory.recall("payments-api deployment_system", tenant_id="inhibition-tenant", as_of=_T)
    assert store.batch_calls == 0


@pytest.mark.asyncio
async def test_matching_trace_reduces_activation_and_recovers() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(store)
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T,
        limit=10,
    )
    jenkins = next(item for item in results if "jenkins" in item.memory.statement)
    selected = next(item for item in results if "github_actions" in item.memory.statement)
    scope = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
        subject_id="payments-api",
        predicate="deployment_system",
        semantic_slot_key=jenkins.memory.slot_key,
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-0000000000a1",
                inhibited_key=jenkins.memory.memory_key,
                selected_key=selected.memory.memory_key,
                scope=scope,
                induced_at=_T,
                pressure=0.8,
            )
        ]
    )
    inhibited = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T,
        limit=10,
    )
    after = next(item for item in inhibited if item.memory.memory_key == jenkins.memory.memory_key)
    assert after.activation == pytest.approx(jenkins.activation - 0.25 * 0.8)
    assert after.components.inhibition == pytest.approx(-0.2)
    later = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T + timedelta(seconds=604_800),
        limit=10,
    )
    recovered = next(item for item in later if item.memory.memory_key == jenkins.memory.memory_key)
    assert abs(recovered.components.inhibition) < abs(after.components.inhibition)


@pytest.mark.asyncio
async def test_unrelated_predicate_is_not_suppressed() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(store)
    await _seed_deployment_pair(memory)
    await memory.observe(
        _fact(
            source_record_id="db",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=3),
        )
    )
    await memory.process(tenant_id="inhibition-tenant", as_of=_T)
    database = await memory.recall(
        "payments-api production_database",
        tenant_id="inhibition-tenant",
        as_of=_T,
        limit=10,
    )
    target = next(item for item in database if "postgresql" in item.memory.statement)
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-0000000000b1",
                inhibited_key=target.memory.memory_key,
                selected_key="other",
                scope=InhibitionScopeSignature(
                    structural_anchor=BehavioralStructuralAnchor.SUBJECT_PREDICATE,
                    subject_id="payments-api",
                    predicate="deployment_system",
                ),
                induced_at=_T,
            )
        ]
    )
    again = await memory.recall(
        "payments-api production_database",
        tenant_id="inhibition-tenant",
        as_of=_T,
        limit=10,
    )
    seen = next(item for item in again if item.memory.memory_key == target.memory.memory_key)
    assert seen.components.inhibition == 0.0


@pytest.mark.asyncio
async def test_recording_off_still_applies_existing_traces() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(
        store,
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
    )
    await _seed_deployment_pair(memory)
    before = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T,
        limit=10,
    )
    jenkins = next(item for item in before if "jenkins" in item.memory.statement)
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-0000000000c1",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=InhibitionScopeSignature(
                    structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
                    subject_id="payments-api",
                    predicate="deployment_system",
                    semantic_slot_key=jenkins.memory.slot_key,
                ),
                induced_at=_T,
            )
        ]
    )
    await memory.record_access([before[0]], tenant_id="inhibition-tenant", referenced_at=_T)
    listed = await memory.list_inhibition_traces(
        tenant_id="inhibition-tenant",
        memory_kind=MemoryKind.SEMANTIC,
        memory_key=jenkins.memory.memory_key,
    )
    assert len(listed) == 1
    after = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T,
        limit=10,
    )
    updated = next(item for item in after if item.memory.memory_key == jenkins.memory.memory_key)
    assert updated.components.inhibition < 0.0


@pytest.mark.asyncio
async def test_future_trace_is_hidden_by_as_of_but_not_by_valid_at() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(store)
    await _seed_deployment_pair(memory)
    current = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T,
        limit=10,
    )
    jenkins = next(item for item in current if "jenkins" in item.memory.statement)
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-0000000000d1",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=InhibitionScopeSignature(
                    structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
                    subject_id="payments-api",
                    predicate="deployment_system",
                    semantic_slot_key=jenkins.memory.slot_key,
                ),
                induced_at=_T,
            )
        ]
    )
    historical = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T - timedelta(days=1),
        limit=10,
    )
    hidden = next(
        item for item in historical if item.memory.memory_key == jenkins.memory.memory_key
    )
    assert hidden.components.inhibition == 0.0
    replay = await memory.recall(
        "payments-api deployment_system",
        tenant_id="inhibition-tenant",
        as_of=_T,
        valid_at=_T - timedelta(days=30),
        limit=10,
    )
    visible = [item for item in replay if item.memory.memory_key == jenkins.memory.memory_key]
    if visible:
        assert visible[0].components.inhibition < 0.0


@pytest.mark.asyncio
async def test_presentation_does_not_write_when_application_is_on() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(store)
    await _seed_deployment_pair(memory)
    query = "payments-api deployment_system"
    await memory.recall(query, tenant_id="inhibition-tenant", as_of=_T)
    await memory.inspect_recall(query, tenant_id="inhibition-tenant", as_of=_T)
    await memory.select_working_memory(query, tenant_id="inhibition-tenant", as_of=_T)
    await memory.prepare_context(
        query, tenant_id="inhibition-tenant", as_of=_T, prompt_budget_tokens=256
    )
    await memory.assess_memory(query, tenant_id="inhibition-tenant", as_of=_T)
    assert store._traces == []


@pytest.mark.asyncio
async def test_threshold_crossing_is_attributed() -> None:
    from cogkura.algorithms.inhibition_application import evaluate_persistent_inhibition
    from cogkura.models import (
        RecallResult,
        SemanticCardinality,
        SemanticMemoryStatus,
        SemanticPolarity,
        StoredSemanticMemory,
    )

    memory = StoredSemanticMemory(
        id="sem",
        tenant_id="inhibition-tenant",
        subject_id=_SUBJECT,
        memory_key="jenkins",
        slot_key="slot-deploy",
        revision_key="rev",
        revision_number=1,
        statement="payments-api deployment_system jenkins",
        subject_entity_id="payments-api",
        predicate="deployment_system",
        object_value="jenkins",
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.MANY,
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
    result = RecallResult(
        memory_kind=MemoryKind.SEMANTIC,
        memory=memory,
        activation=-2.9,
        score=0.5,
        latency_seconds=1.0,
        components=ActivationComponents(-2.9, 0.0, 0.0, 0.0, -2.9),
        reason="test",
        diagnostics=RetrievalDiagnostics(
            rank_activation=-2.9,
            accessibility_partial=0.0,
            ranking_partial=0.0,
            conjunction=0.0,
            text_coverage=0.5,
            text_cue_fit=0.5,
            temporal_mode="current",
            semantic_slot_key="slot-deploy",
        ),
    )
    revised = evaluate_persistent_inhibition(
        result,
        [
            _trace(
                trace_id="00000000-0000-0000-0000-0000000000e1",
                inhibited_key="jenkins",
                selected_key="winner",
                scope=InhibitionScopeSignature(
                    structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
                    subject_id="payments-api",
                    predicate="deployment_system",
                    semantic_slot_key="slot-deploy",
                ),
                induced_at=_T,
                pressure=0.8,
            )
        ],
        config=InhibitionConfig(apply_to_recall=True, inhibition_weight=0.25),
        query_scope=build_behavioral_query_scope(
            RetrievalCue(text="payments-api deployment_system")
        ),
        matcher=DeterministicInhibitionScopeMatcher(),
        evaluated_at=_T,
        retrieval_threshold=-3.0,
        latency_factor=1.0,
        latency_exponent=1.0,
        rank_activation=-2.9,
    )
    assert revised.diagnostics is not None
    assert revised.diagnostics.crossed_activation_threshold_due_to_inhibition
    assert revised.activation < -3.0
