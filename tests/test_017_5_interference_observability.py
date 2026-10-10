"""Behaviour tests for 0.17.5 interference observability."""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_017_3_inhibition import _T, _TENANT, _fact, _seed_deployment_pair
from test_017_4_inhibition_application import _trace

from cogkura import Memory
from cogkura.algorithms.competition import CompetitionEvaluation
from cogkura.algorithms.interference_observability import summarize_retrieval_interference
from cogkura.algorithms.metamemory import DeterministicMemoryMonitor
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.exceptions import ValidationError
from cogkura.models import (
    ActivationConfig,
    BehavioralStructuralAnchor,
    CompetitionConfig,
    CompetitionRunDiagnostics,
    InhibitionConfig,
    InhibitionScopeSignature,
    MemoryAssessmentFlag,
    MemoryKind,
    MetamemoryConfig,
    RetrievalCue,
    RetrievalInterferenceObservability,
    RetrievalInterferenceState,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_dynamics import InMemoryMemoryDynamicsStore
from cogkura.storage.in_memory_entity_relationship import InMemoryEntityRelationshipStore
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore
from cogkura.storage.in_memory_inhibition import InMemoryInhibitionStore
from cogkura.storage.in_memory_learning import InMemoryLearningStore
from cogkura.storage.in_memory_observation import InMemoryCheckpointStore, InMemoryObservationStore
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore

_QUERY = "payments-api deployment_system"


class _CountingInhibitionStore(InMemoryInhibitionStore):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    async def list_for_memories(self, **kwargs: object) -> dict:
        self.batch_calls += 1
        return await super().list_for_memories(**kwargs)  # type: ignore[arg-type]


def _stores(**overrides: object) -> dict[str, object]:
    stores: dict[str, object] = {
        "observation_store": InMemoryObservationStore(),
        "checkpoint_store": InMemoryCheckpointStore(),
        "episode_store": InMemoryEpisodeStore(),
        "semantic_store": InMemorySemanticMemoryStore(),
        "activation_store": InMemoryActivationStore(),
        "dynamics_store": InMemoryMemoryDynamicsStore(),
        "learning_store": InMemoryLearningStore(),
        "entity_relationship_store": InMemoryEntityRelationshipStore(),
        "inhibition_store": InMemoryInhibitionStore(),
    }
    stores.update(overrides)
    return stores


def _memory(**kwargs: object) -> Memory:
    defaults: dict[str, object] = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
        "competition_config": CompetitionConfig(enabled=True, apply_interference=False),
        "inhibition_config": InhibitionConfig(enabled=False, apply_to_recall=False),
    }
    defaults.update(kwargs)
    return Memory(**defaults)  # type: ignore[arg-type]


def _scope(slot_key: str) -> InhibitionScopeSignature:
    return InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
        subject_id="payments-api",
        predicate="deployment_system",
        semantic_slot_key=slot_key,
    )


def _zero_run() -> CompetitionRunDiagnostics:
    return CompetitionRunDiagnostics(
        candidate_count=0,
        potential_competitor_pairs=0,
        evaluated_competitor_pairs=0,
        accepted_competition_pairs=0,
        maximum_competitors_for_candidate=0,
    )


def _observability(
    *,
    state: RetrievalInterferenceState = RetrievalInterferenceState.CLEAR,
    scope_eligible_relationship_count: int = 0,
    transient_interference_evaluated: bool = False,
    max_transient_pressure: float | None = None,
    threshold_suppressed_by_interference_count: int = 0,
    persistently_inhibited_candidate_count: int = 0,
    effective_inhibition_trace_count: int = 0,
) -> RetrievalInterferenceObservability:
    return RetrievalInterferenceObservability(
        state=state,
        competition_evaluated=True,
        transient_interference_evaluated=transient_interference_evaluated,
        persistent_inhibition_evaluated=True,
        diagnostic_relationship_count=scope_eligible_relationship_count,
        scope_eligible_relationship_count=scope_eligible_relationship_count,
        transient_eligible_relationship_count=0,
        competing_candidate_count=0,
        transiently_affected_candidate_count=0,
        persistently_inhibited_candidate_count=persistently_inhibited_candidate_count,
        threshold_suppressed_by_interference_count=threshold_suppressed_by_interference_count,
        threshold_suppressed_by_inhibition_count=0,
        effective_inhibition_trace_count=effective_inhibition_trace_count,
        inactive_matched_inhibition_trace_count=0,
        max_competition_strength=0.0,
        max_transient_pressure=max_transient_pressure,
        max_transient_penalty_magnitude=0.0 if transient_interference_evaluated else None,
        max_persistent_inhibition_pressure=0.0,
        max_persistent_inhibition_penalty_magnitude=0.0,
        strongest_competition=None,
    )


def test_disabled_mechanisms_are_not_evaluated_zero() -> None:
    disabled = summarize_retrieval_interference(
        [],
        None,
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(apply_to_recall=False),
    )
    assert disabled.state is RetrievalInterferenceState.NOT_EVALUATED
    assert disabled.competition_evaluated is False
    assert disabled.transient_interference_evaluated is False
    assert disabled.persistent_inhibition_evaluated is False
    assert disabled.max_competition_strength is None
    assert disabled.max_transient_pressure is None
    assert disabled.max_transient_penalty_magnitude is None
    assert disabled.max_persistent_inhibition_pressure is None
    assert disabled.max_persistent_inhibition_penalty_magnitude is None

    evaluated = summarize_retrieval_interference(
        [],
        CompetitionEvaluation(by_identity={}, run=_zero_run()),
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
        inhibition_config=InhibitionConfig(apply_to_recall=True),
    )
    assert evaluated.state is RetrievalInterferenceState.CLEAR
    assert evaluated.max_competition_strength == 0.0
    assert evaluated.max_transient_pressure == 0.0
    assert evaluated.max_transient_penalty_magnitude == 0.0
    assert evaluated.max_persistent_inhibition_pressure == 0.0
    assert evaluated.max_persistent_inhibition_penalty_magnitude == 0.0


def test_high_interference_threshold_is_validated() -> None:
    MetamemoryConfig(high_interference_pressure_threshold=0.0)
    MetamemoryConfig(high_interference_pressure_threshold=1.0)
    with pytest.raises(ValidationError):
        MetamemoryConfig(high_interference_pressure_threshold=1.01)


def test_empty_pool_can_report_interference_flags_without_quality_flags() -> None:
    assessment = DeterministicMemoryMonitor().assess(
        candidates=[],
        query=RetrievalCue(text=_QUERY),
        goal=RetrievalCue(text=_QUERY),
        tenant_id=_TENANT,
        subject_id=None,
        as_of=_T,
        valid_at=None,
        config=MetamemoryConfig(),
        activation_config=ActivationConfig(),
        interference=_observability(
            state=RetrievalInterferenceState.PERSISTENT_INHIBITION,
            persistently_inhibited_candidate_count=1,
            effective_inhibition_trace_count=1,
        ),
    )
    assert assessment.flags == (
        MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,
        MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE,
    )
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE not in assessment.flags
    assert MemoryAssessmentFlag.COMPETING_MEMORIES not in assessment.flags


def test_interference_flags_do_not_imply_semantic_conflict() -> None:
    assessment = DeterministicMemoryMonitor().assess(
        candidates=[],
        query=RetrievalCue(text=_QUERY),
        goal=RetrievalCue(text=_QUERY),
        tenant_id=_TENANT,
        subject_id=None,
        as_of=_T,
        valid_at=None,
        config=MetamemoryConfig(high_interference_pressure_threshold=0.50),
        activation_config=ActivationConfig(),
        interference=_observability(
            state=RetrievalInterferenceState.COMBINED,
            scope_eligible_relationship_count=2,
            transient_interference_evaluated=True,
            max_transient_pressure=0.80,
            persistently_inhibited_candidate_count=1,
            effective_inhibition_trace_count=2,
        ),
    )
    assert MemoryAssessmentFlag.COMPETING_MEMORIES in assessment.flags
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE in assessment.flags
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE in assessment.flags
    assert MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY not in assessment.flags
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE not in assessment.flags


def test_low_transient_pressure_is_not_high_interference() -> None:
    assessment = DeterministicMemoryMonitor().assess(
        candidates=[],
        query=RetrievalCue(text=_QUERY),
        goal=RetrievalCue(text=_QUERY),
        tenant_id=_TENANT,
        subject_id=None,
        as_of=_T,
        valid_at=None,
        config=MetamemoryConfig(high_interference_pressure_threshold=0.90),
        activation_config=ActivationConfig(),
        interference=_observability(
            state=RetrievalInterferenceState.TRANSIENT_INTERFERENCE,
            transient_interference_evaluated=True,
            max_transient_pressure=0.20,
        ),
    )
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE not in assessment.flags


def test_threshold_suppression_sets_high_interference_below_pressure_threshold() -> None:
    assessment = DeterministicMemoryMonitor().assess(
        candidates=[],
        query=RetrievalCue(text=_QUERY),
        goal=RetrievalCue(text=_QUERY),
        tenant_id=_TENANT,
        subject_id=None,
        as_of=_T,
        valid_at=None,
        config=MetamemoryConfig(high_interference_pressure_threshold=0.90),
        activation_config=ActivationConfig(),
        interference=_observability(
            state=RetrievalInterferenceState.TRANSIENT_INTERFERENCE,
            transient_interference_evaluated=True,
            max_transient_pressure=0.20,
            threshold_suppressed_by_interference_count=1,
        ),
    )
    assert assessment.flags == (
        MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,
        MemoryAssessmentFlag.HIGH_INTERFERENCE,
    )


@pytest.mark.asyncio
async def test_competition_disabled_is_not_evaluated() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=False),
    )
    await _seed_deployment_pair(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert inspection.interference is not None
    assert inspection.interference == assessment.interference
    assert inspection.interference.state is RetrievalInterferenceState.NOT_EVALUATED
    assert inspection.interference.max_persistent_inhibition_pressure is None
    assert MemoryAssessmentFlag.COMPETING_MEMORIES not in assessment.flags
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE not in assessment.flags
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE not in assessment.flags


@pytest.mark.asyncio
async def test_single_memory_evaluated_zero_is_clear() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
    )
    await memory.observe(
        _fact(
            source_record_id="only",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert inspection.interference is not None
    assert inspection.interference.state is RetrievalInterferenceState.CLEAR
    assert inspection.interference.competition_evaluated is True
    assert inspection.interference.transient_interference_evaluated is True
    assert inspection.interference.persistent_inhibition_evaluated is False
    assert inspection.interference.diagnostic_relationship_count == 0
    assert inspection.interference.max_competition_strength == 0.0
    assert inspection.interference.max_transient_pressure == 0.0
    assert inspection.interference.max_persistent_inhibition_pressure is None


@pytest.mark.asyncio
async def test_diagnostic_only_competition_does_not_warn() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=False,
            minimum_strength=0.30,
            same_slot_strength=0.55,
            minimum_behavioral_strength=0.70,
        ),
    )
    await _seed_deployment_pair(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert inspection.interference is not None
    assert inspection.interference.diagnostic_relationship_count > 0
    assert inspection.interference.scope_eligible_relationship_count == 0
    assert inspection.interference.state is RetrievalInterferenceState.CLEAR
    assert inspection.interference.max_competition_strength == 0.0
    assert MemoryAssessmentFlag.COMPETING_MEMORIES not in assessment.flags
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE not in assessment.flags


@pytest.mark.asyncio
async def test_scope_eligible_without_transient_behaviour_is_competing() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
    )
    await _seed_deployment_pair(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    interference = inspection.interference
    assert interference is not None
    assert interference.scope_eligible_relationship_count > 0
    assert interference.transient_interference_evaluated is False
    assert interference.max_transient_pressure is None
    assert interference.state is RetrievalInterferenceState.COMPETING
    assert MemoryAssessmentFlag.COMPETING_MEMORIES in assessment.flags
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE not in assessment.flags
    assert MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY not in assessment.flags
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE not in assessment.flags


@pytest.mark.asyncio
async def test_co_temporal_scope_eligible_competition_has_no_high_interference() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
    )
    await _seed_deployment_pair(memory, same_time=True)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    interference = inspection.interference
    assert interference is not None
    assert interference.scope_eligible_relationship_count > 0
    assert interference.transient_eligible_relationship_count == 0
    assert interference.transiently_affected_candidate_count == 0
    assert interference.max_transient_penalty_magnitude == 0.0
    assert interference.state is RetrievalInterferenceState.COMPETING
    assert MemoryAssessmentFlag.COMPETING_MEMORIES in assessment.flags
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE not in assessment.flags


@pytest.mark.asyncio
async def test_active_transient_interference_is_observable() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
        metamemory_config=MetamemoryConfig(high_interference_pressure_threshold=0.40),
    )
    await _seed_deployment_pair(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    interference = inspection.interference
    assert interference is not None
    assert interference.transiently_affected_candidate_count > 0
    assert interference.max_transient_penalty_magnitude is not None
    assert interference.max_transient_penalty_magnitude > 0.0
    assert interference.max_transient_pressure is not None
    assert interference.max_transient_pressure > 0.40
    assert interference.state is RetrievalInterferenceState.TRANSIENT_INTERFERENCE
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE in assessment.flags
    assert MemoryAssessmentFlag.COMPETING_MEMORIES in assessment.flags
    assert interference == assessment.interference


@pytest.mark.asyncio
async def test_low_transient_pressure_stays_below_high_interference() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
        metamemory_config=MetamemoryConfig(high_interference_pressure_threshold=0.99),
    )
    await _seed_deployment_pair(memory)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    interference = assessment.interference
    assert interference is not None
    assert interference.state is RetrievalInterferenceState.TRANSIENT_INTERFERENCE
    assert interference.max_transient_pressure is not None
    assert interference.max_transient_pressure < 0.99
    assert interference.max_transient_penalty_magnitude is not None
    assert interference.max_transient_penalty_magnitude > 0.0
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE not in assessment.flags


@pytest.mark.asyncio
async def test_transient_threshold_suppression_is_high_interference() -> None:
    stores = _stores()
    probe = _memory(
        **stores,
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
    )
    await _seed_deployment_pair(probe)
    recalled = await probe.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    weaker = min(recalled, key=lambda item: item.activation)
    memory = _memory(
        **stores,
        activation_config=ActivationConfig(
            retrieval_threshold=weaker.activation - 0.05,
            enable_semantic_slot_admission=False,
            enable_entity_slot_admission=False,
        ),
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=True,
            proactive_weight=1.0,
            retroactive_weight=1.0,
        ),
        metamemory_config=MetamemoryConfig(high_interference_pressure_threshold=0.99),
    )
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    interference = inspection.interference
    assert interference is not None
    assert interference.threshold_suppressed_by_interference_count > 0
    assert interference.max_transient_pressure is not None
    assert interference.max_transient_pressure < 0.99
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE in assessment.flags


@pytest.mark.asyncio
async def test_persistent_inhibition_without_current_competition() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
    )
    await _seed_deployment_pair(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000171",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    interference = inspection.interference
    assert interference is not None
    assert interference.competition_evaluated is False
    assert interference.persistent_inhibition_evaluated is True
    assert interference.persistently_inhibited_candidate_count > 0
    assert interference.effective_inhibition_trace_count > 0
    assert interference.max_persistent_inhibition_pressure is not None
    assert interference.max_persistent_inhibition_pressure > 0.0
    assert interference.max_competition_strength is None
    assert interference.state is RetrievalInterferenceState.PERSISTENT_INHIBITION
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE in assessment.flags
    assert MemoryAssessmentFlag.COMPETING_MEMORIES not in assessment.flags
    assert interference == assessment.interference


@pytest.mark.asyncio
async def test_recovered_traces_are_inactive_and_not_active_inhibition() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
    )
    await _seed_deployment_pair(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000172",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    later = _T + timedelta(seconds=6.5 * 604_800)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=later, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=later)
    interference = inspection.interference
    assert interference is not None
    assert interference.inactive_matched_inhibition_trace_count > 0
    assert interference.effective_inhibition_trace_count == 0
    assert interference.persistently_inhibited_candidate_count == 0
    assert interference.state is RetrievalInterferenceState.CLEAR
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE not in assessment.flags


@pytest.mark.asyncio
async def test_apply_to_recall_false_does_not_read_traces_for_observability() -> None:
    store = _CountingInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
        inhibition_config=InhibitionConfig(enabled=True, apply_to_recall=False),
    )
    await _seed_deployment_pair(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000173",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    store.batch_calls = 0
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert store.batch_calls == 0
    assert inspection.interference is not None
    assert inspection.interference.persistent_inhibition_evaluated is False
    assert inspection.interference.max_persistent_inhibition_pressure is None
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE not in assessment.flags


@pytest.mark.asyncio
async def test_suppressed_candidate_remains_in_query_observability() -> None:
    stores = _stores()
    probe = _memory(
        **stores,
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=False),
    )
    await _seed_deployment_pair(probe)
    recalled = await probe.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    store = stores["inhibition_store"]
    assert isinstance(store, InMemoryInhibitionStore)
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000174",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
                pressure=0.8,
            )
        ]
    )
    memory = _memory(
        **stores,
        activation_config=ActivationConfig(
            retrieval_threshold=jenkins.activation - 0.05,
            enable_semantic_slot_admission=False,
            enable_entity_slot_admission=False,
        ),
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(
            enabled=False,
            apply_to_recall=True,
            inhibition_weight=1.0,
        ),
    )
    results = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert all(item.memory.memory_key != jenkins.memory.memory_key for item in results)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    interference = inspection.interference
    assert interference is not None
    assert interference.persistently_inhibited_candidate_count > 0
    assert interference.threshold_suppressed_by_inhibition_count > 0
    assert jenkins.memory.memory_key not in {item.memory.memory_key for item in inspection.returned}


@pytest.mark.asyncio
async def test_empty_recall_caused_by_inhibition_keeps_the_flag() -> None:
    stores = _stores()
    probe = _memory(
        **stores,
        competition_config=CompetitionConfig(enabled=False),
    )
    await probe.observe(
        _fact(
            source_record_id="only-jenkins",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
        )
    )
    await probe.process(tenant_id=_TENANT, as_of=_T)
    recalled = await probe.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert len(recalled) == 1
    only = recalled[0]
    store = stores["inhibition_store"]
    assert isinstance(store, InMemoryInhibitionStore)
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000175",
                inhibited_key=only.memory.memory_key,
                selected_key="winner",
                scope=_scope(only.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    memory = _memory(
        **stores,
        activation_config=ActivationConfig(
            retrieval_threshold=only.activation - 0.05,
            enable_semantic_slot_admission=False,
            enable_entity_slot_admission=False,
        ),
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(
            enabled=False,
            apply_to_recall=True,
            inhibition_weight=1.0,
        ),
    )
    results = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert results == []
    assert assessment.retrieved_count == 0
    assert MemoryAssessmentFlag.NO_RETRIEVED_MEMORY in assessment.flags
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE in assessment.flags
    assert assessment.interference is not None
    assert assessment.interference.threshold_suppressed_by_inhibition_count > 0


@pytest.mark.asyncio
async def test_combined_transient_and_persistent_state() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
        metamemory_config=MetamemoryConfig(high_interference_pressure_threshold=0.40),
    )
    await _seed_deployment_pair(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000176",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    interference = inspection.interference
    assert interference is not None
    assert interference.state is RetrievalInterferenceState.COMBINED
    assert MemoryAssessmentFlag.COMPETING_MEMORIES in assessment.flags
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE in assessment.flags
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE in assessment.flags
    assert MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY not in assessment.flags


@pytest.mark.asyncio
async def test_strongest_scope_eligible_pair_is_deterministic() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=True,
            same_slot_strength=0.70,
            entity_overlap_weight=0.99,
            feature_overlap_weight=0.0,
            minimum_strength=0.40,
            minimum_behavioral_strength=0.50,
        ),
    )
    await _seed_deployment_pair(memory)
    await memory.observe(
        _fact(
            source_record_id="postgres",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=40),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    first = await memory.inspect_recall("payments-api", tenant_id=_TENANT, as_of=_T, limit=10)
    second = await memory.inspect_recall("payments-api", tenant_id=_TENANT, as_of=_T, limit=1)
    assert first.interference == second.interference
    interference = first.interference
    assert interference is not None
    assert interference.strongest_competition is not None

    eligible: list[tuple[float, tuple[str, str], tuple[str, str]]] = []
    ineligible_strengths: list[float] = []
    for candidate in (*first.returned, *first.rejected):
        if candidate.competition is None:
            continue
        candidate_key = (candidate.memory_kind.value, candidate.memory.memory_key)
        for evidence in candidate.competition.competitors:
            eligibility = evidence.behavioral_eligibility
            if eligibility is None or not eligibility.scope_eligible:
                if eligibility is not None:
                    ineligible_strengths.append(eligibility.competition_strength)
                continue
            competitor_key = (
                evidence.competitor_identity.memory_kind.value,
                evidence.competitor_identity.memory_key,
            )
            eligible.append((eligibility.competition_strength, candidate_key, competitor_key))
    assert eligible
    expected = min(eligible, key=lambda item: (-item[0], item[1], item[2]))
    selected = interference.strongest_competition
    assert selected.competition_strength + 1e-9 >= expected[0]
    selected_pair = (
        (selected.candidate_identity.memory_kind.value, selected.candidate_identity.memory_key),
        (selected.competitor_identity.memory_kind.value, selected.competitor_identity.memory_key),
    )
    attached_at_strength = {
        (item[1], item[2])
        for item in eligible
        if abs(item[0] - selected.competition_strength) <= 1e-9
    }
    if selected_pair in attached_at_strength:
        assert selected_pair == (expected[1], expected[2])
    if ineligible_strengths and max(ineligible_strengths) > selected.competition_strength:
        assert any(item[0] == pytest.approx(selected.competition_strength) for item in eligible)


@pytest.mark.asyncio
async def test_activation_margin_uses_pre_transient_rank_activation() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
        inhibition_config=InhibitionConfig(
            enabled=False, apply_to_recall=True, inhibition_weight=1.0
        ),
    )
    await _seed_deployment_pair(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000177",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    interference = inspection.interference
    assert interference is not None
    selected = interference.strongest_competition
    assert selected is not None
    by_key = {
        (candidate.memory_kind, candidate.memory.memory_key): candidate
        for candidate in (*inspection.returned, *inspection.rejected)
    }
    candidate = by_key[
        (selected.candidate_identity.memory_kind, selected.candidate_identity.memory_key)
    ]
    competitor = by_key[
        (selected.competitor_identity.memory_kind, selected.competitor_identity.memory_key)
    ]
    assert candidate.diagnostics is not None
    assert competitor.diagnostics is not None
    candidate_rank = candidate.diagnostics.rank_activation_before_interference
    competitor_rank = competitor.diagnostics.rank_activation_before_interference
    assert candidate_rank is not None
    assert competitor_rank is not None
    assert selected.activation_margin == pytest.approx(abs(candidate_rank - competitor_rank))
    if candidate.components.inhibition < 0.0 or competitor.components.inhibition < 0.0:
        before_inhibition = candidate.diagnostics.rank_activation_before_inhibition
        competitor_before = competitor.diagnostics.rank_activation_before_inhibition
        assert before_inhibition is not None
        assert competitor_before is not None
        assert abs(selected.activation_margin - abs(before_inhibition - competitor_before)) > 1e-6


@pytest.mark.asyncio
async def test_inspect_and_assess_share_interference_across_limits() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
    )
    await _seed_deployment_pair(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=1)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert inspection.interference == assessment.interference
    assert len(inspection.returned) <= 1
    assert assessment.retrieved_count >= len(inspection.returned)


@pytest.mark.asyncio
async def test_historical_as_of_hides_later_traces_and_valid_at_does_not() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
    )
    await _seed_deployment_pair(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000178",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    historical = await memory.assess_memory(
        _QUERY,
        tenant_id=_TENANT,
        as_of=_T - timedelta(days=1),
    )
    assert historical.interference is not None
    assert historical.interference.effective_inhibition_trace_count == 0
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE not in historical.flags

    current = await memory.assess_memory(
        _QUERY,
        tenant_id=_TENANT,
        as_of=_T,
        valid_at=_T - timedelta(days=30),
    )
    assert current.interference is not None
    assert current.interference.effective_inhibition_trace_count > 0
    assert current.interference.state is RetrievalInterferenceState.PERSISTENT_INHIBITION
    assert MemoryAssessmentFlag.RETRIEVAL_INHIBITION_ACTIVE in current.flags


@pytest.mark.asyncio
async def test_recovery_lowers_observed_pressure_without_writing_traces() -> None:
    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
    )
    await _seed_deployment_pair(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        item
        for item in recalled
        if item.memory_kind is MemoryKind.SEMANTIC and "jenkins" in item.memory.statement
    )
    await store.append_traces(
        [
            _trace(
                trace_id="00000000-0000-0000-0000-000000000179",
                inhibited_key=jenkins.memory.memory_key,
                selected_key="winner",
                scope=_scope(jenkins.memory.slot_key),
                induced_at=_T,
            )
        ]
    )
    before_traces = list(store._traces)
    early = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    later = await memory.assess_memory(
        _QUERY,
        tenant_id=_TENANT,
        as_of=_T + timedelta(seconds=604_800),
    )
    assert early.interference is not None
    assert later.interference is not None
    assert early.interference.max_persistent_inhibition_pressure is not None
    assert later.interference.max_persistent_inhibition_pressure is not None
    assert (
        later.interference.max_persistent_inhibition_pressure
        < early.interference.max_persistent_inhibition_pressure
    )
    assert store._traces == before_traces


@pytest.mark.asyncio
async def test_inspect_and_assess_do_not_write() -> None:
    stores = _stores()
    memory = _memory(
        **stores,
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
        inhibition_config=InhibitionConfig(enabled=True, apply_to_recall=True),
    )
    await _seed_deployment_pair(memory)
    activation = stores["activation_store"]
    dynamics = stores["dynamics_store"]
    learning = stores["learning_store"]
    inhibition = stores["inhibition_store"]
    assert isinstance(activation, InMemoryActivationStore)
    assert isinstance(dynamics, InMemoryMemoryDynamicsStore)
    assert isinstance(learning, InMemoryLearningStore)
    assert isinstance(inhibition, InMemoryInhibitionStore)
    references = list(activation._references)
    dynamic_state = dict(dynamics._dynamics)
    learning_state = dict(learning._states)
    traces = list(inhibition._traces)
    before = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    after = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert activation._references == references
    assert dynamics._dynamics == dynamic_state
    assert learning._states == learning_state
    assert inhibition._traces == traces
    assert [
        (item.memory_kind, item.memory.memory_key, item.activation, item.score) for item in before
    ] == [(item.memory_kind, item.memory.memory_key, item.activation, item.score) for item in after]


@pytest.mark.asyncio
async def test_prepare_context_assessment_matches_assess_memory() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
    )
    await _seed_deployment_pair(memory)
    context = await memory.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        as_of=_T,
        prompt_budget_tokens=512,
    )
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert context.assessment.interference == assessment.interference
