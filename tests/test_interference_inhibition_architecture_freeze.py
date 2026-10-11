"""Architecture-freeze invariants for the 0.17 interference and inhibition line."""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_017_3_inhibition import (
    _T as INHIBITION_T,
)
from test_017_3_inhibition import (
    _TENANT as INHIBITION_TENANT,
)
from test_017_3_inhibition import (
    _burst_memory,
    _seed_burst,
    _seed_deployment_pair,
    _selected,
)
from test_017_6_transient_reachability import (
    _QUERY,
    _T,
    _TENANT,
    _episodes,
    _fact,
    _inspect_episodes,
    _memory,
    _seed_deployment_episodes,
    _stores,
)
from test_inhibition import (
    _episode,
    _induce,
    _lineage_snapshot,
    _result,
    _support,
    _with_snapshots,
)

from cogkura.models import (
    ActivationConfig,
    CompetitionConfig,
    CompetitionDirection,
    InhibitionConfig,
    MemoryIdentity,
    MemoryKind,
    RetrievalInterferenceState,
    SemanticMemoryStatus,
)
from cogkura.observations.encoding_context import RetrievalContext
from cogkura.storage.in_memory_inhibition import InMemoryInhibitionStore


class _CountingInhibitionStore(InMemoryInhibitionStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def list_for_memories(self, **kwargs: object) -> dict:
        self.reads += 1
        return await super().list_for_memories(**kwargs)  # type: ignore[arg-type]


def test_documented_defaults_match_configs() -> None:
    competition = CompetitionConfig()
    inhibition = InhibitionConfig()
    assert competition.enabled is True
    assert competition.apply_interference is False
    assert competition.minimum_strength == 0.45
    assert competition.max_competitors_per_candidate == 8
    assert competition.same_slot_strength == 0.95
    assert competition.same_predicate_strength == 0.80
    assert competition.entity_overlap_weight == 0.20
    assert competition.feature_overlap_weight == 0.30
    assert competition.minimum_behavioral_strength == 0.60
    assert competition.minimum_behavioral_cue_fit == 0.40
    assert competition.proactive_weight == 0.20
    assert competition.retroactive_weight == 0.20
    assert inhibition.enabled is False
    assert inhibition.apply_to_recall is False
    assert inhibition.minimum_induction_pressure == 0.20
    assert inhibition.max_traces_per_use == 32
    assert inhibition.inhibition_weight == 0.25
    assert inhibition.recovery_half_life_seconds == 604_800.0
    assert inhibition.minimum_remaining_strength == 0.01
    assert inhibition.max_traces_per_memory == 64


@pytest.mark.asyncio
async def test_diagnostic_competition_does_not_authorize_suppression() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=True,
            minimum_strength=0.30,
            same_slot_strength=0.55,
            minimum_behavioral_strength=0.70,
        ),
        inhibition_config=InhibitionConfig(enabled=True, apply_to_recall=False),
    )
    await _seed_deployment_episodes(memory)
    query = "payments-api deployment_system"
    inspection = await memory.inspect_recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    assert inspection.interference is not None
    assert inspection.interference.diagnostic_relationship_count > 0
    assert inspection.interference.scope_eligible_relationship_count == 0
    recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    assert all(result.components.interference == 0.0 for result in recalled)
    if recalled:
        await memory.record_access(recalled[:1], tenant_id=_TENANT, referenced_at=_T)
    episodes = await memory.list_episodes(tenant_id=_TENANT)
    for episode in episodes:
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=MemoryKind.EPISODE,
            memory_key=episode.memory_key,
        )
        assert traces == ()


@pytest.mark.asyncio
async def test_proactive_and_retroactive_directions_coexist() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    newer = _inspect_episodes(inspection, "GitHub Actions")[0]
    older = _inspect_episodes(inspection, "Jenkins")[0]
    assert newer.competition is not None and newer.competition.interference is not None
    assert older.competition is not None and older.competition.interference is not None
    assert newer.competition.interference.proactive_penalty < 0.0
    assert older.competition.interference.retroactive_penalty < 0.0
    proactive = [
        item
        for item in newer.competition.interference.contributions
        if item.direction is CompetitionDirection.PROACTIVE
    ]
    retroactive = [
        item
        for item in older.competition.interference.contributions
        if item.direction is CompetitionDirection.RETROACTIVE
    ]
    assert proactive and all(item.pressure > 0.0 for item in proactive)
    assert retroactive and all(item.pressure > 0.0 for item in retroactive)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert _episodes(recalled, "GitHub Actions")[0].components.interference < 0.0
    assert _episodes(recalled, "Jenkins")[0].components.interference < 0.0


@pytest.mark.asyncio
async def test_co_temporal_and_isolation_controls_have_no_penalty() -> None:
    co_temporal = _memory()
    await _seed_deployment_episodes(co_temporal, same_time=True)
    inspection = await co_temporal.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    episode = _inspect_episodes(inspection, "GitHub Actions")[0]
    assert episode.competition is not None
    co_temporal_pairs = [
        evidence
        for evidence in episode.competition.competitors
        if evidence.behavioral_eligibility is not None
        and evidence.behavioral_eligibility.scope_eligible
        and evidence.direction is CompetitionDirection.CO_TEMPORAL
    ]
    assert co_temporal_pairs
    assert all(not evidence.behavioral_eligibility.eligible for evidence in co_temporal_pairs)  # type: ignore[union-attr]
    recalled = await co_temporal.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert all(result.components.interference == 0.0 for result in recalled)

    isolated = _memory()
    await isolated.observe(
        _fact(
            source_record_id="deploy",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
            content="payments-api deployments use GitHub Actions.",
        )
    )
    await isolated.observe(
        _fact(
            source_record_id="db",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=20),
            content="payments-api production database is PostgreSQL.",
        )
    )
    await isolated.observe(
        _fact(
            source_record_id="identity",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="identity-service",
            observed_at=_T - timedelta(days=30),
            content="identity-service deployments use Jenkins.",
        )
    )
    await isolated.process(tenant_id=_TENANT, as_of=_T)
    for query in (_QUERY, "What is the production database?", "How is identity-service deployed?"):
        results = await isolated.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
        assert all(result.components.interference == 0.0 for result in results)


@pytest.mark.asyncio
async def test_same_lineage_and_unrelated_access_do_not_suppress() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    await memory.observe(
        _fact(
            source_record_id="db",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="billing-api",
            observed_at=_T - timedelta(days=2),
            content="billing-api production database is PostgreSQL.",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    database = await memory.recall(
        "What database does billing-api use?",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    if database:
        for index in range(3):
            await memory.record_access(
                database[:1],
                tenant_id=_TENANT,
                referenced_at=_T,
                request_id=f"freeze-boost-{index}",
            )
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    for candidate in (*inspection.returned, *inspection.rejected):
        if candidate.competition is None or candidate.competition.interference is None:
            continue
        supported = set()
        if candidate.diagnostics is not None:
            supported = {
                item.semantic_memory_key for item in candidate.diagnostics.support_provenance
            }
        for contribution in candidate.competition.interference.contributions:
            assert contribution.competitor_identity.memory_key not in supported
            assert "billing-api" not in contribution.competitor_identity.memory_key
        assert candidate.competition.interference.total_penalty <= 0.0
        assert candidate.components.interference <= 0.0


@pytest.mark.asyncio
async def test_presentation_apis_write_no_traces_and_explicit_use_can() -> None:
    memory = _memory(inhibition_config=InhibitionConfig(enabled=True, apply_to_recall=False))
    await _seed_deployment_episodes(memory)
    query = "payments-api deployment_system"
    await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    await memory.inspect_recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    await memory.assess_memory(query, tenant_id=_TENANT, as_of=_T)
    await memory.prepare_context(query, tenant_id=_TENANT, as_of=_T, prompt_budget_tokens=512)
    semantics = await memory.list_semantic_memories(tenant_id=_TENANT)
    for item in semantics:
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=item.memory_key,
        )
        assert traces == ()
    recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    selected = next(result for result in recalled if "github_actions" in result.memory.statement)
    await memory.record_access(
        [selected], tenant_id=_TENANT, referenced_at=_T, request_id="freeze-use"
    )
    await memory.record_access(
        [selected], tenant_id=_TENANT, referenced_at=_T, request_id="freeze-use"
    )
    competitor = next(result for result in recalled if "jenkins" in result.memory.statement)
    traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=competitor.memory_kind,
        memory_key=competitor.memory.memory_key,
    )
    assert len(traces) == 1


@pytest.mark.asyncio
async def test_burst_limit_does_not_redefine_consumption() -> None:
    from cogkura.storage.in_memory_activation import InMemoryActivationStore

    store = InMemoryActivationStore()
    memory = _burst_memory(store)
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=INHIBITION_TENANT,
        as_of=INHIBITION_T,
        limit=10,
    )
    selected_a = _selected(results, "github_actions")
    selected_b = _selected(results, "jenkins")
    identity_a = MemoryIdentity(
        memory_kind=selected_a.memory_kind,
        memory_key=selected_a.memory.memory_key,
    )
    await _seed_burst(store, identity_a)
    await memory.record_access(
        [selected_a, selected_b],
        tenant_id=INHIBITION_TENANT,
        referenced_at=INHIBITION_T,
    )
    for selected in (selected_a, selected_b):
        traces = await memory.list_inhibition_traces(
            tenant_id=INHIBITION_TENANT,
            memory_kind=selected.memory_kind,
            memory_key=selected.memory.memory_key,
        )
        assert traces == ()


def test_logical_co_consumption_and_third_competitor() -> None:
    redis = _with_snapshots(
        _result("redis", ()),
        (
            _lineage_snapshot(
                competitor_key="memcached-episode",
                competitor_kind=MemoryKind.EPISODE,
                competitor_lineage="semantic:memcached",
                candidate_lineage="semantic:redis",
            ),
            _lineage_snapshot(
                competitor_key="dragonfly",
                competitor_lineage="semantic:dragonfly",
                candidate_lineage="semantic:redis",
            ),
        ),
    )
    memcached = _with_snapshots(
        _result("memcached", ()),
        (
            _lineage_snapshot(
                competitor_key="redis",
                competitor_lineage="semantic:redis",
                candidate_lineage="semantic:memcached",
            ),
        ),
    )
    both = _induce([redis, memcached])
    reversed_order = _induce([memcached, redis])
    assert {trace.inhibited_identity.memory_key for trace in both} == {"dragonfly"}
    assert {
        (trace.selected_identity.memory_key, trace.inhibited_identity.memory_key) for trace in both
    } == {
        (trace.selected_identity.memory_key, trace.inhibited_identity.memory_key)
        for trace in reversed_order
    }
    only_redis = _induce(
        [
            _with_snapshots(
                _result("redis", ()),
                (
                    _lineage_snapshot(
                        competitor_key="memcached",
                        competitor_lineage="semantic:memcached",
                        candidate_lineage="semantic:redis",
                    ),
                ),
            )
        ]
    )
    assert [trace.inhibited_identity.memory_key for trace in only_redis] == ["memcached"]
    episode = _with_snapshots(
        _episode("old", (_support("memcached", SemanticMemoryStatus.SUPERSEDED),)),
        (
            _lineage_snapshot(
                competitor_key="memcached",
                competitor_lineage="semantic:memcached",
            ),
        ),
    )
    assert [trace.inhibited_identity.memory_key for trace in _induce([episode])] == ["memcached"]


@pytest.mark.asyncio
async def test_persistent_scope_recovery_repetition_and_reselection() -> None:
    from cogkura.models import BehavioralStructuralAnchor, InhibitionScopeSignature, InhibitoryTrace

    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=False),
        inhibition_config=InhibitionConfig(
            enabled=False,
            apply_to_recall=True,
            inhibition_weight=1.0,
        ),
    )
    await _seed_deployment_episodes(memory)
    query = "payments-api deployment_system"
    baseline = await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        result
        for result in baseline
        if result.memory_kind is MemoryKind.SEMANTIC and "jenkins" in result.memory.statement
    )
    github = next(
        result
        for result in baseline
        if result.memory_kind is MemoryKind.SEMANTIC and "github_actions" in result.memory.statement
    )
    scope = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
        subject_id="payments-api",
        predicate="deployment_system",
        semantic_slot_key=jenkins.memory.slot_key,
    )

    def _trace(trace_id: str, induced_at) -> InhibitoryTrace:
        return InhibitoryTrace(
            id=trace_id,
            tenant_id=_TENANT,
            inhibited_identity=MemoryIdentity(
                memory_kind=MemoryKind.SEMANTIC,
                memory_key=jenkins.memory.memory_key,
            ),
            selected_identity=MemoryIdentity(
                memory_kind=MemoryKind.SEMANTIC,
                memory_key=github.memory.memory_key,
            ),
            direction=CompetitionDirection.PROACTIVE,
            scope=scope,
            competition_strength=0.8,
            competitor_accessibility=1.0,
            induction_pressure=0.8,
            retrieval_evaluated_at=induced_at,
            induced_at=induced_at,
        )

    await store.append_traces([_trace("00000000-0000-0000-0000-000000000171", _T)])
    inhibited = await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    affected = next(
        result for result in inhibited if result.memory.memory_key == jenkins.memory.memory_key
    )
    assert affected.components.inhibition < 0.0
    other = await memory.recall(
        "What is the production database?",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    assert all(result.components.inhibition == 0.0 for result in other)
    later = await memory.recall(
        query,
        tenant_id=_TENANT,
        as_of=_T + timedelta(seconds=604_800),
        limit=10,
    )
    recovered = next(
        result for result in later if result.memory.memory_key == jenkins.memory.memory_key
    )
    assert abs(recovered.components.inhibition) < abs(affected.components.inhibition)
    hidden = await memory.recall(
        query,
        tenant_id=_TENANT,
        as_of=_T - timedelta(days=1),
        limit=10,
    )
    before_trace = next(
        result for result in hidden if result.memory.memory_key == jenkins.memory.memory_key
    )
    assert before_trace.components.inhibition == 0.0
    historical_facts = await memory.recall(
        query,
        tenant_id=_TENANT,
        as_of=_T,
        valid_at=_T - timedelta(days=30),
        limit=10,
    )
    assert all("GitHub Actions" not in result.memory.statement for result in historical_facts)
    await store.append_traces(
        [_trace("00000000-0000-0000-0000-000000000172", _T + timedelta(seconds=1))]
    )
    repeated = await memory.recall(
        query, tenant_id=_TENANT, as_of=_T + timedelta(seconds=1), limit=10
    )
    stronger = next(
        result for result in repeated if result.memory.memory_key == jenkins.memory.memory_key
    )
    assert abs(stronger.components.inhibition) > abs(affected.components.inhibition)
    assert len(store._traces) == 2
    reselected = await memory.recall(
        query, tenant_id=_TENANT, as_of=_T + timedelta(days=30), limit=10
    )
    assert any(result.memory.memory_key == jenkins.memory.memory_key for result in reselected)


@pytest.mark.asyncio
async def test_persistent_inhibition_precedes_transient_interference() -> None:
    from cogkura.models import BehavioralStructuralAnchor, InhibitionScopeSignature, InhibitoryTrace

    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
    )
    await _seed_deployment_episodes(memory)
    query = "payments-api deployment_system"
    before = await memory.inspect_recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        result
        for result in recalled
        if result.memory_kind is MemoryKind.SEMANTIC and "jenkins" in result.memory.statement
    )
    await store.append_traces(
        [
            InhibitoryTrace(
                id="00000000-0000-0000-0000-000000000173",
                tenant_id=_TENANT,
                inhibited_identity=MemoryIdentity(
                    memory_kind=MemoryKind.SEMANTIC,
                    memory_key=jenkins.memory.memory_key,
                ),
                selected_identity=MemoryIdentity(
                    memory_kind=MemoryKind.SEMANTIC,
                    memory_key="winner",
                ),
                direction=CompetitionDirection.PROACTIVE,
                scope=InhibitionScopeSignature(
                    structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
                    subject_id="payments-api",
                    predicate="deployment_system",
                    semantic_slot_key=jenkins.memory.slot_key,
                ),
                competition_strength=0.8,
                competitor_accessibility=1.0,
                induction_pressure=0.8,
                retrieval_evaluated_at=_T,
                induced_at=_T,
            )
        ]
    )
    after = await memory.inspect_recall(query, tenant_id=_TENANT, as_of=_T, limit=10)

    def _jenkins_pressure(inspection) -> float:
        candidate = next(
            item
            for item in (*inspection.returned, *inspection.rejected)
            if item.memory_kind is MemoryKind.SEMANTIC and "github_actions" in item.memory.statement
        )
        assert candidate.competition is not None and candidate.competition.interference is not None
        pressures = [
            item.pressure
            for item in candidate.competition.interference.contributions
            if item.competitor_identity.memory_key == jenkins.memory.memory_key
        ]
        assert pressures
        return max(pressures)

    assert _jenkins_pressure(after) < _jenkins_pressure(before)
    affected = next(
        item
        for item in (*after.returned, *after.rejected)
        if item.memory.memory_key == jenkins.memory.memory_key
    )
    assert affected.components.inhibition < 0.0
    assert affected.components.interference <= 0.0
    assert affected.components.inhibition != affected.components.interference


@pytest.mark.asyncio
async def test_clocks_reconsolidation_forgetting_and_reinforcement_stay_separate() -> None:
    stores = _stores()
    memory = _memory(
        **stores,
        inhibition_config=InhibitionConfig(enabled=True, apply_to_recall=False),
    )
    await memory.observe(
        _fact(
            source_record_id="mysql",
            predicate="production_database",
            object_value="mysql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=200),
            content="payments-api production database was MySQL.",
            cardinality="one",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T - timedelta(days=100))
    await memory.observe(
        _fact(
            source_record_id="postgres",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
            content="payments-api production database is PostgreSQL.",
            cardinality="one",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    current = await memory.recall(
        "What is the production database?",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    assert current
    assert all("MySQL" not in result.memory.statement for result in current)
    dynamics = stores["dynamics_store"]
    assert dynamics._dynamics == {}  # type: ignore[attr-defined]
    selected = current[0]
    references_before = list(stores["activation_store"]._references)  # type: ignore[attr-defined]
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T, request_id="use-db")
    assert len(stores["activation_store"]._references) == len(references_before) + 1  # type: ignore[attr-defined]
    assert dynamics._dynamics == {}  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_observability_is_read_only_and_shared() -> None:
    stores = _stores()
    memory = _memory(**stores)
    await _seed_deployment_episodes(memory)
    references = list(stores["activation_store"]._references)  # type: ignore[attr-defined]
    traces = list(stores["inhibition_store"]._traces)  # type: ignore[attr-defined]
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    context = await memory.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        as_of=_T,
        prompt_budget_tokens=512,
    )
    contextual = await memory.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        as_of=_T,
        prompt_budget_tokens=512,
        retrieval_context=RetrievalContext(domain="payments-api"),
    )
    assert inspection.interference == assessment.interference
    assert context.assessment.interference == assessment.interference
    assert inspection.interference is not None
    assert inspection.interference.state is RetrievalInterferenceState.TRANSIENT_INTERFERENCE
    assert contextual.assessment.interference is not None
    assert contextual.working_memory.items
    assert all(
        item.recall.memory_kind in {MemoryKind.EPISODE, MemoryKind.SEMANTIC}
        for item in contextual.working_memory.items
    )
    again = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    first = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert [(item.memory.memory_key, item.activation) for item in again] == [
        (item.memory.memory_key, item.activation) for item in first
    ]
    assert stores["activation_store"]._references == references  # type: ignore[attr-defined]
    assert stores["inhibition_store"]._traces == traces  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_threshold_suppression_stays_observable() -> None:
    stores = _stores()
    probe = _memory(
        **stores,
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
    )
    await _seed_deployment_episodes(probe)
    recalled = await probe.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    target = _episodes(recalled, "GitHub Actions")[0]
    memory = _memory(
        **stores,
        activation_config=ActivationConfig(
            retrieval_threshold=target.activation - 0.05,
            enable_semantic_slot_admission=False,
            enable_entity_slot_admission=False,
        ),
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=True,
            proactive_weight=1.0,
            retroactive_weight=1.0,
        ),
    )
    recalled_on = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert all(item.memory.memory_key != target.memory.memory_key for item in recalled_on)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert assessment.interference is not None
    assert assessment.interference.threshold_suppressed_by_interference_count > 0


@pytest.mark.asyncio
async def test_disabled_behaviour_matches_and_skips_inhibition_reads() -> None:
    off_a = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=False))
    off_b = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=False))
    for memory in (off_a, off_b):
        await _seed_deployment_episodes(memory)
    left = await off_a.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    right = await off_b.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert [(item.memory.statement, item.activation) for item in left] == [
        (item.memory.statement, item.activation) for item in right
    ]
    store = _CountingInhibitionStore()
    quiet = _memory(
        inhibition_store=store,
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
        inhibition_config=InhibitionConfig(enabled=True, apply_to_recall=False),
    )
    await _seed_deployment_episodes(quiet)
    await quiet.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    await quiet.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    await quiet.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert store.reads == 0
