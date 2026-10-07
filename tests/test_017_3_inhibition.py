"""Behaviour tests for 0.17.3 retrieval-induced inhibitory traces."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    ActivationConfig,
    ActivationReferenceKind,
    CompetitionConfig,
    CompetitionDirection,
    InhibitionConfig,
    MemoryIdentity,
    MemoryKind,
    MemoryReference,
    WorkingMemoryConfig,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_dynamics import InMemoryMemoryDynamicsStore

_TENANT = "inhibition-tenant"
_SUBJECT = "operator-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _memory(**kwargs: object) -> Memory:
    defaults: dict[str, object] = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
        "competition_config": CompetitionConfig(enabled=True, apply_interference=False),
        "inhibition_config": InhibitionConfig(enabled=True),
    }
    defaults.update(kwargs)
    return Memory(**defaults)  # type: ignore[arg-type]


def _fact(
    *,
    source_record_id: str,
    predicate: str,
    object_value: str,
    subject_entity_id: str,
    observed_at: datetime,
) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        source_namespace="facts",
        source_record_id=source_record_id,
        source_type="fact",
        content=f"{subject_entity_id} {predicate} {object_value}",
        observed_at=observed_at,
        metadata={
            "entity_ids": [subject_entity_id],
            "semantic_facts": [
                {
                    "predicate": predicate,
                    "object_value": object_value,
                    "subject_entity_id": subject_entity_id,
                    "cardinality": "many",
                    "polarity": "affirm",
                    "qualifiers": {},
                }
            ],
        },
    )


async def _seed_deployment_pair(memory: Memory, *, same_time: bool = False) -> None:
    await memory.observe(
        _fact(
            source_record_id="jenkins",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=0 if same_time else 120),
        )
    )
    await memory.observe(
        _fact(
            source_record_id="gha",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=0 if same_time else 10),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)


def _selected(results: list, token: str):
    matched = [result for result in results if token in result.memory.statement]
    assert matched
    return matched[0]


@pytest.mark.asyncio
async def test_presentation_apis_do_not_record_traces() -> None:
    memory = _memory()
    await _seed_deployment_pair(memory)
    query = "payments-api deployment_system"
    await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    await memory.inspect_recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    await memory.select_working_memory(
        query,
        tenant_id=_TENANT,
        as_of=_T,
        prompt_budget_tokens=512,
    )
    await memory.prepare_context(query, tenant_id=_TENANT, as_of=_T, prompt_budget_tokens=512)
    await memory.assess_memory(query, tenant_id=_TENANT, as_of=_T)
    semantics = await memory.list_semantic_memories(tenant_id=_TENANT)
    for item in semantics:
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=item.memory_key,
        )
        assert traces == ()


@pytest.mark.asyncio
async def test_record_access_creates_selective_trace() -> None:
    memory = _memory()
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = _selected(results, "github_actions")
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T)
    jenkins = next(item for item in results if "jenkins" in item.memory.statement)
    traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=jenkins.memory_kind,
        memory_key=jenkins.memory.memory_key,
    )
    assert len(traces) == 1
    assert traces[0].selected_identity.memory_key == selected.memory.memory_key
    assert traces[0].inhibited_identity.memory_key == jenkins.memory.memory_key
    assert traces[0].direction is CompetitionDirection.PROACTIVE
    assert "payments-api deployment_system" not in str(traces[0].scope.to_canonical_dict())


@pytest.mark.asyncio
async def test_co_consumed_competitors_do_not_inhibit() -> None:
    memory = _memory()
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    await memory.record_access(results, tenant_id=_TENANT, referenced_at=_T)
    for result in results:
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=result.memory_kind,
            memory_key=result.memory.memory_key,
        )
        assert traces == ()


@pytest.mark.asyncio
async def test_co_temporal_scope_eligible_pair_can_trace() -> None:
    memory = _memory()
    await _seed_deployment_pair(memory, same_time=True)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = results[0]
    other = results[1]
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T)
    traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=other.memory_kind,
        memory_key=other.memory.memory_key,
    )
    assert len(traces) == 1
    assert traces[0].direction is CompetitionDirection.CO_TEMPORAL


@pytest.mark.asyncio
async def test_cross_service_competitor_is_not_traced() -> None:
    memory = _memory()
    await memory.observe(
        _fact(
            source_record_id="payments",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
        )
    )
    await memory.observe(
        _fact(
            source_record_id="identity",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="identity-service",
            observed_at=_T - timedelta(days=20),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = _selected(results, "payments-api")
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T)
    identity = [result for result in results if "identity-service" in result.memory.statement]
    for result in identity:
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=result.memory_kind,
            memory_key=result.memory.memory_key,
        )
        assert traces == ()


@pytest.mark.asyncio
async def test_low_induction_pressure_skips_trace() -> None:
    memory = _memory(
        inhibition_config=InhibitionConfig(enabled=True, minimum_induction_pressure=0.99)
    )
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    await memory.record_access([results[0]], tenant_id=_TENANT, referenced_at=_T)
    for result in results:
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=result.memory_kind,
            memory_key=result.memory.memory_key,
        )
        assert traces == ()


@pytest.mark.asyncio
async def test_request_id_is_idempotent_and_missing_id_repeats() -> None:
    memory = _memory()
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = results[0]
    other = results[1]
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T, request_id="use-1")
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T, request_id="use-1")
    once = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=other.memory_kind,
        memory_key=other.memory.memory_key,
    )
    assert len(once) == 1
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T)
    await memory.record_access(
        [selected],
        tenant_id=_TENANT,
        referenced_at=_T + timedelta(seconds=1),
    )
    repeated = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=other.memory_kind,
        memory_key=other.memory.memory_key,
    )
    assert len(repeated) == 3


@pytest.mark.asyncio
async def test_recorded_inhibition_does_not_change_recall() -> None:
    inhibited = _memory()
    control = _memory(inhibition_config=InhibitionConfig(enabled=False))
    await _seed_deployment_pair(inhibited)
    await _seed_deployment_pair(control)
    query = "payments-api deployment_system"
    before = await inhibited.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    selected = before[0]
    await inhibited.record_access([selected], tenant_id=_TENANT, referenced_at=_T)
    control_before = await control.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    await control.record_access([control_before[0]], tenant_id=_TENANT, referenced_at=_T)
    after = await inhibited.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    control_after = await control.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    assert [(item.memory.memory_key, item.activation, item.score) for item in after] == [
        (item.memory.memory_key, item.activation, item.score) for item in control_after
    ]


@pytest.mark.asyncio
async def test_inhibition_does_not_change_forgetting_or_semantic_status() -> None:
    dynamics = InMemoryMemoryDynamicsStore()
    memory = _memory(dynamics_store=dynamics)
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = results[0]
    other = results[1]
    before = await memory.list_semantic_memories(tenant_id=_TENANT)
    statuses = {item.memory_key: item.status for item in before}
    other_identity = MemoryIdentity(
        memory_kind=other.memory_kind,
        memory_key=other.memory.memory_key,
    )
    before_dynamics = await dynamics.get_many(tenant_id=_TENANT, identities=[other_identity])
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T)
    after = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert {item.memory_key: item.status for item in after} == statuses
    after_dynamics = await dynamics.get_many(tenant_id=_TENANT, identities=[other_identity])
    assert before_dynamics.get(other_identity) == after_dynamics.get(other_identity)


@pytest.mark.asyncio
async def test_historical_recall_cannot_inhibit_future_competitor() -> None:
    memory = _memory()
    historical = datetime(2025, 6, 1, tzinfo=UTC)
    await memory.observe(
        _fact(
            source_record_id="mysql",
            predicate="production_database",
            object_value="mysql",
            subject_entity_id="platform",
            observed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=historical)
    results = await memory.recall(
        "platform production_database",
        tenant_id=_TENANT,
        valid_at=historical,
        as_of=historical,
        limit=10,
    )
    await memory.observe(
        _fact(
            source_record_id="pg",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="platform",
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    await memory.record_access(results, tenant_id=_TENANT, referenced_at=_T)
    later = await memory.list_semantic_memories(tenant_id=_TENANT)
    future = [item for item in later if "postgresql" in item.statement]
    for item in future:
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=item.memory_key,
        )
        assert traces == ()


async def _reference_count(store: InMemoryActivationStore, identity: MemoryIdentity) -> int:
    traces = await store.list_reference_traces(
        tenant_id=_TENANT,
        identities=[identity],
        before_or_at=_T + timedelta(days=1),
    )
    return len(traces.get(identity, ()))


async def _seed_burst(store: InMemoryActivationStore, identity: MemoryIdentity) -> None:
    await store.append_references(
        [
            MemoryReference(
                tenant_id=_TENANT,
                memory_kind=identity.memory_kind,
                memory_key=identity.memory_key,
                reference_kind=ActivationReferenceKind.RETRIEVED,
                referenced_at=_T - timedelta(minutes=5),
            )
        ]
    )


def _burst_memory(store: InMemoryActivationStore) -> Memory:
    return _memory(
        activation_store=store,
        activation_config=ActivationConfig(
            access_burst_limit=1,
            access_burst_window_seconds=3600.0,
        ),
    )


@pytest.mark.asyncio
async def test_burst_throttled_co_consumption_does_not_inhibit() -> None:
    store = InMemoryActivationStore()
    memory = _burst_memory(store)
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected_a = _selected(results, "github_actions")
    selected_b = _selected(results, "jenkins")
    identity_a = MemoryIdentity(
        memory_kind=selected_a.memory_kind,
        memory_key=selected_a.memory.memory_key,
    )
    identity_b = MemoryIdentity(
        memory_kind=selected_b.memory_kind,
        memory_key=selected_b.memory.memory_key,
    )
    await _seed_burst(store, identity_a)
    await memory.record_access([selected_a, selected_b], tenant_id=_TENANT, referenced_at=_T)
    assert await _reference_count(store, identity_a) == 1
    assert await _reference_count(store, identity_b) == 1
    for identity in (identity_a, identity_b):
        traces = await memory.list_inhibition_traces(
            tenant_id=_TENANT,
            memory_kind=identity.memory_kind,
            memory_key=identity.memory_key,
        )
        assert traces == ()


@pytest.mark.asyncio
async def test_burst_throttled_selection_still_inhibits_competitor() -> None:
    store = InMemoryActivationStore()
    memory = _burst_memory(store)
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = _selected(results, "github_actions")
    competitor = _selected(results, "jenkins")
    selected_identity = MemoryIdentity(
        memory_kind=selected.memory_kind,
        memory_key=selected.memory.memory_key,
    )
    await _seed_burst(store, selected_identity)
    await memory.record_access([selected], tenant_id=_TENANT, referenced_at=_T)
    assert await _reference_count(store, selected_identity) == 1
    traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=competitor.memory_kind,
        memory_key=competitor.memory.memory_key,
    )
    assert len(traces) == 1
    assert traces[0].selected_identity == selected_identity


@pytest.mark.asyncio
async def test_retry_after_reference_write_still_records_inhibition() -> None:
    store = InMemoryActivationStore()
    memory = _burst_memory(store)
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = _selected(results, "github_actions")
    competitor = _selected(results, "jenkins")
    selected_identity = MemoryIdentity(
        memory_kind=selected.memory_kind,
        memory_key=selected.memory.memory_key,
    )
    await store.append_references(
        [
            MemoryReference(
                tenant_id=_TENANT,
                memory_kind=selected_identity.memory_kind,
                memory_key=selected_identity.memory_key,
                reference_kind=ActivationReferenceKind.RETRIEVED,
                referenced_at=_T - timedelta(minutes=1),
                request_id="request-123",
            )
        ]
    )
    await memory.record_access(
        [selected],
        tenant_id=_TENANT,
        referenced_at=_T,
        request_id="request-123",
    )
    await memory.record_access(
        [selected],
        tenant_id=_TENANT,
        referenced_at=_T,
        request_id="request-123",
    )
    assert await _reference_count(store, selected_identity) == 1
    traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=competitor.memory_kind,
        memory_key=competitor.memory.memory_key,
    )
    assert len(traces) == 1
    assert traces[0].request_id == "request-123"


@pytest.mark.asyncio
async def test_min_score_excludes_result_from_consumed_set() -> None:
    memory = _memory()
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    selected = _selected(results, "github_actions")
    competitor = replace(_selected(results, "jenkins"), score=0.05)
    assert selected.score >= 0.5
    await memory.record_access(
        [selected, competitor],
        tenant_id=_TENANT,
        referenced_at=_T,
        min_score=0.5,
    )
    traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=competitor.memory_kind,
        memory_key=competitor.memory.memory_key,
    )
    assert len(traces) == 1
    assert traces[0].selected_identity.memory_key == selected.memory.memory_key
    selected_traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=selected.memory_kind,
        memory_key=selected.memory.memory_key,
    )
    assert selected_traces == ()


@pytest.mark.asyncio
async def test_retroactive_trace_direction() -> None:
    memory = _memory()
    await _seed_deployment_pair(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    older = _selected(results, "jenkins")
    newer = _selected(results, "github_actions")
    await memory.record_access([older], tenant_id=_TENANT, referenced_at=_T)
    traces = await memory.list_inhibition_traces(
        tenant_id=_TENANT,
        memory_kind=newer.memory_kind,
        memory_key=newer.memory.memory_key,
    )
    assert len(traces) == 1
    assert traces[0].direction is CompetitionDirection.RETROACTIVE
    assert traces[0].selected_identity.memory_key == older.memory.memory_key


@pytest.mark.asyncio
async def test_record_context_use_creates_one_trace() -> None:
    memory = _memory(working_memory_config=WorkingMemoryConfig(max_items=1, enable_chunking=False))
    await _seed_deployment_pair(memory)
    context = await memory.prepare_context(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        as_of=_T,
        prompt_budget_tokens=512,
    )
    consumed_keys = {result.memory.memory_key for result in context.recall_results}
    assert len(consumed_keys) == 1
    await memory.record_context_use(context, referenced_at=_T)
    semantics = await memory.list_semantic_memories(tenant_id=_TENANT)
    unconsumed = [item for item in semantics if item.memory_key not in consumed_keys]
    traces = []
    for item in unconsumed:
        traces.extend(
            await memory.list_inhibition_traces(
                tenant_id=_TENANT,
                memory_kind=MemoryKind.SEMANTIC,
                memory_key=item.memory_key,
            )
        )
    assert len(traces) == 1
    assert traces[0].selected_identity.memory_key in consumed_keys


@pytest.mark.asyncio
async def test_inhibition_requires_competition() -> None:
    with pytest.raises(Exception, match="CompetitionConfig.enabled"):
        _memory(
            competition_config=CompetitionConfig(enabled=False),
            inhibition_config=InhibitionConfig(enabled=True),
        )


@pytest.mark.asyncio
async def test_scope_key_is_order_independent() -> None:
    from cogkura.models import BehavioralStructuralAnchor, InhibitionScopeSignature

    left = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.QUERY_SCOPE,
        entity_ids=("b", "a"),
        feature_ids=("deploy", "system"),
    )
    right = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.QUERY_SCOPE,
        entity_ids=("a", "b"),
        feature_ids=("system", "deploy"),
    )
    assert left.scope_key == right.scope_key
    other = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor.QUERY_SCOPE,
        entity_ids=("a",),
        feature_ids=("database",),
    )
    assert left.scope_key != other.scope_key
