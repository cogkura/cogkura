"""Unit tests for inhibitory trace induction and the in-memory store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura.algorithms.inhibition import build_inhibitory_traces
from cogkura.models import (
    ActivationComponents,
    BehavioralStructuralAnchor,
    CompetitionDirection,
    InhibitionConfig,
    InhibitionScopeSignature,
    InhibitoryTrace,
    MemoryIdentity,
    MemoryKind,
    RecallResult,
    RetrievalCompetitionSnapshot,
    RetrievalDiagnostics,
    SemanticCardinality,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredSemanticMemory,
)
from cogkura.storage.in_memory_inhibition import InMemoryInhibitionStore

_T = datetime(2026, 8, 1, tzinfo=UTC)


def _semantic(memory_key: str) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id=memory_key,
        tenant_id="tenant",
        subject_id="operator",
        memory_key=memory_key,
        slot_key="slot",
        revision_key="rev",
        revision_number=1,
        statement=memory_key,
        subject_entity_id="payments-api",
        predicate="deployment_system",
        object_value=memory_key,
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


def _result(memory_key: str, snapshots: tuple[RetrievalCompetitionSnapshot, ...]) -> RecallResult:
    diagnostics = RetrievalDiagnostics(
        rank_activation=0.0,
        accessibility_partial=0.0,
        ranking_partial=0.0,
        conjunction=0.0,
        text_coverage=0.5,
        text_cue_fit=0.5,
        temporal_mode="current",
        inhibition_candidates=snapshots,
    )
    return RecallResult(
        memory_kind=MemoryKind.SEMANTIC,
        memory=_semantic(memory_key),
        activation=1.0,
        score=0.8,
        latency_seconds=0.1,
        components=ActivationComponents(0.0, 0.0, 0.0, 0.0, 1.0),
        reason="test",
        diagnostics=diagnostics,
    )


def _snapshot(
    competitor_key: str, strength: float, accessibility: float
) -> RetrievalCompetitionSnapshot:
    return RetrievalCompetitionSnapshot(
        competitor_identity=MemoryIdentity(
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=competitor_key,
        ),
        direction=CompetitionDirection.PROACTIVE,
        competition_strength=strength,
        competitor_accessibility=accessibility,
        scope_eligible=True,
        scope=InhibitionScopeSignature(
            structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
            subject_id="payments-api",
            predicate="deployment_system",
            semantic_slot_key=f"slot-{competitor_key}",
        ),
        retrieval_evaluated_at=_T,
    )


def test_trace_selection_is_bounded_and_deterministic() -> None:
    snapshots = tuple(
        _snapshot(f"c-{index}", strength=0.9 - (index * 0.01), accessibility=0.8)
        for index in range(5)
    )
    selected = _result("selected", snapshots)
    traces = build_inhibitory_traces(
        [selected],
        tenant_id="tenant",
        config=InhibitionConfig(enabled=True, max_traces_per_use=2, minimum_induction_pressure=0.2),
        induced_at=_T,
        request_id=None,
    )
    assert len(traces) == 2
    assert traces[0].induction_pressure >= traces[1].induction_pressure
    assert traces[0].inhibited_identity.memory_key == "c-0"


@pytest.mark.asyncio
async def test_in_memory_request_id_is_idempotent() -> None:
    store = InMemoryInhibitionStore()
    selected = _result("selected", (_snapshot("loser", 0.8, 0.8),))
    traces = build_inhibitory_traces(
        [selected],
        tenant_id="tenant",
        config=InhibitionConfig(enabled=True),
        induced_at=_T,
        request_id="req-1",
    )
    await store.append_traces(traces)
    await store.append_traces(traces)
    listed = await store.list_for_memory(
        tenant_id="tenant",
        identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="loser"),
    )
    assert len(listed) == 1


def _scope(anchor: BehavioralStructuralAnchor, **kwargs: object) -> InhibitionScopeSignature:
    return InhibitionScopeSignature(structural_anchor=anchor, **kwargs)  # type: ignore[arg-type]


def _manual_trace(
    *,
    trace_id: str,
    induced_at: datetime,
    scope: InhibitionScopeSignature,
    request_id: str | None = None,
    inhibited_key: str = "loser",
) -> InhibitoryTrace:
    return InhibitoryTrace(
        id=trace_id,
        tenant_id="tenant",
        inhibited_identity=MemoryIdentity(
            memory_kind=MemoryKind.SEMANTIC, memory_key=inhibited_key
        ),
        selected_identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="winner"),
        direction=CompetitionDirection.RETROACTIVE,
        scope=scope,
        competition_strength=0.8,
        competitor_accessibility=0.5,
        induction_pressure=0.4,
        retrieval_evaluated_at=_T,
        induced_at=induced_at,
        request_id=request_id,
    )


def _listing_fixtures() -> tuple[InhibitoryTrace, ...]:
    return (
        _manual_trace(
            trace_id="00000000-0000-0000-0000-000000000003",
            induced_at=_T + timedelta(hours=2),
            scope=_scope(
                BehavioralStructuralAnchor.SEMANTIC_SLOT,
                subject_id="payments-api",
                predicate="deployment_system",
                semantic_slot_key="slot-b",
            ),
        ),
        _manual_trace(
            trace_id="00000000-0000-0000-0000-000000000001",
            induced_at=_T,
            scope=_scope(
                BehavioralStructuralAnchor.QUERY_SCOPE,
                entity_ids=("payments-api",),
                feature_ids=("deploy",),
            ),
        ),
        _manual_trace(
            trace_id="00000000-0000-0000-0000-000000000002",
            induced_at=_T,
            scope=_scope(
                BehavioralStructuralAnchor.SUBJECT_PREDICATE,
                subject_id="payments-api",
                predicate="deployment_system",
                entity_ids=("payments-api",),
            ),
        ),
    )


def _loser() -> MemoryIdentity:
    return MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="loser")


@pytest.mark.asyncio
async def test_in_memory_repeated_use_without_request_id() -> None:
    store = InMemoryInhibitionStore()
    first = _manual_trace(
        trace_id="00000000-0000-0000-0000-00000000000a",
        induced_at=_T,
        scope=_scope(BehavioralStructuralAnchor.SEMANTIC_SLOT, semantic_slot_key="slot"),
        request_id=None,
    )
    second = _manual_trace(
        trace_id="00000000-0000-0000-0000-00000000000b",
        induced_at=_T + timedelta(seconds=1),
        scope=first.scope,
        request_id=None,
    )
    await store.append_traces([first])
    await store.append_traces([second])
    listed = await store.list_for_memory(tenant_id="tenant", identity=_loser())
    assert [trace.id for trace in listed] == [first.id, second.id]


@pytest.mark.asyncio
async def test_in_memory_as_of_and_order() -> None:
    store = InMemoryInhibitionStore()
    fixtures = _listing_fixtures()
    await store.append_traces(fixtures)
    listed = await store.list_for_memory(tenant_id="tenant", identity=_loser())
    expected = sorted(
        fixtures, key=lambda trace: (trace.induced_at, trace.scope.scope_key, trace.id)
    )
    assert [trace.id for trace in listed] == [trace.id for trace in expected]
    cutoff = _T + timedelta(hours=1)
    visible = await store.list_for_memory(tenant_id="tenant", identity=_loser(), as_of=cutoff)
    assert [trace.id for trace in visible] == [
        trace.id for trace in expected if trace.induced_at <= cutoff
    ]


@pytest.mark.asyncio
async def test_list_for_memories_bounds_and_isolates_tenants() -> None:
    store = InMemoryInhibitionStore()
    older = _manual_trace(
        trace_id="00000000-0000-0000-0000-000000000011",
        induced_at=_T,
        scope=_scope(BehavioralStructuralAnchor.SEMANTIC_SLOT, semantic_slot_key="slot"),
    )
    newer = _manual_trace(
        trace_id="00000000-0000-0000-0000-000000000012",
        induced_at=_T + timedelta(hours=3),
        scope=_scope(BehavioralStructuralAnchor.SEMANTIC_SLOT, semantic_slot_key="slot"),
    )
    other_tenant = InhibitoryTrace(
        id="00000000-0000-0000-0000-000000000013",
        tenant_id="other",
        inhibited_identity=_loser(),
        selected_identity=MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="ep"),
        direction=CompetitionDirection.PROACTIVE,
        scope=older.scope,
        competition_strength=0.5,
        competitor_accessibility=0.5,
        induction_pressure=0.5,
        retrieval_evaluated_at=_T,
        induced_at=_T + timedelta(hours=4),
    )
    episode = _manual_trace(
        trace_id="00000000-0000-0000-0000-000000000014",
        induced_at=_T,
        scope=older.scope,
        inhibited_key="episode-key",
    )
    episode = InhibitoryTrace(
        id=episode.id,
        tenant_id=episode.tenant_id,
        inhibited_identity=MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-key"),
        selected_identity=episode.selected_identity,
        direction=episode.direction,
        scope=episode.scope,
        competition_strength=episode.competition_strength,
        competitor_accessibility=episode.competitor_accessibility,
        induction_pressure=episode.induction_pressure,
        retrieval_evaluated_at=episode.retrieval_evaluated_at,
        induced_at=episode.induced_at,
    )
    await store.append_traces([older, newer, other_tenant, episode])
    empty = await store.list_for_memories(
        tenant_id="tenant",
        identities=[],
        before_or_at=_T + timedelta(days=1),
    )
    assert empty == {}
    listed = await store.list_for_memories(
        tenant_id="tenant",
        identities=[
            _loser(),
            MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-key"),
        ],
        after=_T + timedelta(hours=1),
        before_or_at=_T + timedelta(days=1),
        limit_per_memory=1,
    )
    assert [trace.id for trace in listed[_loser()]] == [newer.id]
    assert MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-key") not in listed
