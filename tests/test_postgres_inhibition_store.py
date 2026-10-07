"""PostgreSQL inhibitory trace store tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura.migrations import apply_migrations
from cogkura.models import (
    BehavioralStructuralAnchor,
    CompetitionDirection,
    InhibitionScopeSignature,
    InhibitoryTrace,
    MemoryIdentity,
    MemoryKind,
)
from cogkura.storage.postgres import PostgresInhibitionStore

pytestmark = pytest.mark.postgres
_T = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
async def memory_engine() -> AsyncIterator[AsyncEngine]:
    import os

    url = os.environ.get("COGKURA_POSTGRES_MEMORY_URL")
    if url is None:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    engine = create_async_engine(url)
    await apply_migrations(engine)
    yield engine
    await engine.dispose()


def _trace(*, request_id: str | None, trace_id: str) -> InhibitoryTrace:
    return InhibitoryTrace(
        id=trace_id,
        tenant_id="tenant",
        inhibited_identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="loser"),
        selected_identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="winner"),
        direction=CompetitionDirection.CO_TEMPORAL,
        scope=InhibitionScopeSignature(
            structural_anchor=BehavioralStructuralAnchor.SUBJECT_PREDICATE,
            subject_id="payments-api",
            predicate="deployment_system",
            entity_ids=("payments-api",),
        ),
        competition_strength=0.8,
        competitor_accessibility=0.7,
        induction_pressure=0.56,
        retrieval_evaluated_at=_T,
        induced_at=_T,
        request_id=request_id,
    )


@pytest.mark.asyncio
async def test_postgres_inhibition_round_trip_and_request_idempotency(
    memory_engine: AsyncEngine,
) -> None:
    store = PostgresInhibitionStore(memory_engine)
    await store.clear(tenant_id="tenant")
    trace = _trace(request_id="req-1", trace_id="11111111-1111-1111-1111-111111111111")
    await store.append_traces([trace])
    await store.append_traces(
        [_trace(request_id="req-1", trace_id="22222222-2222-2222-2222-222222222222")]
    )
    listed = await store.list_for_memory(
        tenant_id="tenant",
        identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="loser"),
    )
    assert len(listed) == 1
    assert listed[0].direction is CompetitionDirection.CO_TEMPORAL
    assert listed[0].scope.scope_key == trace.scope.scope_key
    assert listed[0].induction_pressure == pytest.approx(0.56)
    await store.clear(tenant_id="tenant")


def _scope(anchor: BehavioralStructuralAnchor, **kwargs: object) -> InhibitionScopeSignature:
    return InhibitionScopeSignature(structural_anchor=anchor, **kwargs)  # type: ignore[arg-type]


def _stored(
    *,
    trace_id: str,
    induced_at: datetime,
    scope: InhibitionScopeSignature,
    request_id: str | None = None,
) -> InhibitoryTrace:
    return InhibitoryTrace(
        id=trace_id,
        tenant_id="tenant",
        inhibited_identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="loser"),
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


def _loser() -> MemoryIdentity:
    return MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="loser")


@pytest.mark.asyncio
async def test_postgres_repeated_use_without_request_id(memory_engine: AsyncEngine) -> None:
    store = PostgresInhibitionStore(memory_engine)
    await store.clear(tenant_id="tenant")
    scope = _scope(
        BehavioralStructuralAnchor.SEMANTIC_SLOT,
        subject_id="payments-api",
        predicate="deployment_system",
        semantic_slot_key="slot",
    )
    first = _stored(
        trace_id="00000000-0000-0000-0000-00000000000a",
        induced_at=_T,
        scope=scope,
    )
    second = _stored(
        trace_id="00000000-0000-0000-0000-00000000000b",
        induced_at=_T + timedelta(seconds=1),
        scope=scope,
    )
    await store.append_traces([first])
    await store.append_traces([second])
    listed = await store.list_for_memory(tenant_id="tenant", identity=_loser())
    assert [trace.id for trace in listed] == [first.id, second.id]
    await store.clear(tenant_id="tenant")


@pytest.mark.asyncio
async def test_postgres_as_of_order_and_scope_round_trip(memory_engine: AsyncEngine) -> None:
    store = PostgresInhibitionStore(memory_engine)
    await store.clear(tenant_id="tenant")
    fixtures = (
        _stored(
            trace_id="00000000-0000-0000-0000-000000000003",
            induced_at=_T + timedelta(hours=2),
            scope=_scope(
                BehavioralStructuralAnchor.SEMANTIC_SLOT,
                subject_id="payments-api",
                predicate="deployment_system",
                semantic_slot_key="slot-b",
            ),
        ),
        _stored(
            trace_id="00000000-0000-0000-0000-000000000001",
            induced_at=_T,
            scope=_scope(
                BehavioralStructuralAnchor.QUERY_SCOPE,
                entity_ids=("payments-api",),
                feature_ids=("deploy",),
            ),
        ),
        _stored(
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
    await store.append_traces(fixtures)
    listed = await store.list_for_memory(tenant_id="tenant", identity=_loser())
    expected = sorted(
        fixtures, key=lambda trace: (trace.induced_at, trace.scope.scope_key, trace.id)
    )
    assert [trace.id for trace in listed] == [trace.id for trace in expected]
    by_id = {trace.id: trace for trace in listed}
    for original in fixtures:
        hydrated = by_id[original.id]
        assert hydrated.scope.to_canonical_dict() == original.scope.to_canonical_dict()
        assert hydrated.scope.scope_key == original.scope.scope_key
        assert hydrated.scope.structural_anchor is original.scope.structural_anchor
    cutoff = _T + timedelta(hours=1)
    visible = await store.list_for_memory(tenant_id="tenant", identity=_loser(), as_of=cutoff)
    assert [trace.id for trace in visible] == [
        trace.id for trace in expected if trace.induced_at <= cutoff
    ]
    await store.clear(tenant_id="tenant")


@pytest.mark.asyncio
async def test_postgres_list_for_memories_matches_bounds(memory_engine: AsyncEngine) -> None:
    store = PostgresInhibitionStore(memory_engine)
    await store.clear(tenant_id="tenant")
    scope = _scope(
        BehavioralStructuralAnchor.SEMANTIC_SLOT,
        subject_id="payments-api",
        predicate="deployment_system",
        semantic_slot_key="slot",
    )
    older = _stored(
        trace_id="00000000-0000-0000-0000-000000000021",
        induced_at=_T,
        scope=scope,
    )
    newer = _stored(
        trace_id="00000000-0000-0000-0000-000000000022",
        induced_at=_T + timedelta(hours=3),
        scope=scope,
    )
    await store.append_traces([older, newer])
    listed = await store.list_for_memories(
        tenant_id="tenant",
        identities=[_loser()],
        after=_T + timedelta(hours=1),
        before_or_at=_T + timedelta(days=1),
        limit_per_memory=1,
    )
    assert [trace.id for trace in listed[_loser()]] == [newer.id]
    assert (
        await store.list_for_memories(
            tenant_id="tenant",
            identities=[],
            before_or_at=_T,
        )
        == {}
    )
    await store.clear(tenant_id="tenant")
