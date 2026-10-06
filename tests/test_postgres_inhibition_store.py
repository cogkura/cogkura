"""PostgreSQL inhibitory trace store tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

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
