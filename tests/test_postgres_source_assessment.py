"""Real PostgreSQL parity for the optional bounded inspection capability."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from test_source_assessment import SLOT, T0, TENANT, adopt, memory, observation

from cogkura import Memory, SemanticKnowledgeState, SourceInspectionLimits
from cogkura.migrations import apply_migrations
from cogkura.storage.postgres import (
    PostgresEpisodeStore,
    PostgresObservationStore,
    PostgresSemanticMemoryStore,
)

pytestmark = pytest.mark.postgres


@pytest.fixture
async def pg_memory(postgres_memory_url: str | None) -> AsyncIterator[Memory]:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    m = memory(
        diagnostics=True,
        store=PostgresSemanticMemoryStore(engine),
        observation_store=PostgresObservationStore(engine),
        episode_store=PostgresEpisodeStore(engine),
    )
    await m.clear(tenant_id=TENANT)
    yield m
    await m.clear(tenant_id=TENANT)
    await engine.dispose()


def normalized(report: Any, registry: dict[str, Any] | None = None) -> Any:
    obs_ids = {
        s.observation_id: (s.source_namespace, s.source_record_id)
        for r in getattr(report, "revisions", ())
        for s in r.sources
    }

    obs_ids.update(registry or {})

    def convert(value: Any) -> Any:
        if is_dataclass(value):
            return {f.name: convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, Mapping):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(convert(v) for v in value)
        if isinstance(value, str) and value in obs_ids:
            return obs_ids[value]
        return value

    return convert(report)


async def test_postgres_slot_adopt_replace_withdraw_replay_parity(pg_memory: Memory) -> None:
    from datetime import timedelta

    control = memory()
    keys = [await adopt(m) for m in (control, pg_memory)]
    assert keys[0] == keys[1]
    t1, t2 = T0 + timedelta(days=1), T0 + timedelta(days=2)
    for m in (control, pg_memory):
        await m.observe(
            observation("correction", value="pipeline", at=t1, correction=keys[0], effective=t2)
        )
        await m.process(tenant_id=TENANT, as_of=t1)
    for when, valid_at in ((T0, t2), (t1, None), (t2, None), (t2, T0)):
        reports = [
            await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=when, valid_at=valid_at)
            for m in (control, pg_memory)
        ]
        assert normalized(reports[0]) == normalized(reports[1])
        assert reports[0].state is SemanticKnowledgeState.KNOWN
    successor = next(
        r.revision_key
        for r in await control.list_semantic_revisions(tenant_id=TENANT)
        if r.revision_key != keys[0]
    )
    for m in (control, pg_memory):
        await m.observe(
            observation(
                "withdraw",
                at=t2,
                withdrawals=[
                    {
                        "target_revision_key": successor,
                        "effective_at": t2.isoformat(),
                        "authority": {"basis": "application_asserted"},
                    }
                ],
            )
        )
        # Effective time must follow the predecessor start.
    t3 = t2 + timedelta(days=1)
    for m in (control, pg_memory):
        await m.observe(
            observation(
                "withdraw",
                version="2",
                at=t3,
                withdrawals=[
                    {
                        "target_revision_key": successor,
                        "effective_at": t3.isoformat(),
                        "authority": {"basis": "application_asserted"},
                    }
                ],
            )
        )
        for _ in range(3):
            await m.process(tenant_id=TENANT, as_of=t3)
    reports = [
        await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t3)
        for m in (control, pg_memory)
    ]
    assert reports[0].state is SemanticKnowledgeState.WITHDRAWN
    registries = [
        {
            o.id: (o.source_namespace, o.source_record_id)
            for o in await m._observation_store.list(tenant_id=TENANT)
        }
        for m in (control, pg_memory)
    ]
    assert normalized(reports[0], registries[0]) == normalized(reports[1], registries[1])


async def test_postgres_inspection_has_constant_query_count_and_is_read_only(
    pg_memory: Memory,
) -> None:
    await adopt(pg_memory)
    engine = pg_memory._semantic_store._engine
    statements = []

    def capture(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        for _ in range(3):
            report = await pg_memory.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
            assert report.state is SemanticKnowledgeState.KNOWN
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) <= 36
    assert all(s.lstrip().upper().startswith(("SELECT", "SET TRANSACTION")) for s in statements)
    # Other tenants and subject scopes do not disclose this slot.
    assert (
        await pg_memory.inspect_semantic_slot(SLOT, tenant_id="other", as_of=T0)
    ).state is SemanticKnowledgeState.UNKNOWN
    assert (
        await pg_memory.inspect_semantic_slot(SLOT, tenant_id=TENANT, subject_id="other", as_of=T0)
    ).state is SemanticKnowledgeState.UNKNOWN


async def test_postgres_storage_bounds_have_stable_prefixes(pg_memory: Memory) -> None:
    from datetime import timedelta

    key = await adopt(pg_memory)
    await pg_memory.observe(
        observation("correction", value="pipeline", at=T0 + timedelta(days=1), correction=key)
    )
    await pg_memory.process(tenant_id=TENANT, as_of=T0 + timedelta(days=1))
    full = await pg_memory.inspect_semantic_slot(
        SLOT, tenant_id=TENANT, as_of=T0 + timedelta(days=1)
    )
    cut = await pg_memory.inspect_semantic_slot(
        SLOT,
        tenant_id=TENANT,
        as_of=T0 + timedelta(days=1),
        limits=SourceInspectionLimits(max_revisions=1),
    )
    assert not cut.inventory_complete
    assert cut.state is SemanticKnowledgeState.INDETERMINATE
    assert [r.revision_key for r in cut.revisions] == [r.revision_key for r in full.revisions[:1]]


async def test_postgres_pass_decisions_match_in_memory_with_competing_edits(
    pg_memory: Memory,
) -> None:
    from dataclasses import replace
    from datetime import timedelta

    control = memory(diagnostics=True)
    key = await adopt(control)
    assert await adopt(pg_memory) == key
    t1 = T0 + timedelta(days=1)
    for m in (control, pg_memory):
        for n in (1, 2):
            await m.observe(observation(f"edit-{n}", value=f"pipeline-{n}", at=t1, correction=key))
    a, b = [await m.process(tenant_id=TENANT, as_of=t1) for m in (control, pg_memory)]
    registries = [
        {
            o.id: (o.source_namespace, o.source_record_id)
            for o in await m._observation_store.list(tenant_id=TENANT)
        }
        for m in (control, pg_memory)
    ]
    assert a.semantics.corrections_deferred == b.semantics.corrections_deferred == 2
    assert replace(a.semantics, decisions=None) == replace(b.semantics, decisions=None)
    assert normalized(a.semantics.decisions, registries[0]) == normalized(
        b.semantics.decisions, registries[1]
    )
