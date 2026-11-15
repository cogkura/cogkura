"""Cross-feature 0.18 contracts; focused algorithm coverage stays in its own suites."""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta
from typing import Any
from uuid import UUID

import pytest
from test_source_assessment import SLOT, T0, TENANT, adopt, memory, observation, seed_contested

from cogkura import (
    ComplementaryLearningSemanticConsolidator,
    Memory,
    SemanticCorrectionConfig,
    SemanticDecisionOperation,
    SemanticDecisionOutcome,
    SemanticDecisionReason,
    SemanticDiagnosticsConfig,
    SemanticKnowledgeState,
    SemanticRevisionViewState,
    SourceAssessmentReason,
    SourceIndependenceState,
    SourceInspectionLimits,
)


@pytest.fixture(params=["in_memory", pytest.param("postgres", marks=pytest.mark.postgres)])
async def source_memory(
    request: pytest.FixtureRequest, postgres_memory_url: str | None
) -> AsyncIterator[Memory]:
    engine = None
    if request.param == "postgres":
        if not postgres_memory_url:
            pytest.skip("COGKURA_POSTGRES_MEMORY_URL unset; freeze parity NOT EXECUTED")
        from sqlalchemy.ext.asyncio import create_async_engine

        from cogkura.migrations import apply_migrations
        from cogkura.storage.postgres import (
            PostgresEpisodeStore,
            PostgresObservationStore,
            PostgresSemanticMemoryStore,
        )

        engine = create_async_engine(postgres_memory_url)
        await apply_migrations(engine)
        m = memory(
            diagnostics=True,
            store=PostgresSemanticMemoryStore(engine),
            observation_store=PostgresObservationStore(engine),
            episode_store=PostgresEpisodeStore(engine),
        )
    else:
        m = memory(diagnostics=True)
    await m.clear(tenant_id=TENANT)
    try:
        yield m
    finally:
        await m.clear(tenant_id=TENANT)
        if engine is not None:
            await engine.dispose()


def test_source_defaults_preserve_opt_in_boundaries() -> None:
    assert not ComplementaryLearningSemanticConsolidator().allow_authoritative_single_episode
    assert not SemanticCorrectionConfig().enabled
    assert not SemanticDiagnosticsConfig().enabled


async def withdraw(m: Memory, key: str, when: Any) -> Any:
    await m.observe(
        replace(
            observation(
                "withdraw",
                at=when,
                withdrawals=[
                    {
                        "target_revision_key": key,
                        "effective_at": when.isoformat(),
                        "authority": {"basis": "application_asserted"},
                    }
                ],
            ),
            content="Deployment policy is withdrawn.",
        )
    )
    return await m.process(tenant_id=TENANT, as_of=when)


async def test_authority_correction_history_withdrawal_and_source_views(
    source_memory: Memory,
) -> None:
    m = source_memory
    original = await adopt(m)
    t1, t2, t3, t4 = (T0 + timedelta(days=n) for n in (1, 2, 3, 4))
    await m.observe(
        observation("correction", value="pipeline", at=t1, correction=original, effective=t2)
    )
    result = await m.process(tenant_id=TENANT, as_of=t1)
    edits = [
        d
        for d in result.semantics.decisions.items
        if d.operation is SemanticDecisionOperation.REPLACE
    ]
    assert len(edits) == result.semantics.authoritative_corrections_applied == 1
    assert edits[0].outcome is SemanticDecisionOutcome.APPLIED
    successor = edits[0].resulting_revision_key
    assert successor is not None
    for when, valid_at, expected in (
        (T0, t3, original),
        (t1, None, original),
        (t2, None, successor),
        (t3, T0, original),
    ):
        report = await m.inspect_semantic_slot(
            SLOT, tenant_id=TENANT, as_of=when, valid_at=valid_at
        )
        assert report.state is SemanticKnowledgeState.KNOWN
        assert [
            r.revision_key
            for r in report.revisions
            if r.view_state is SemanticRevisionViewState.VISIBLE_CURRENT
        ] == [expected]
        assert all(s.is_latest_known_revision is None for r in report.revisions for s in r.sources)
    result = await withdraw(m, successor, t3)
    assert result.semantics.authoritative_withdrawals_applied == 1
    before = await m.list_semantic_revisions(tenant_id=TENANT)
    replay = await m.process(tenant_id=TENANT, as_of=t3)
    assert replay.semantics.authoritative_withdrawals_applied == 0
    assert [r.revision_key for r in await m.list_semantic_revisions(tenant_id=TENANT)] == [
        r.revision_key for r in before
    ]
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t3)
    assert report.state is SemanticKnowledgeState.WITHDRAWN
    assert report.quality.revision_coverage_complete
    history = await m.prepare_context(
        "deployment policy startup", tenant_id=TENANT, as_of=t3, valid_at=T0
    )
    await m.observe(observation("original", at=t4, version="2", value="new-unprocessed-prose"))
    provenance = await m.resolve_provenance(history)
    sources = [s for i in provenance.items for s in i.sources if s.source_record_id == "original"]
    assert sources and all(
        s.source_version == "1" and s.is_latest_known_revision is False for s in sources
    )
    assessment = await m.assess_source_memory(history)
    assert assessment.provenance == provenance
    assert "new-unprocessed-prose" not in history.render()
    for tenant, subject in (("foreign", None), (TENANT, "foreign")):
        assert (
            await m.inspect_semantic_slot(SLOT, tenant_id=tenant, subject_id=subject, as_of=t4)
        ).state is SemanticKnowledgeState.UNKNOWN


async def test_conflict_prefixes_and_storage_bounds_remain_observational(
    source_memory: Memory,
) -> None:
    m = source_memory
    await seed_contested(m)
    before = await m.list_semantic_revisions(tenant_id=TENANT)
    full = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert full.state is SemanticKnowledgeState.CONTESTED
    assert full.conflict.independence is SourceIndependenceState.UNKNOWN
    sources = [s for r in full.revisions for s in r.sources]
    for r in full.revisions:
        logical = [
            (s.source_namespace, s.source_record_id, s.observation_revision) for s in r.sources
        ]
        assert logical == sorted(logical)
    assert len(sources) == 4
    for cap in (1, 2, 4, 5):
        cut = await m.inspect_semantic_slot(
            SLOT, tenant_id=TENANT, as_of=T0, limits=SourceInspectionLimits(max_evidence=cap)
        )
        assert [s for r in cut.revisions for s in r.sources] == sources[:cap]
        assert cut.state is SemanticKnowledgeState.CONTESTED
        assert cut.quality.referenced_count == 4
        assert cut.quality.is_truncated is (cap < 4)
        assert cut.quality.revision_coverage_complete is (cap >= 4)
    bounded = await m.inspect_semantic_slot(
        SLOT,
        tenant_id=TENANT,
        as_of=T0,
        limits=SourceInspectionLimits(max_revisions=1, max_relations=1),
    )
    assert len(bounded.revisions) == 1
    assert not bounded.inventory_complete
    assert bounded.state is SemanticKnowledgeState.INDETERMINATE
    assert await m.list_semantic_revisions(tenant_id=TENANT) == before


async def test_competing_edits_defer_without_inventing_contested_durable_state(
    source_memory: Memory,
) -> None:
    m = source_memory
    key = await adopt(m)
    t1 = T0 + timedelta(days=1)
    for n in (1, 2):
        await m.observe(observation(f"edit-{n}", value=f"pipeline-{n}", at=t1, correction=key))
    result = await m.process(tenant_id=TENANT, as_of=t1)
    edits = [
        d
        for d in result.semantics.decisions.items
        if d.operation is SemanticDecisionOperation.REPLACE
    ]
    assert len(edits) == result.semantics.corrections_deferred == 2
    assert all(
        d.outcome is SemanticDecisionOutcome.DEFERRED
        and d.reasons == (SemanticDecisionReason.COMPETING_CORRECTIONS,)
        for d in edits
    )
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t1)
    assert report.state is SemanticKnowledgeState.KNOWN
    assert [r.revision_key for r in report.revisions] == [key]
    assert report.decision_history.value == "unavailable"


@pytest.mark.parametrize("authority", [False, True])
async def test_withdrawn_readmission_never_renders_as_current_evidence(
    source_memory: Memory, authority: bool
) -> None:
    m = source_memory
    key = await adopt(m)
    t1, t2 = T0 + timedelta(days=1), T0 + timedelta(days=2)
    await withdraw(m, key, t1)
    obs = observation("readmission", authority=authority, at=t2)
    await m.observe(
        replace(obs, content="Deployment policy is startup again. Ledger database is postgresql.")
    )
    result = await m.process(tenant_id=TENANT, as_of=t2)
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t2)
    assert report.state is SemanticKnowledgeState.WITHDRAWN
    if authority:
        assert result.semantics.authority_deferred_conflict >= 1
    before = await m.list_semantic_revisions(tenant_id=TENANT)
    query = "current deployment policy startup ledger postgresql"
    recall = await m.recall(query, tenant_id=TENANT, as_of=t2)
    context = await m.prepare_context(query, tenant_id=TENANT, as_of=t2)
    assert "startup" not in context.render()
    assert "postgresql" in context.render()
    assert all("startup" not in r.memory.statement for r in recall)
    inspection = await m.inspect_recall(query, tenant_id=TENANT, as_of=t2)
    assert all("startup" not in r.memory.statement for r in inspection.returned)
    assert await m.list_semantic_revisions(tenant_id=TENANT) == before
    assert (
        "startup"
        in (
            await m.prepare_context(
                "deployment policy startup", tenant_id=TENANT, as_of=t2, valid_at=T0
            )
        ).render()
    )


@pytest.mark.parametrize(
    "difference",
    [
        {"subject_entity_id": "other"},
        {"predicate": "cache_policy"},
        {"object_value": "pipeline"},
        {"object_entity_id": "other"},
        {"polarity": "deny"},
        {"cardinality": "many"},
        {"qualifiers": {"environment": "preview"}},
        {"predicate": ""},
    ],
)
async def test_readmission_filter_requires_complete_exact_claim_identity(
    difference: dict[str, Any],
) -> None:
    m = memory()
    key = await adopt(m)
    t1, t2 = T0 + timedelta(days=1), T0 + timedelta(days=2)
    await withdraw(m, key, t1)
    obs = observation("different-claim", authority=False, at=t2)
    fact = {**obs.metadata["semantic_facts"][0], **difference}
    text = "Deployment policy is startup in this separate assertion."
    await m.observe(replace(obs, content=text, metadata={**obs.metadata, "semantic_facts": [fact]}))
    await m.process(tenant_id=TENANT, as_of=t2)
    context = await m.prepare_context("deployment policy startup", tenant_id=TENANT, as_of=t2)
    assert text in context.render()  # Generic lexical overlap cannot establish identity.


@pytest.mark.parametrize("original_matches", [False, True])
async def test_readmission_projection_uses_exact_episode_revision_not_latest_metadata(
    source_memory: Memory, original_matches: bool
) -> None:
    m = source_memory
    key = await adopt(m)
    t1, t2, t3 = (T0 + timedelta(days=n) for n in (1, 2, 3))
    await withdraw(m, key, t1)
    obs = observation("edited-readmission", authority=False, at=t2)
    fact = dict(
        obs.metadata["semantic_facts"][0],
        predicate="deployment_policy" if original_matches else "cache_policy",
    )
    text = "Deployment policy is startup again. Ledger database is postgresql."
    await m.observe(replace(obs, content=text, metadata={**obs.metadata, "semantic_facts": [fact]}))
    await m.process(tenant_id=TENANT, as_of=t2)
    latest = observation("edited-readmission", authority=False, at=t3, version="2")
    fact = dict(
        latest.metadata["semantic_facts"][0],
        predicate="cache_policy" if original_matches else "deployment_policy",
    )
    await m.observe(replace(latest, metadata={**latest.metadata, "semantic_facts": [fact]}))
    context = await m.prepare_context(
        "deployment policy startup ledger", tenant_id=TENANT, as_of=t3
    )
    assert ("startup" in context.render()) is not original_matches
    assert "postgresql" in context.render()


async def test_legacy_history_and_missing_exact_evidence_do_not_fabricate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = memory()
    await adopt(m)
    m._observation_store.revisions.clear()
    original = m._semantic_store.inspect_slot

    async def legacy(**kwargs: Any) -> Any:
        records = await original(**kwargs)
        return replace(records, revisions=tuple(replace(r, snapshot={}) for r in records.revisions))

    monkeypatch.setattr(m._semantic_store, "inspect_slot", legacy)
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert report.state is SemanticKnowledgeState.INDETERMINATE
    assert SourceAssessmentReason.LEGACY_HISTORY_INCOMPLETE in report.reasons
    assert all(
        r.projected_status is None and not r.historical_attribution_complete
        for r in report.revisions
    )


async def test_diagnostics_and_interleaved_source_reads_preserve_cognitive_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = []
    for enabled in (False, True):
        rng = random.Random(184)

        def identity(allocator: random.Random = rng) -> UUID:
            return UUID(int=allocator.getrandbits(128), version=4)

        with monkeypatch.context() as patch:
            for module in ("in_memory_observation", "in_memory_episode", "in_memory_semantic"):
                patch.setattr(f"cogkura.storage.{module}.uuid4", identity)
            m = memory(diagnostics=enabled)
            key = await adopt(m)
            t1 = T0 + timedelta(days=1)
            await m.observe(observation("correction", value="pipeline", at=t1, correction=key))
            result = await m.process(tenant_id=TENANT, as_of=t1)
            context = await m.prepare_context(
                "deployment policy pipeline", tenant_id=TENANT, as_of=t1
            )
            recall = await m.recall("deployment policy pipeline", tenant_id=TENANT, as_of=t1)
            if enabled:
                for _ in range(3):
                    await m.resolve_provenance(context)
                    await m.assess_source_memory(context)
                    await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t1)
            assert recall == await m.recall(
                "deployment policy pipeline", tenant_id=TENANT, as_of=t1
            )
            assert context == await m.prepare_context(
                "deployment policy pipeline", tenant_id=TENANT, as_of=t1
            )
            outputs.append(
                (
                    replace(result.semantics, decisions=None),
                    recall,
                    context,
                    await m.list_semantic_revisions(tenant_id=TENANT),
                )
            )
    assert outputs[0] == outputs[1]
