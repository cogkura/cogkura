"""Source diagnostics classify stored evidence without altering cognitive behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from cogkura import (
    ActivationConfig,
    DecisionHistoryAvailability,
    EvidenceQuality,
    Memory,
    ObservationInput,
    SemanticCorrectionConfig,
    SemanticDiagnosticsConfig,
    SemanticKnowledgeState,
    SemanticMemoryStatus,
    SemanticRevisionViewState,
    SemanticSlotQuery,
    SourceAssessmentReason,
    SourceConflictState,
    SourceEvidenceResolutionStatus,
    SourceIndependenceState,
    SourceInspectionLimits,
)
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator, semantic_slot_key
from cogkura.exceptions import ValidationError
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore

T0 = datetime(2026, 11, 1, tzinfo=UTC)
TENANT = "source-assessment"
SLOT = SemanticSlotQuery("service", "deployment_policy")


def memory(
    *, diagnostics: bool = False, store: Any = None, threshold: float = -10, **stores: Any
) -> Memory:
    return Memory(
        semantic_store=store,
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
        semantic_diagnostics_config=stores.pop(
            "semantic_diagnostics_config", SemanticDiagnosticsConfig(enabled=diagnostics)
        ),
        activation_config=ActivationConfig(retrieval_threshold=threshold),
        **stores,
    )


def observation(
    record: str,
    *,
    value: str = "startup",
    at: datetime = T0,
    version: str = "1",
    locator: bool = True,
    authority: bool = True,
    correction: str | None = None,
    effective: datetime | None = None,
    withdrawals: list[dict[str, Any]] | None = None,
    cardinality: str = "one",
    tenant: str = TENANT,
    deleted: bool = False,
) -> ObservationInput:
    fact: dict[str, Any] = {
        "subject_entity_id": "service",
        "predicate": "deployment_policy",
        "object_value": value,
        "cardinality": cardinality,
    }
    if authority:
        fact["authority"] = {"basis": "application_asserted"}
    if correction is not None:
        fact["correction"] = {
            "replaces_revision_key": correction,
            "effective_at": (effective or at).isoformat(),
        }
    metadata: dict[str, Any] = {"conversation_id": record, "semantic_facts": [fact]}
    if locator:
        metadata["source_ref"] = f"docs/{record}-v{version}.md"
    if withdrawals is not None:
        metadata["semantic_facts"] = []
        metadata["semantic_withdrawals"] = withdrawals
    return ObservationInput(
        tenant_id=tenant,
        subject_id="service",
        source_namespace="docs",
        source_record_id=record,
        source_version=version,
        content=f"Deployment policy is {value} ({record}, {version}).",
        observed_at=at,
        metadata=metadata,
        is_deleted=deleted,
    )


async def adopt(m: Memory) -> str:
    await m.observe(observation("original"))
    await m.process(tenant_id=TENANT, as_of=T0)
    return (await m.list_semantic_revisions(tenant_id=TENANT))[0].revision_key


async def test_episode_relation_coverage_is_not_applicable_and_locator_is_separate() -> None:
    m = memory()
    await m.observe(observation("plain", locator=False, authority=False))
    await m.encode_episodes(tenant_id=TENANT, as_of=T0)
    context = await m.prepare_context("deployment policy", tenant_id=TENANT, as_of=T0)
    report = await m.assess_source_memory(context)
    assert report.quality.has_evidence
    assert report.quality.revision_coverage_complete
    assert not report.quality.locator_coverage_complete
    assert report.quality.relation_applicable_count == 0
    assert report.quality.relation_not_applicable_count == report.quality.referenced_count
    assert report.quality.relation_coverage_complete is None
    assert not report.provenance.is_complete  # Preserve the older contract.
    assert SourceAssessmentReason.LOCATOR_MISSING in report.reasons


async def test_context_semantic_relation_coverage_and_shared_members() -> None:
    m = memory()
    await adopt(m)
    context = await m.prepare_context("deployment policy", tenant_id=TENANT, as_of=T0)
    report = await m.assess_source_memory(context)
    assert report.quality.relation_applicable_count > 0
    assert report.quality.relation_coverage_complete is True
    assert report.quality.distinct_source_count == 1
    assert report.quality.referenced_count >= report.quality.distinct_source_count
    for item in report.items:
        if item.revision_key is not None:
            assert item.quality.relation_coverage_complete is True
        else:
            assert item.quality.relation_coverage_complete is None
    assert report.provenance == await m.resolve_provenance(context)
    assert await m.assess_source_memory(context) == report


async def test_empty_context_is_not_a_claim_of_unknown_knowledge() -> None:
    m = memory()
    ctx = await m.prepare_context("anything", tenant_id=TENANT, as_of=T0)
    report = await m.assess_source_memory(ctx)
    assert report.items == ()
    assert not report.quality.has_evidence
    assert report.quality.relation_coverage_complete is None


async def test_exact_historical_revision_and_currentness() -> None:
    m = memory()
    await adopt(m)
    ctx = await m.prepare_context("deployment policy", tenant_id=TENANT, as_of=T0)
    await m.observe(
        observation("original", version="2", value="pipeline", at=T0 + timedelta(days=1))
    )
    report = await m.assess_source_memory(ctx)
    assert any(i.currentness.older_revision_count for i in report.items)
    assert {s.source_version for i in report.provenance.items for s in i.sources} == {"1"}
    slot = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert slot.state is SemanticKnowledgeState.KNOWN
    assert all(s.is_latest_known_revision is None for i in slot.revisions for s in i.sources)


async def test_unknown_known_and_inaccessible_are_distinct() -> None:
    m = memory(threshold=100)
    unknown = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert unknown.state is SemanticKnowledgeState.UNKNOWN
    assert unknown.decision_history is DecisionHistoryAvailability.UNAVAILABLE
    await adopt(m)
    assert await m.recall("unrelated bananas", tenant_id=TENANT, as_of=T0) == []
    known = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert known.state is SemanticKnowledgeState.KNOWN
    assert known.quality.revision_coverage_complete


async def test_correction_projection_hides_future_status_and_links() -> None:
    m = memory()
    predecessor = await adopt(m)
    t1, t2 = T0 + timedelta(days=1), T0 + timedelta(days=2)
    await m.observe(
        observation("correction", value="pipeline", at=t1, correction=predecessor, effective=t2)
    )
    await m.process(tenant_id=TENANT, as_of=t1)
    before = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0, valid_at=t2)
    assert before.state is SemanticKnowledgeState.KNOWN
    assert len(before.revisions) == 1
    assert before.revisions[0].revision_key == predecessor
    assert before.revisions[0].projected_status is SemanticMemoryStatus.ACTIVE
    assert before.revisions[0].valid_until is None
    assert before.revisions[0].transitions == ()
    pending = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t1)
    assert pending.state is SemanticKnowledgeState.KNOWN
    assert any(
        i.view_state is SemanticRevisionViewState.PENDING_EFFECTIVE for i in pending.revisions
    )
    at = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t2)
    assert at.state is SemanticKnowledgeState.KNOWN
    assert [
        i.revision_key
        for i in at.revisions
        if i.view_state is SemanticRevisionViewState.VISIBLE_CURRENT
    ] != [predecessor]
    historical = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t2, valid_at=T0)
    assert historical.state is SemanticKnowledgeState.KNOWN
    assert any(
        i.revision_key == predecessor and i.view_state is SemanticRevisionViewState.VISIBLE_CURRENT
        for i in historical.revisions
    )


async def test_explicit_withdrawal_and_source_deletion_are_different() -> None:
    m = memory()
    key = await adopt(m)
    t1 = T0 + timedelta(days=1)
    await m.observe(observation("original", version="2", at=t1, deleted=True))
    assert (
        await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t1)
    ).state is SemanticKnowledgeState.KNOWN
    await m.observe(
        observation(
            "withdraw",
            at=t1,
            withdrawals=[
                {
                    "target_revision_key": key,
                    "effective_at": t1.isoformat(),
                    "authority": {"basis": "application_asserted"},
                }
            ],
        )
    )
    await m.process(tenant_id=TENANT, as_of=t1)
    withdrawn = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=t1)
    assert withdrawn.state is SemanticKnowledgeState.WITHDRAWN
    assert SourceAssessmentReason.FACT_WITHDRAWN in withdrawn.reasons
    before = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert before.state is SemanticKnowledgeState.KNOWN
    assert len(before.revisions) == 1
    assert before.revisions[0].transitions == ()


async def test_inventory_and_evidence_limits_are_not_false_completeness() -> None:
    m = memory()
    key = await adopt(m)
    t1 = T0 + timedelta(days=1)
    await m.observe(observation("correction", value="pipeline", at=t1, correction=key))
    await m.process(tenant_id=TENANT, as_of=t1)
    truncated = await m.inspect_semantic_slot(
        SLOT, tenant_id=TENANT, as_of=t1, limits=SourceInspectionLimits(max_revisions=1)
    )
    assert not truncated.inventory_complete
    assert truncated.state is SemanticKnowledgeState.INDETERMINATE
    evidence = await m.inspect_semantic_slot(
        SLOT, tenant_id=TENANT, as_of=t1, limits=SourceInspectionLimits(max_evidence=1)
    )
    assert evidence.quality.is_truncated
    assert not evidence.quality.revision_coverage_complete
    assert evidence == await m.inspect_semantic_slot(
        SLOT, tenant_id=TENANT, as_of=t1, limits=SourceInspectionLimits(max_evidence=1)
    )


async def test_contested_sources_do_not_claim_independence() -> None:
    m = Memory(activation_config=ActivationConfig(retrieval_threshold=-10))
    for value in ("startup", "pipeline"):
        for n in range(2):
            obs = observation(f"{value}-{n}", value=value, authority=False)
            fact = dict(
                obs.metadata["semantic_facts"][0],
                valid_from=T0.isoformat(),
                valid_until=(T0 + timedelta(days=10)).isoformat(),
            )
            obs = replace(obs, metadata={**obs.metadata, "semantic_facts": [fact]})
            await m.observe(obs)
    await m.process(tenant_id=TENANT, as_of=T0)
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert report.state is SemanticKnowledgeState.CONTESTED
    assert report.conflict.state is SourceConflictState.SEMANTIC_CONFLICT
    assert report.conflict.cross_source_disagreement is True
    assert report.conflict.independence is SourceIndependenceState.UNKNOWN


async def seed_contested(m: Memory) -> None:
    """Two ordinary independent identities per claim, with overlapping validity."""
    for value in ("startup", "pipeline"):
        for n in range(2):
            obs = observation(f"{value}-{n}", value=value, authority=False)
            fact = dict(
                obs.metadata["semantic_facts"][0],
                valid_from=T0.isoformat(),
                valid_until=(T0 + timedelta(days=10)).isoformat(),
            )
            await m.observe(replace(obs, metadata={**obs.metadata, "semantic_facts": [fact]}))
    await m.process(tenant_id=TENANT, as_of=T0)


async def test_slot_prefix_ignores_uuid_derived_sequences_and_preserves_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = []
    sequences = []
    for allocation in ((40, 30, 20, 10), (10, 20, 30, 40)):
        ids = iter(allocation)
        m = memory()
        with monkeypatch.context() as patch:
            patch.setattr(
                "cogkura.storage.in_memory_observation.uuid4", lambda ids=ids: UUID(int=next(ids))
            )
            await seed_contested(m)
        stored = await m.list_semantic_memories(tenant_id=TENANT)
        registry = {
            o.id: o.source_record_id for o in await m._observation_store.list(tenant_id=TENANT)
        }
        sequence = {
            s.memory_key: tuple(
                registry[e.observation_id]
                for e in sorted(s.observation_evidence, key=lambda e: e.sequence_number)
            )
            for s in stored
        }
        sequences.append(sequence)
        context = await m.prepare_context("deployment policy", tenant_id=TENANT, as_of=T0)
        provenance = await m.resolve_provenance(context)
        assessment = await m.assess_source_memory(context)
        full = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
        logical = [
            (s.source_namespace, s.source_record_id, s.observation_revision)
            for r in full.revisions
            for s in r.sources
        ]
        reports.append(logical)
        assert full.state is SemanticKnowledgeState.CONTESTED
        assert logical == [
            ("docs", f"{value}-{n}", 1) for value in ("pipeline", "startup") for n in range(2)
        ]
        for cap in (1, 2, 3, 4, 5):
            cut = await m.inspect_semantic_slot(
                SLOT, tenant_id=TENANT, as_of=T0, limits=SourceInspectionLimits(max_evidence=cap)
            )
            assert [s for r in cut.revisions for s in r.sources] == [
                s for r in full.revisions for s in r.sources
            ][:cap]
            assert cut.quality.referenced_count == 4
            assert cut.quality.is_truncated is (cap < 4)
            assert cut.quality.revision_coverage_complete is (cap >= 4)
            assert cut.state is full.state
        assert await m.list_semantic_memories(tenant_id=TENANT) == stored
        assert await m.resolve_provenance(context) == provenance
        assert await m.assess_source_memory(context) == assessment
        assert assessment.provenance == provenance
        for item in provenance.items:
            if item.memory_identity.memory_key in sequence:
                assert (
                    tuple(s.source_record_id for s in item.sources)
                    == sequence[item.memory_identity.memory_key]
                )
    assert sequences[0] != sequences[1]  # A UUID tie-break-only fix would fail.
    assert reports[0] == reports[1]


async def test_slot_orders_mixed_availability_without_metadata_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = memory()
    await seed_contested(m)
    store = m._observation_store
    rows = await store.list(tenant_id=TENANT)
    ids = {o.source_record_id: o.id for o in rows}
    store.revisions[ids["startup-0"]].clear()
    store.revisions[ids["pipeline-0"]].clear()
    del store._observations[(TENANT, "docs", "pipeline-0")]
    store.revisions[ids["startup-1"]][0].metadata.pop("source_ref")
    calls = []
    original_current, original_revisions = store.get_many, store.get_revisions_many

    async def current(**kwargs: Any) -> Any:
        calls.append(("current", len(kwargs["observation_ids"])))
        return await original_current(**kwargs)

    async def revisions(**kwargs: Any) -> Any:
        calls.append(("revision", len(kwargs["references"])))
        return await original_revisions(**kwargs)

    monkeypatch.setattr(store, "get_many", current)
    monkeypatch.setattr(store, "get_revisions_many", revisions)
    full = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    sources = [s for r in full.revisions for s in r.sources]
    assert [s.source_record_id for s in sources] == ["pipeline-1", None, "startup-0", "startup-1"]
    assert [s.resolution_status for s in sources] == [
        SourceEvidenceResolutionStatus.RESOLVED,
        SourceEvidenceResolutionStatus.SOURCE_UNAVAILABLE,
        SourceEvidenceResolutionStatus.REVISION_UNAVAILABLE,
        SourceEvidenceResolutionStatus.SOURCE_REF_MISSING,
    ]
    assert sources[2].source_version is sources[2].source_ref is sources[2].observed_at is None
    assert sources[2].is_latest_known_revision is None
    assert full.state is SemanticKnowledgeState.CONTESTED
    assert not full.quality.revision_coverage_complete
    assert not full.quality.locator_coverage_complete
    calls.clear()
    cut = await m.inspect_semantic_slot(
        SLOT, tenant_id=TENANT, as_of=T0, limits=SourceInspectionLimits(max_evidence=1)
    )
    assert [s for r in cut.revisions for s in r.sources] == sources[:1]
    assert sorted(calls) == [("current", 4), ("revision", 4)]
    assert cut.quality.referenced_count == 4
    assert cut.quality.is_truncated


async def test_slot_future_evidence_cannot_consume_cap_or_determine_visible_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = memory()
    await seed_contested(m)
    store = m._observation_store
    original = store.get_revisions_many
    future_name = "a-future-secret"

    async def revisions(**kwargs: Any) -> Any:
        rows = await original(**kwargs)
        return {
            identity: replace(
                row,
                observed_at=T0 + timedelta(days=1),
                source_namespace=future_name,
                source_record_id=future_name,
                source_version=future_name,
                metadata={"source_ref": future_name},
            )
            if row.source_record_id == "pipeline-0"
            else row
            for identity, row in rows.items()
        }

    monkeypatch.setattr(store, "get_revisions_many", revisions)
    reports = []
    for name in ("a-future-secret", "z-future-secret"):
        future_name = name
        report = await m.inspect_semantic_slot(
            SLOT, tenant_id=TENANT, as_of=T0, limits=SourceInspectionLimits(max_evidence=1)
        )
        assert [s.source_record_id for r in report.revisions for s in r.sources] == ["pipeline-1"]
        assert report.quality.referenced_count == 3
        assert report.quality.returned_count == 1
        assert report.state is SemanticKnowledgeState.INDETERMINATE
        assert SourceAssessmentReason.TEMPORAL_HISTORY_UNAVAILABLE in report.reasons
        assert name not in repr(report)
        reports.append(report)
    assert reports[0] == reports[1]


async def test_slot_global_cap_preserves_repeated_occurrences_across_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = memory()
    await seed_contested(m)
    original = m._semantic_store.inspect_slot

    async def inspect_slot(**kwargs: Any) -> Any:
        records = await original(**kwargs)
        # Two revision snapshots may cite the same exact observation revisions.
        evidence = records.revisions[0].snapshot["observation_evidence"]
        return replace(
            records,
            revisions=tuple(
                replace(r, snapshot={**r.snapshot, "observation_evidence": evidence})
                for r in records.revisions
            ),
        )

    monkeypatch.setattr(m._semantic_store, "inspect_slot", inspect_slot)
    full = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    sources = [s for r in full.revisions for s in r.sources]
    assert len(sources) == 4
    assert [s.observation_id for s in sources[:2]] == [s.observation_id for s in sources[2:]]
    assert full.quality.distinct_source_count == 2
    assert full.quality.referenced_count == 4
    for cap in (1, 2, 3, 4, 5):
        cut = await m.inspect_semantic_slot(
            SLOT, tenant_id=TENANT, as_of=T0, limits=SourceInspectionLimits(max_evidence=cap)
        )
        assert [s for r in cut.revisions for s in r.sources] == sources[:cap]
        assert cut.quality.referenced_count == 4
        assert cut.quality.returned_count == min(cap, 4)


async def test_slot_case_sensitive_source_identity_and_numeric_revision_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = memory()
    for namespace, record in (("a", "a"), ("A", "a"), ("A", "Z")):
        for revision in range(1, 11):
            await m.observe(
                replace(observation(record, version=str(revision)), source_namespace=namespace)
            )
    await m.process(tenant_id=TENANT, as_of=T0)
    store = m._semantic_store
    original = store.inspect_slot

    async def inspect_slot(**kwargs: Any) -> Any:
        records = await original(**kwargs)
        # A supported custom snapshot can retain two revisions of one observation.
        revisions = []
        for revision in records.revisions:
            snapshot = dict(revision.snapshot)
            evidence = snapshot["observation_evidence"]
            snapshot["observation_evidence"] = [
                *evidence,
                *[dict(e, observation_revision=2) for e in evidence],
            ]
            revisions.append(replace(revision, snapshot=snapshot))
        return replace(records, revisions=tuple(revisions))

    monkeypatch.setattr(store, "inspect_slot", inspect_slot)
    full = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    expected = [
        (namespace, record, revision)
        for namespace, record in (("A", "Z"), ("A", "a"), ("a", "a"))
        for revision in (2, 10)
    ]
    assert [
        (s.source_namespace, s.source_record_id, s.observation_revision)
        for r in full.revisions
        for s in r.sources
    ] == expected
    assert full.quality.revision_coverage_complete
    cut = await m.inspect_semantic_slot(
        SLOT, tenant_id=TENANT, as_of=T0, limits=SourceInspectionLimits(max_evidence=3)
    )
    assert [s for r in cut.revisions for s in r.sources] == [
        s for r in full.revisions for s in r.sources
    ][:3]


async def test_cardinality_many_coexists_without_disagreement() -> None:
    m = memory()
    for value in ("startup", "pipeline"):
        await m.observe(observation(value, value=value, cardinality="many"))
    await m.process(tenant_id=TENANT, as_of=T0)
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert report.state is SemanticKnowledgeState.KNOWN
    assert report.conflict.state is SourceConflictState.NONE_OBSERVED
    assert report.conflict.cross_source_disagreement is False


async def test_missing_evidence_does_not_erase_known_status_or_substitute_current() -> None:
    m = memory()
    await adopt(m)
    observation_store = m._observation_store
    observation_store.revisions.clear()
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert report.state is SemanticKnowledgeState.KNOWN
    assert not report.quality.revision_coverage_complete
    assert all(s.source_version is None for i in report.revisions for s in i.sources)


async def test_legacy_history_is_indeterminate_without_invented_promotion() -> None:
    store = InMemorySemanticMemoryStore()
    m = memory(store=store)
    await adopt(m)
    for k, revision in store._revisions.items():
        store._revisions[k] = replace(revision, snapshot={})
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    assert report.state is SemanticKnowledgeState.INDETERMINATE
    assert SourceAssessmentReason.LEGACY_HISTORY_INCOMPLETE in report.reasons
    assert report.revisions[0].promotion is None


async def test_custom_store_without_inspection_is_unavailable_not_unknown() -> None:
    class LegacyStore:
        pass

    report = await memory(store=LegacyStore()).inspect_semantic_slot(
        SLOT, tenant_id=TENANT, as_of=T0
    )
    assert report.state is SemanticKnowledgeState.INDETERMINATE
    assert SourceAssessmentReason.INSPECTION_CAPABILITY_UNAVAILABLE in report.reasons


async def test_tenant_subject_isolation_and_no_cognitive_state_changes() -> None:
    m = memory()
    await adopt(m)
    ctx = await m.prepare_context("deployment policy", tenant_id=TENANT, as_of=T0)
    before = repr(
        (
            m._observation_store.__dict__,
            m._episode_store.__dict__,
            m._semantic_store.__dict__,
            m._activation_store.__dict__,
            m._dynamics_store.__dict__,
            m._learning_store.__dict__,
            m._inhibition_store.__dict__,
            m._checkpoint_store.__dict__,
        )
    )
    for _ in range(3):
        await m.assess_source_memory(ctx)
        await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    after = repr(
        (
            m._observation_store.__dict__,
            m._episode_store.__dict__,
            m._semantic_store.__dict__,
            m._activation_store.__dict__,
            m._dynamics_store.__dict__,
            m._learning_store.__dict__,
            m._inhibition_store.__dict__,
            m._checkpoint_store.__dict__,
        )
    )
    assert before == after
    assert ctx == await m.prepare_context("deployment policy", tenant_id=TENANT, as_of=T0)
    assert (
        await m.inspect_semantic_slot(SLOT, tenant_id="foreign", as_of=T0)
    ).state is SemanticKnowledgeState.UNKNOWN
    assert (
        await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, subject_id="other", as_of=T0)
    ).state is SemanticKnowledgeState.UNKNOWN


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_limits_are_positive_integers(value: Any) -> None:
    with pytest.raises(ValidationError):
        SourceInspectionLimits(max_evidence=value)
    with pytest.raises(ValidationError):
        SemanticDiagnosticsConfig(max_items=value)


def test_models_validate_and_freeze_query_values() -> None:
    original = {"environment": "production"}
    slot = SemanticSlotQuery("service", "policy", original)
    original["environment"] = "test"
    assert slot.qualifiers["environment"] == "production"
    with pytest.raises(TypeError):
        slot.qualifiers["environment"] = "test"
    with pytest.raises(FrozenInstanceError):
        slot.predicate = "changed"
    for invalid in ({"x": []}, {"x": float("nan")}, {1: "x"}):
        with pytest.raises(ValidationError):
            SemanticSlotQuery("service", "policy", invalid)
    with pytest.raises(ValidationError):
        EvidenceQuality(1, 2, 2, 2, 2, 1, 0, 1)


async def test_source_api_validation() -> None:
    m = memory()
    with pytest.raises(ValidationError):
        await m.inspect_semantic_slot(SLOT, tenant_id=" ")
    with pytest.raises(ValidationError):
        await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, valid_at=datetime(2026, 1, 1))
    ctx = await m.prepare_context("x", tenant_id=TENANT, as_of=T0)
    with pytest.raises(ValidationError):
        await m.assess_source_memory(ctx, max_evidence=True)


def test_shared_slot_key_preserves_canonicalization() -> None:
    key = semantic_slot_key(
        tenant_id=TENANT,
        subject_entity_id=" SERVICE ",
        predicate=" Deployment_Policy ",
        qualifiers={},
    )
    assert key == semantic_slot_key(
        tenant_id=TENANT, subject_entity_id="service", predicate="deployment_policy", qualifiers={}
    )


async def test_negative_assertion_and_expired_history_are_not_withdrawal() -> None:
    m = memory()
    obs = observation("denial")
    fact = dict(
        obs.metadata["semantic_facts"][0],
        polarity="deny",
        valid_from=T0.isoformat(),
        valid_until=(T0 + timedelta(days=1)).isoformat(),
    )
    await m.observe(replace(obs, metadata={**obs.metadata, "semantic_facts": [fact]}))
    await m.process(tenant_id=TENANT, as_of=T0)
    current = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0)
    expired = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0 + timedelta(days=2))
    assert current.state is SemanticKnowledgeState.KNOWN
    assert expired.state is SemanticKnowledgeState.HISTORICAL_ONLY
    assert SourceAssessmentReason.FACT_WITHDRAWN not in expired.reasons


async def test_unsupported_ordinary_history_exposes_no_present_status_or_false_clear_conflict() -> (
    None
):
    m = memory()
    await adopt(m)
    # Reinforcement updates a revision without retaining all previous counter states.
    await m.process(tenant_id=TENANT, as_of=T0 + timedelta(days=2))
    report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0 + timedelta(days=1))
    assert report.state is SemanticKnowledgeState.INDETERMINATE
    assert report.conflict.state is SourceConflictState.INDETERMINATE
    assert all(i.projected_status is None for i in report.revisions)
    assert SourceAssessmentReason.TEMPORAL_HISTORY_UNAVAILABLE in report.reasons


async def test_semantic_relation_unknown_is_not_episodic_not_applicable() -> None:
    from cogkura import SourceEvidence, SourceEvidenceResolutionStatus
    from cogkura.algorithms.source_assessment import evidence_quality

    source = SourceEvidence(
        observation_id="observation",
        observation_revision=1,
        source_namespace="docs",
        source_record_id="record",
        source_version="1",
        source_ref=None,
        observed_at=T0,
        contribution_score=1.0,
        relation=None,
        resolution_status=SourceEvidenceResolutionStatus.SOURCE_REF_MISSING,
        is_latest_known_revision=True,
    )
    quality = evidence_quality((source,), referenced_count=1, semantic_references=1)
    assert quality.revision_coverage_complete
    assert quality.relation_coverage_complete is False
    assert quality.relation_not_applicable_count == 0


def test_omitted_non_applicable_episode_does_not_lower_semantic_relation_coverage() -> None:
    quality = EvidenceQuality(
        referenced_count=2,
        returned_count=1,
        revision_resolved_count=1,
        locator_present_count=1,
        relation_known_count=1,
        relation_applicable_count=1,
        relation_not_applicable_count=1,
        distinct_source_count=1,
    )
    assert quality.is_truncated
    assert quality.relation_coverage_complete is True
    assert not quality.revision_coverage_complete


async def test_cap_omissions_do_not_report_complete_source_currentness() -> None:
    m = memory()
    await adopt(m)
    ctx = await m.prepare_context("deployment policy", tenant_id=TENANT, as_of=T0)
    report = await m.assess_source_memory(ctx, max_evidence=1)
    assert report.quality.is_truncated
    for item in report.items:
        if item.quality.is_truncated:
            assert item.currentness.is_truncated
            assert not item.currentness.is_complete


async def test_multiple_correction_boundaries_restore_only_the_known_predecessor() -> None:
    m = memory()
    original = await adopt(m)
    predecessor = original
    keys = [original]
    for n in (1, 2, 3):
        when = T0 + timedelta(days=n)
        await m.observe(
            observation(f"correction-{n}", value=f"pipeline-{n}", at=when, correction=predecessor)
        )
        await m.process(tenant_id=TENANT, as_of=when)
        predecessor = next(
            r.revision_key
            for r in await m.list_semantic_revisions(tenant_id=TENANT)
            if r.status is SemanticMemoryStatus.ACTIVE
        )
        keys.append(predecessor)
    for n, key in enumerate(keys):
        report = await m.inspect_semantic_slot(SLOT, tenant_id=TENANT, as_of=T0 + timedelta(days=n))
        assert report.state is SemanticKnowledgeState.KNOWN
        assert [
            i.revision_key
            for i in report.revisions
            if i.view_state is SemanticRevisionViewState.VISIBLE_CURRENT
        ] == [key]
        assert not set(keys[n + 1 :]) & {i.revision_key for i in report.revisions}
