"""Pass-local decisions observe the existing pipeline and preserve its counters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import timedelta
from typing import Any

import pytest
from test_source_assessment import T0, TENANT, adopt, memory, observation

from cogkura import (
    Memory,
    SemanticDiagnosticsConfig,
)
from cogkura import (
    SemanticDecisionOperation as Operation,
)
from cogkura import (
    SemanticDecisionOutcome as Outcome,
)
from cogkura import (
    SemanticDecisionReason as Reason,
)
from cogkura.algorithms.semantic import MetadataSemanticExtractor
from cogkura.exceptions import StorageError
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore


def decisions(result: Any, operation: Operation) -> list[Any]:
    assert result.semantics.decisions is not None
    return [i for i in result.semantics.decisions.items if i.operation is operation]


async def test_admission_reports_are_optional_and_replay_is_not_new_application() -> None:
    off, on = memory(), memory(diagnostics=True)
    for m in (off, on):
        await m.observe(observation("original"))
    off_result, on_result = [await m.process(tenant_id=TENANT, as_of=T0) for m in (off, on)]
    assert off_result.semantics.decisions is None
    assert replace(on_result.semantics, decisions=None) == off_result.semantics
    item = decisions(on_result, Operation.ADMIT)[0]
    assert item.outcome is Outcome.APPLIED
    assert item.asserting_evidence is not None
    replay = await on.process(tenant_id=TENANT, as_of=T0)
    assert decisions(replay, Operation.ADMIT)[0].outcome is Outcome.ALREADY_APPLIED
    assert replay.semantics.authority_promoted == 1  # Old per-pass counter stays unchanged.


async def test_unknown_correction_target_is_deferred_not_invalid() -> None:
    m = memory(diagnostics=True)
    await m.observe(observation("missing", correction="unknown", at=T0 + timedelta(days=1)))
    result = await m.process(tenant_id=TENANT, as_of=T0 + timedelta(days=1))
    assert result.semantics.corrections_deferred == 1
    assert result.semantics.corrections_invalid == 0
    item = decisions(result, Operation.REPLACE)[0]
    assert item.outcome is Outcome.DEFERRED
    assert item.reasons == (Reason.TARGET_UNAVAILABLE,)


async def test_competing_corrections_report_each_actual_declaration_once() -> None:
    m = memory(diagnostics=True)
    key = await adopt(m)
    t1 = T0 + timedelta(days=1)
    for n in (1, 2):
        await m.observe(
            observation(f"correction-{n}", value=f"pipeline-{n}", at=t1, correction=key)
        )
    result = await m.process(tenant_id=TENANT, as_of=t1)
    items = decisions(result, Operation.REPLACE)
    assert len(items) == 2
    assert all(
        i.outcome is Outcome.DEFERRED and i.reasons == (Reason.COMPETING_CORRECTIONS,)
        for i in items
    )
    assert result.semantics.corrections_deferred == 2
    assert result.semantics.authoritative_corrections_applied == 0


async def test_replace_withdraw_conflict_has_no_applied_record() -> None:
    m = memory(diagnostics=True)
    key = await adopt(m)
    t1 = T0 + timedelta(days=1)
    await m.observe(observation("correction", value="pipeline", at=t1, correction=key))
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
    result = await m.process(tenant_id=TENANT, as_of=t1)
    edits = decisions(result, Operation.REPLACE) + decisions(result, Operation.WITHDRAW)
    assert len(edits) == 2
    assert all(
        i.outcome is Outcome.DEFERRED and i.reasons == (Reason.REPLACE_WITHDRAW_CONFLICT,)
        for i in edits
    )
    assert result.semantics.corrections_deferred == 2


@pytest.mark.parametrize("invalid", ["authority", "time", "malformed"])
async def test_invalid_corrections_are_observed_at_the_rejecting_branch(invalid: str) -> None:
    m = memory(diagnostics=True)
    key = await adopt(m)
    t1 = T0 + timedelta(days=1)
    obs = observation(
        "correction",
        value="pipeline",
        at=t1,
        correction=key,
        authority=invalid != "authority",
        effective=T0 if invalid == "time" else t1,
    )
    if invalid == "malformed":
        facts = [dict(obs.metadata["semantic_facts"][0], correction={"wrong": "shape"})]
        obs = replace(obs, metadata={**obs.metadata, "semantic_facts": facts})
    await m.observe(obs)
    result = await m.process(tenant_id=TENANT, as_of=t1)
    item = decisions(result, Operation.REPLACE)[0]
    assert item.outcome is Outcome.INVALID
    assert item.reasons == (
        {
            "authority": Reason.AUTHORITY_REQUIRED,
            "time": Reason.INVALID_EFFECTIVE_TIME,
            "malformed": Reason.MALFORMED_DECLARATION,
        }[invalid],
    )
    assert result.semantics.corrections_invalid == 1


async def test_correction_and_withdrawal_reports_track_committed_replay() -> None:
    m = memory(diagnostics=True)
    key = await adopt(m)
    t1, t2 = T0 + timedelta(days=1), T0 + timedelta(days=2)
    await m.observe(observation("correction", value="pipeline", at=t1, correction=key))
    result = await m.process(tenant_id=TENANT, as_of=t1)
    edit = decisions(result, Operation.REPLACE)[0]
    assert edit.outcome is Outcome.APPLIED
    assert edit.resulting_revision_key
    replay = await m.process(tenant_id=TENANT, as_of=t1)
    assert decisions(replay, Operation.REPLACE)[0].outcome is Outcome.ALREADY_APPLIED
    await m.observe(
        observation(
            "withdraw",
            at=t2,
            withdrawals=[
                {
                    "target_revision_key": edit.resulting_revision_key,
                    "effective_at": t2.isoformat(),
                    "authority": {"basis": "application_asserted"},
                }
            ],
        )
    )
    result = await m.process(tenant_id=TENANT, as_of=t2)
    assert decisions(result, Operation.WITHDRAW)[0].outcome is Outcome.APPLIED
    assert result.semantics.authoritative_withdrawals_applied == 1
    replay = await m.process(tenant_id=TENANT, as_of=t2)
    # Existing 0.18.2 qualification can defer the original target on replay;
    # diagnostics must report that branch rather than invent a new decision.
    assert decisions(replay, Operation.WITHDRAW)[0].outcome is Outcome.DEFERRED
    assert decisions(replay, Operation.WITHDRAW)[0].reasons == (Reason.TARGET_NOT_UNIQUE_CURRENT,)
    assert replay.semantics.authoritative_withdrawals_applied == 0


async def test_authority_deferrals_and_malformed_authority_are_explained() -> None:
    m = memory(diagnostics=True)
    await adopt(m)
    await m.observe(observation("competing", value="pipeline"))
    result = await m.process(tenant_id=TENANT, as_of=T0)
    assert any(
        i.reasons == (Reason.AUTHORITY_COMPETES_WITH_CURRENT,)
        for i in decisions(result, Operation.ADMIT)
    )
    bad = observation("bad", value="other")
    bad = replace(
        bad,
        metadata={
            **bad.metadata,
            "semantic_facts": [dict(bad.metadata["semantic_facts"][0], authority="invalid")],
        },
    )
    await m.observe(bad)
    result = await m.process(tenant_id=TENANT, as_of=T0)
    assert any(
        i.outcome is Outcome.INVALID and i.reasons == (Reason.MALFORMED_DECLARATION,)
        for i in decisions(result, Operation.ADMIT)
    )


async def test_diagnostics_bounds_and_fixed_replay_order() -> None:
    m = memory(semantic_diagnostics_config=SemanticDiagnosticsConfig(enabled=True, max_items=1))
    for n in range(4):
        await m.observe(observation(f"claim-{n}", value=f"value-{n}", cardinality="many"))
    result = await m.process(tenant_id=TENANT, as_of=T0)
    report = result.semantics.decisions
    assert report and len(report.items) == 1 and report.is_truncated and report.total_decisions == 4
    again = await m.process(tenant_id=TENANT, as_of=T0)
    assert [i.claim_key for i in report.items] == [
        i.claim_key for i in again.semantics.decisions.items
    ]


async def test_disabled_mechanisms_are_unevaluated_and_custom_extractor_is_incomplete() -> None:
    m = Memory(semantic_diagnostics_config=SemanticDiagnosticsConfig(enabled=True))
    await m.observe(observation("ignored", correction="unknown"))
    result = await m.process(tenant_id=TENANT, as_of=T0)
    report = result.semantics.decisions
    assert report and not report.authority_evaluated and not report.corrections_evaluated
    assert report.items == ()

    class CustomExtractor(MetadataSemanticExtractor):
        pass

    m = memory(diagnostics=True, semantic_extractor=CustomExtractor())
    await m.observe(observation("original"))
    result = await m.process(tenant_id=TENANT, as_of=T0)
    assert not result.semantics.decisions.diagnostics_complete


async def test_failed_storage_never_returns_a_success_report() -> None:
    class FailingStore(InMemorySemanticMemoryStore):
        async def apply_reconciliation(self, *args: Any, **kwargs: Any) -> Any:
            raise StorageError("write failed")

    m = memory(diagnostics=True, store=FailingStore())
    await m.observe(observation("original"))
    with pytest.raises(StorageError, match="write failed"):
        await m.process(tenant_id=TENANT, as_of=T0)


async def semantic_state(m: Memory) -> Any:
    registry = {
        o.id: (o.source_namespace, o.source_record_id)
        for o in await m._observation_store.list(tenant_id=TENANT)
    }
    registry.update({e.id: e.memory_key for e in await m.list_episodes(tenant_id=TENANT)})

    def convert(value: Any) -> Any:
        if is_dataclass(value):
            return {
                f.name: convert(getattr(value, f.name)) for f in fields(value) if f.name != "id"
            }
        if isinstance(value, Mapping):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(convert(v) for v in value)
        return registry.get(value, value) if isinstance(value, str) else value

    return convert(await m.list_semantic_revisions(tenant_id=TENANT))


async def test_diagnostics_on_off_preserve_all_old_counters_history_recall_and_render() -> None:
    off = memory()
    # Identical stored evidence identities, with independent semantic/retrieval stores.
    on = memory(
        diagnostics=True, observation_store=off._observation_store, episode_store=off._episode_store
    )
    keys = [await adopt(m) for m in (off, on)]
    t1 = T0 + timedelta(days=1)
    for m in (off, on):
        await m.observe(observation("correction", value="pipeline", at=t1, correction=keys[0]))
    for _ in range(3):
        a, b = [await m.process(tenant_id=TENANT, as_of=t1) for m in (off, on)]
        assert a.semantics == replace(b.semantics, decisions=None)
        assert await semantic_state(off) == await semantic_state(on)
        contexts = [
            await m.prepare_context("current deployment policy", tenant_id=TENANT, as_of=t1)
            for m in (off, on)
        ]
        assert contexts[0].render() == contexts[1].render()
        assert [
            (r.memory_kind, r.memory.memory_key, r.score, r.activation, r.components)
            for r in contexts[0].recall_results
        ] == [
            (r.memory_kind, r.memory.memory_key, r.score, r.activation, r.components)
            for r in contexts[1].recall_results
        ]
