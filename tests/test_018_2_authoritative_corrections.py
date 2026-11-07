"""Authoritative corrections and withdrawals (0.18.2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import (
    ComplementaryLearningSemanticConsolidator,
    Memory,
    MemoryKind,
    ObservationInput,
    SemanticCorrectionConfig,
    SemanticMemoryStatus,
)
from cogkura.algorithms.corrections import SNAPSHOT_VERSION

_T = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
_CORRECTION = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
_LATER = datetime(2026, 10, 25, 10, 0, tzinfo=UTC)
_BEFORE = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
_TENANT = "repository"


def _memory(*, correction: bool = True) -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True,
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=correction),
    )


def _fact(
    predicate: str,
    object_value: str,
    *,
    authority: bool = True,
    correction: dict[str, object] | None = None,
) -> dict[str, object]:
    fact: dict[str, object] = {
        "predicate": predicate,
        "object_value": object_value,
        "cardinality": "one",
    }
    if authority:
        fact["authority"] = {"basis": "application_asserted"}
    if correction is not None:
        fact["correction"] = correction
    return fact


def _observation(
    *,
    record_id: str,
    content: str,
    facts: list[dict[str, object]] | None = None,
    observed_at: datetime | None = None,
    version: str = "1",
    tenant_id: str = _TENANT,
    subject_id: str = "payments-api",
    withdrawals: list[dict[str, object]] | None = None,
) -> ObservationInput:
    metadata: dict[str, object] = {"conversation_id": record_id}
    if facts is not None:
        metadata["semantic_facts"] = facts
    if withdrawals is not None:
        metadata["semantic_withdrawals"] = withdrawals
    return ObservationInput(
        tenant_id=tenant_id,
        subject_id=subject_id,
        source_namespace="repository",
        source_record_id=record_id,
        source_version=version,
        content=content,
        observed_at=observed_at or _T,
        metadata=metadata,
    )


async def _values(memory: Memory, *, valid_at: datetime | None = None) -> list[str]:
    memories = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        valid_at=valid_at,
    )
    return sorted(
        item.object_value for item in memories if item.status is not SemanticMemoryStatus.RETRACTED
    )


async def _adopt(memory: Memory, *, value: str = "application_startup") -> str:
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content=f"Strategy is {value}.",
            facts=[_fact("migration_execution_strategy", value)],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    return revisions[0].revision_key


@pytest.mark.asyncio
async def test_valid_correction_replaces_the_named_revision() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    historical = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        status=SemanticMemoryStatus.SUPERSEDED,
    )
    assert result.semantics.authoritative_corrections_applied == 1
    assert [item.object_value for item in current] == ["deployment_pipeline"]
    assert current[0].support_count == 1
    assert current[0].valid_from == _CORRECTION
    previous = next(item for item in historical if item.object_value == "application_startup")
    assert previous.status is SemanticMemoryStatus.SUPERSEDED
    assert previous.valid_until == _CORRECTION
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    successor = next(item for item in revisions if item.revision_key == current[0].revision_key)
    assert successor.snapshot["version"] == SNAPSHOT_VERSION
    assert successor.snapshot["correction"]["replaces_revision_key"] == predecessor
    assert successor.snapshot["observation_evidence"]


@pytest.mark.asyncio
async def test_historical_and_boundary_queries_follow_half_open_intervals() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_LATER)
    assert await _values(memory, valid_at=_BEFORE) == []
    assert await _values(memory, valid_at=datetime(2026, 10, 15, tzinfo=UTC)) == [
        "application_startup"
    ]
    historical_rows = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        valid_at=datetime(2026, 10, 15, tzinfo=UTC),
    )
    current_rows = await memory.list_semantic_memories(tenant_id=_TENANT, valid_at=_LATER)
    assert {item.observation_id for item in historical_rows[0].observation_evidence}.isdisjoint(
        {item.observation_id for item in current_rows[0].observation_evidence}
    )
    assert await _values(memory, valid_at=_CORRECTION) == ["deployment_pipeline"]
    assert await _values(memory, valid_at=_LATER) == ["deployment_pipeline"]
    historical = await memory.recall(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=_LATER,
        valid_at=datetime(2026, 10, 15, tzinfo=UTC),
    )
    semantic = [item for item in historical if item.memory_kind is MemoryKind.SEMANTIC]
    assert [item.memory.object_value for item in semantic] == ["application_startup"]
    current = await memory.recall(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=_LATER,
    )
    assert [
        item.memory.object_value for item in current if item.memory_kind is MemoryKind.SEMANTIC
    ] == ["deployment_pipeline"]
    context = await memory.prepare_context(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=_LATER,
        valid_at=datetime(2026, 10, 15, tzinfo=UTC),
    )
    rendered = context.render()
    assert "application_startup" in rendered
    assert "deployment_pipeline" not in rendered
    assert "replaces_revision_key" not in rendered


@pytest.mark.asyncio
async def test_future_correction_does_not_leak_into_earlier_knowledge_time() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_LATER)
    recalled = await memory.recall(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=_T,
    )
    values = [
        item.memory.object_value for item in recalled if item.memory_kind is MemoryKind.SEMANTIC
    ]
    assert values == ["application_startup"]


@pytest.mark.asyncio
async def test_disabled_correction_keeps_authority_deferral() -> None:
    memory = _memory(correction=False)
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert result.semantics.authoritative_corrections_applied == 0
    assert result.semantics.authority_deferred_conflict >= 1
    assert [item.object_value for item in current] == ["application_startup"]


@pytest.mark.asyncio
async def test_correction_without_authority_does_not_replace() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    authority=False,
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert result.semantics.authoritative_corrections_applied == 0
    assert result.semantics.corrections_invalid == 1
    assert [item.object_value for item in current] == ["application_startup"]


@pytest.mark.asyncio
async def test_unknown_predecessor_is_rejected() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": "missing-revision",
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    assert result.semantics.authoritative_corrections_applied == 0
    assert result.semantics.corrections_deferred == 1
    assert await memory.list_semantic_memories(tenant_id=_TENANT) == []


@pytest.mark.asyncio
async def test_repeated_processing_does_not_duplicate_the_correction() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    first = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    second = await memory.process(tenant_id=_TENANT, as_of=_LATER)
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    assert first.semantics.authoritative_corrections_applied == 1
    assert second.semantics.authoritative_corrections_applied == 0
    assert len(revisions) == 2
    assert [
        item.object_value for item in await memory.list_semantic_memories(tenant_id=_TENANT)
    ] == ["deployment_pipeline"]


@pytest.mark.asyncio
async def test_explicit_withdrawal_removes_the_current_value_and_keeps_history() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0001-withdraw",
            content="The migration strategy is withdrawn.",
            observed_at=_CORRECTION,
            withdrawals=[
                {
                    "target_revision_key": predecessor,
                    "effective_at": _CORRECTION.isoformat(),
                    "authority": {"basis": "application_asserted"},
                }
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    assert result.semantics.authoritative_withdrawals_applied == 1
    assert await _values(memory) == []
    assert await _values(memory, valid_at=datetime(2026, 10, 15, tzinfo=UTC)) == [
        "application_startup"
    ]
    recalled = await memory.recall(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=_LATER,
    )
    assert [item for item in recalled if item.memory_kind is MemoryKind.SEMANTIC] == []


@pytest.mark.asyncio
async def test_source_edit_without_withdrawal_does_not_retract() -> None:
    memory = _memory()
    await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            version="2",
            content="The document was edited without a withdrawal.",
            observed_at=_CORRECTION,
            facts=[_fact("migration_execution_strategy", "application_startup", authority=False)],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert result.semantics.authoritative_withdrawals_applied == 0
    assert [item.object_value for item in current] == ["application_startup"]
    assert current[0].status is SemanticMemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_reprocessing_does_not_restore_superseded_authority() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    await memory.process(tenant_id=_TENANT, as_of=_LATER)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    historical = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        status=SemanticMemoryStatus.SUPERSEDED,
    )
    assert [item.object_value for item in current] == ["deployment_pipeline"]
    previous = next(item for item in historical if item.object_value == "application_startup")
    assert previous.status is SemanticMemoryStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_competing_corrections_have_no_winner() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-a",
            content="Use the pipeline.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    await memory.observe(
        _observation(
            record_id="adr-b",
            content="Use manual migrations.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "manual",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert result.semantics.authoritative_corrections_applied == 0
    assert result.semantics.corrections_deferred == 2
    assert [item.object_value for item in current] == ["application_startup"]


@pytest.mark.asyncio
async def test_same_batch_correction_uses_the_predecessor_created_in_the_batch() -> None:
    predictor = _memory()
    predecessor = await _adopt(predictor)
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Strategy is application_startup.",
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_LATER)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert result.semantics.authoritative_corrections_applied == 1
    assert [item.object_value for item in current] == ["deployment_pipeline"]


@pytest.mark.asyncio
async def test_ordinary_recurrence_still_promotes_without_a_correction() -> None:
    memory = _memory()
    for record_id in ("note-1", "note-2"):
        await memory.observe(
            _observation(
                record_id=record_id,
                content="Repeated startup strategy.",
                facts=[
                    _fact(
                        "migration_execution_strategy",
                        "application_startup",
                        authority=False,
                    )
                ],
            )
        )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert result.semantics.recurrence_promoted == 1
    assert result.semantics.authoritative_corrections_applied == 0
    assert [item.object_value for item in current] == ["application_startup"]
    assert current[0].support_count == 2
