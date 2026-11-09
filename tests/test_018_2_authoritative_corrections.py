"""Authoritative corrections and withdrawals (0.18.2)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import (
    ActivationConfig,
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


def _memory(*, correction: bool = True, retrieval_threshold: float | None = None) -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True,
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=correction),
        activation_config=(
            ActivationConfig(retrieval_threshold=retrieval_threshold)
            if retrieval_threshold is not None
            else None
        ),
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


@pytest.mark.asyncio
async def test_repeated_withdrawal_stays_withdrawn() -> None:
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
    applied = []
    for _ in range(3):
        result = await memory.process(tenant_id=_TENANT, as_of=_LATER)
        applied.append(result.semantics.authoritative_withdrawals_applied)
        assert await _values(memory) == []
        assert await _values(memory, valid_at=datetime(2026, 10, 15, tzinfo=UTC)) == [
            "application_startup"
        ]
        current = await memory.list_semantic_memories(tenant_id=_TENANT)
        assert current == [] or all(
            item.status is SemanticMemoryStatus.RETRACTED for item in current
        )
    assert applied == [1, 0, 0]
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    assert {item.status for item in revisions} == {
        SemanticMemoryStatus.ACTIVE,
        SemanticMemoryStatus.RETRACTED,
    }
    closed = next(item for item in revisions if item.status is SemanticMemoryStatus.ACTIVE)
    assert closed.valid_until == _CORRECTION
    assert closed.snapshot["closed_by"]["kind"] == "withdrawal"


@pytest.mark.asyncio
async def test_recall_before_withdrawal_keeps_the_known_decision() -> None:
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
    await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    recalled = await memory.recall(
        "payments-api migration execution strategy application startup",
        tenant_id=_TENANT,
        as_of=_T,
    )
    semantic = [item for item in recalled if item.memory_kind is MemoryKind.SEMANTIC]
    assert [item.memory.object_value for item in semantic] == ["application_startup"]
    assert semantic[0].memory.observation_evidence


@pytest.mark.asyncio
async def test_valid_at_after_withdrawal_has_no_authoritative_value() -> None:
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
    await memory.process(tenant_id=_TENANT, as_of=_LATER)
    assert await memory.list_semantic_memories(tenant_id=_TENANT, valid_at=_LATER) == []
    recalled = await memory.recall(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=_LATER,
        valid_at=_LATER,
    )
    assert [item for item in recalled if item.memory_kind is MemoryKind.SEMANTIC] == []
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    assert any(item.status is SemanticMemoryStatus.RETRACTED for item in revisions)


@pytest.mark.asyncio
async def test_future_effective_correction_stays_pending() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    effective = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
    between = datetime(2026, 10, 25, 10, 0, tzinfo=UTC)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations will run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": effective.isoformat(),
                    },
                )
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=between)
    pending = await memory.recall(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=between,
    )
    assert [
        item.memory.object_value for item in pending if item.memory_kind is MemoryKind.SEMANTIC
    ] == ["application_startup"]
    current = await memory.recall(
        "payments-api migration execution strategy",
        tenant_id=_TENANT,
        as_of=effective,
    )
    assert [
        item.memory.object_value for item in current if item.memory_kind is MemoryKind.SEMANTIC
    ] == ["deployment_pipeline"]
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    assert any(item.valid_from == effective for item in revisions)


@pytest.mark.asyncio
async def test_replacement_and_withdrawal_of_one_predecessor_both_defer() -> None:
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
    await memory.observe(
        _observation(
            record_id="adr-0003",
            content="Withdraw the strategy instead.",
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
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert result.semantics.authoritative_corrections_applied == 0
    assert result.semantics.authoritative_withdrawals_applied == 0
    assert result.semantics.corrections_deferred == 2
    assert [item.object_value for item in current] == ["application_startup"]
    assert current[0].status is SemanticMemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_withdrawal_snapshot_records_the_authorising_observation() -> None:
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
    await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    retracted = next(item for item in revisions if item.status is SemanticMemoryStatus.RETRACTED)
    historical = next(item for item in revisions if item.status is SemanticMemoryStatus.ACTIVE)
    authorising = retracted.snapshot["withdrawal"]["observation_id"]
    historical_ids = {
        item["observation_id"] for item in historical.snapshot["observation_evidence"]
    }
    assert retracted.snapshot["withdrawal"]["observation_revision"] == 1
    assert authorising not in historical_ids


@pytest.mark.asyncio
async def test_correcting_one_fact_keeps_another_fact_in_the_same_episode() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Startup migrations and a PostgreSQL database.",
            facts=[
                _fact("migration_execution_strategy", "application_startup"),
                _fact("database_engine", "postgresql"),
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    strategy = next(
        item
        for item in await memory.list_semantic_memories(tenant_id=_TENANT)
        if item.predicate == "migration_execution_strategy"
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
                        "replaces_revision_key": strategy.revision_key,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert {item.predicate: item.object_value for item in current}[
        "database_engine"
    ] == "postgresql"
    recalled = await memory.recall(
        "postgresql database",
        tenant_id=_TENANT,
        as_of=_CORRECTION,
    )
    assert recalled
    assert any(
        item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "database_engine"
        for item in recalled
    )


@pytest.mark.asyncio
async def test_same_batch_recurrence_blocks_an_explicit_correction() -> None:
    memory = _memory()
    predecessor = await _adopt(memory)
    for record_id in ("note-b1", "note-b2"):
        await memory.observe(
            _observation(
                record_id=record_id,
                content="Repeated deployment pipeline strategy.",
                observed_at=_CORRECTION,
                facts=[
                    _fact(
                        "migration_execution_strategy",
                        "deployment_pipeline",
                        authority=False,
                    )
                ],
            )
        )
    await memory.observe(
        _observation(
            record_id="adr-c",
            content="Use a canary release instead.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "canary_release",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    current = [
        item.object_value
        for item in await memory.list_semantic_memories(tenant_id=_TENANT)
        if item.status in (SemanticMemoryStatus.ACTIVE, SemanticMemoryStatus.CONTESTED)
    ]
    assert result.semantics.authoritative_corrections_applied == 0
    assert result.semantics.corrections_deferred >= 1
    assert "canary_release" not in current
    assert len(set(current)) == 1


@pytest.mark.asyncio
async def test_withdrawn_episode_is_not_a_current_decision() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Boot migrations use application_startup before traffic.",
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    predecessor = (await memory.list_semantic_revisions(tenant_id=_TENANT))[0].revision_key
    await memory.observe(
        _observation(
            record_id="adr-0001-withdraw",
            content="The boot migration strategy is withdrawn.",
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
    await memory.process(tenant_id=_TENANT, as_of=_LATER)
    query = "What is the migration strategy currently? Boot migrations use application_startup."
    recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_LATER)
    rendered = (await memory.prepare_context(query, tenant_id=_TENANT, as_of=_LATER)).render()
    assert "application_startup" not in rendered
    assert [
        item
        for item in recalled
        if getattr(item.memory, "object_value", None) == "application_startup"
        or "application_startup" in item.memory.statement
    ] == []


@pytest.mark.asyncio
async def test_shared_episode_keeps_unrelated_fact_after_withdrawal() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content=(
                "Boot migrations use application_startup before traffic. "
                "Ledger database is postgresql."
            ),
            facts=[
                _fact("migration_execution_strategy", "application_startup"),
                _fact("database_engine", "postgresql"),
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    strategy = next(
        item
        for item in await memory.list_semantic_memories(tenant_id=_TENANT)
        if item.predicate == "migration_execution_strategy"
    )
    await memory.observe(
        _observation(
            record_id="adr-0001-withdraw",
            content="The boot migration strategy is withdrawn.",
            observed_at=_CORRECTION,
            withdrawals=[
                {
                    "target_revision_key": strategy.revision_key,
                    "effective_at": _CORRECTION.isoformat(),
                    "authority": {"basis": "application_asserted"},
                }
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_LATER)
    withdrawn_query = "currently boot migrations use application_startup before traffic"
    withdrawn = await memory.recall(withdrawn_query, tenant_id=_TENANT, as_of=_LATER)
    withdrawn_render = (
        await memory.prepare_context(withdrawn_query, tenant_id=_TENANT, as_of=_LATER)
    ).render()
    assert "application_startup" not in withdrawn_render
    assert not any("application_startup" in item.memory.statement for item in withdrawn)
    live_query = "currently ledger database is postgresql"
    live = await memory.recall(live_query, tenant_id=_TENANT, as_of=_LATER)
    live_render = (
        await memory.prepare_context(live_query, tenant_id=_TENANT, as_of=_LATER)
    ).render()
    assert any(
        item.memory_kind is MemoryKind.SEMANTIC and item.memory.object_value == "postgresql"
        for item in live
    )
    assert "postgresql" in live_render


@pytest.mark.asyncio
async def test_withdrawal_preserves_an_unconsolidated_fact_in_a_mixed_episode() -> None:
    memory = _memory(retrieval_threshold=-10.0)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content=(
                "Boot migrations use application_startup before traffic. "
                "The ledger database is postgresql."
            ),
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    predecessor = (await memory.list_semantic_revisions(tenant_id=_TENANT))[0].revision_key
    await memory.observe(
        _observation(
            record_id="adr-0001-withdraw",
            content="The boot migration strategy is withdrawn.",
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
    await memory.process(tenant_id=_TENANT, as_of=_LATER)

    query = "ledger database postgresql"
    recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_LATER)
    rendered = (await memory.prepare_context(query, tenant_id=_TENANT, as_of=_LATER)).render()

    assert any(
        item.memory_kind is MemoryKind.EPISODE and "postgresql" in item.memory.statement
        for item in recalled
    )
    assert all("application_startup" not in item.memory.statement for item in recalled)
    assert "postgresql" in rendered
    assert "application_startup" not in rendered


@pytest.mark.asyncio
async def test_broad_current_query_cannot_leak_a_withdrawn_mixed_episode_assertion() -> None:
    memory = _memory()
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content=(
                "Boot migrations use application_startup before traffic. "
                "The ledger database is postgresql."
            ),
            facts=[
                _fact("migration_execution_strategy", "application_startup"),
                _fact("database_engine", "postgresql"),
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    predecessor = next(
        item
        for item in await memory.list_semantic_revisions(tenant_id=_TENANT)
        if item.snapshot["predicate"] == "migration_execution_strategy"
    ).revision_key
    await memory.observe(
        _observation(
            record_id="adr-0001-withdraw",
            content="The boot migration strategy is withdrawn.",
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
    await memory.process(tenant_id=_TENANT, as_of=_LATER)

    query = "Summarise the current ledger database architecture."
    recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_LATER)
    rendered = (await memory.prepare_context(query, tenant_id=_TENANT, as_of=_LATER)).render()

    assert all("application_startup" not in item.memory.statement for item in recalled)
    assert "application_startup" not in rendered
    assert "postgresql" in rendered


@pytest.mark.asyncio
async def test_current_valid_at_snapshot_cannot_leak_a_withdrawn_episode_assertion() -> None:
    memory = _memory(retrieval_threshold=-10.0)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content=(
                "Boot migrations use application_startup before traffic. "
                "The ledger database is postgresql."
            ),
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    predecessor = (await memory.list_semantic_revisions(tenant_id=_TENANT))[0].revision_key
    await memory.observe(
        _observation(
            record_id="adr-0001-withdraw",
            content="The boot migration strategy is withdrawn.",
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
    await memory.process(tenant_id=_TENANT, as_of=_LATER)

    query = "current ledger database architecture"
    recalled = await memory.recall(
        query,
        tenant_id=_TENANT,
        as_of=_LATER,
        valid_at=_LATER,
    )
    rendered = (
        await memory.prepare_context(
            query,
            tenant_id=_TENANT,
            as_of=_LATER,
            valid_at=_LATER,
        )
    ).render()

    assert all("application_startup" not in item.memory.statement for item in recalled)
    assert "application_startup" not in rendered
    assert "postgresql" in rendered


@pytest.mark.asyncio
async def test_future_valid_at_snapshot_keeps_the_predecessor_until_known() -> None:
    memory = _memory(retrieval_threshold=-10.0)
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
        "migration execution strategy deployment",
        tenant_id=_TENANT,
        as_of=_T,
        valid_at=_LATER,
    )

    semantic_values = [
        item.memory.object_value for item in recalled if item.memory_kind is MemoryKind.SEMANTIC
    ]
    assert semantic_values == ["application_startup"]
    assert all("deployment_pipeline" not in item.memory.statement for item in recalled)


@pytest.mark.asyncio
async def test_withdrawal_keeps_ordinary_sentence_with_a_shared_object_value() -> None:
    memory = _memory(retrieval_threshold=-10.0)
    await memory.observe(
        _observation(
            record_id="feature-flag",
            content="Feature flag is enabled. Cache is enabled.",
            facts=[_fact("feature_flag_state", "enabled")],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    predecessor = (await memory.list_semantic_revisions(tenant_id=_TENANT))[0].revision_key
    await memory.observe(
        _observation(
            record_id="feature-flag-withdraw",
            content="The feature flag is withdrawn.",
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
    await memory.process(tenant_id=_TENANT, as_of=_LATER)

    recalled = await memory.recall("cache enabled", tenant_id=_TENANT, as_of=_LATER)

    assert any(
        item.memory_kind is MemoryKind.EPISODE and item.memory.statement == "Cache is enabled."
        for item in recalled
    )
    assert all("Feature flag is enabled." not in item.memory.statement for item in recalled)


@pytest.mark.asyncio
async def test_readmission_after_withdrawal_does_not_restore_the_assertion() -> None:
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
    await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    before = await memory.list_semantic_revisions(tenant_id=_TENANT)
    await memory.observe(
        _observation(
            record_id="adr-0004",
            content="Strategy is application_startup again.",
            observed_at=_LATER,
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_LATER)
    after = await memory.list_semantic_revisions(tenant_id=_TENANT)
    assert result.semantics.authority_deferred_conflict >= 1
    assert await _values(memory) == []
    assert [item.revision_key for item in after] == [item.revision_key for item in before]
    assert not any(
        item.status is SemanticMemoryStatus.ACTIVE and item.valid_until is None for item in after
    )
