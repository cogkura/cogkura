"""0.15.11 cardinality-one temporal contract verification (no reconciler changes)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import SemanticMemoryStatus

_TENANT = "lab"
_SUBJECT = "person-1"
_T0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
_T1 = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
_T2 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
_T3 = datetime(2027, 1, 1, 10, 0, tzinfo=UTC)
_QUERY_AS_OF = datetime(2026, 12, 1, 12, 0, tzinfo=UTC)


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


def _size_fact(
    object_value: str,
    *,
    confidence: float = 1.0,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> dict:
    fact: dict = {
        "predicate": "size",
        "object_value": object_value,
        "cardinality": "one",
        "polarity": "affirm",
        "qualifiers": {},
        "confidence": confidence,
    }
    if valid_from is not None:
        fact["valid_from"] = valid_from
    if valid_until is not None:
        fact["valid_until"] = valid_until
    return fact


async def _observe_size(
    memory: Memory,
    *,
    source_record_id: str,
    object_value: str,
    observed_at: datetime,
    confidence: float = 1.0,
    valid_from: str | None = None,
    valid_until: str | None = None,
) -> None:
    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id=source_record_id,
            event_type="message",
            content=f"Recorded size {object_value}.",
            observed_at=observed_at,
            metadata={
                "conversation_id": source_record_id,
                "entity_ids": [_SUBJECT],
                "semantic_facts": [
                    _size_fact(
                        object_value,
                        confidence=confidence,
                        valid_from=valid_from,
                        valid_until=valid_until,
                    )
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)


async def _status_by_value(memory: Memory) -> dict[str, SemanticMemoryStatus]:
    active = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    superseded = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        status=SemanticMemoryStatus.SUPERSEDED,
    )
    contested = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        status=SemanticMemoryStatus.CONTESTED,
    )
    return {
        item.object_value.casefold(): item.status
        for item in (*active, *superseded, *contested)
        if item.predicate == "size"
    }


@pytest.mark.asyncio
async def test_implicit_sequential_m_to_l_supersedes() -> None:
    memory = _memory()
    await _observe_size(memory, source_record_id="size-m", object_value="M", observed_at=_T0)
    await _observe_size(memory, source_record_id="size-l", object_value="L", observed_at=_T1)
    statuses = await _status_by_value(memory)
    assert statuses["m"] is SemanticMemoryStatus.SUPERSEDED
    assert statuses["l"] is SemanticMemoryStatus.ACTIVE
    current = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    assert len([item for item in current if item.predicate == "size"]) == 1
    assert current[0].object_value.casefold() == "l"


@pytest.mark.asyncio
async def test_implicit_sequential_m_to_l_to_s_supersedes() -> None:
    memory = _memory()
    await _observe_size(memory, source_record_id="size-m", object_value="M", observed_at=_T0)
    await _observe_size(memory, source_record_id="size-l", object_value="L", observed_at=_T1)
    await _observe_size(memory, source_record_id="size-s", object_value="S", observed_at=_T2)
    statuses = await _status_by_value(memory)
    assert statuses["m"] is SemanticMemoryStatus.SUPERSEDED
    assert statuses["l"] is SemanticMemoryStatus.SUPERSEDED
    assert statuses["s"] is SemanticMemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_same_value_reinforces_without_contesting() -> None:
    memory = _memory()
    await _observe_size(memory, source_record_id="size-m-1", object_value="M", observed_at=_T0)
    before = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    assert len(before) == 1
    assert before[0].support_count == 1
    await _observe_size(memory, source_record_id="size-m-2", object_value="M", observed_at=_T1)
    after = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    assert len(after) == 1
    assert after[0].status is SemanticMemoryStatus.ACTIVE
    assert after[0].support_count >= 2


@pytest.mark.asyncio
async def test_explicit_non_overlapping_windows_resolve_by_valid_at() -> None:
    memory = _memory()
    await _observe_size(
        memory,
        source_record_id="size-m",
        object_value="M",
        observed_at=_T0,
        valid_from=_T0.isoformat(),
        valid_until=_T1.isoformat(),
    )
    await _observe_size(
        memory,
        source_record_id="size-l",
        object_value="L",
        observed_at=_T1,
        valid_from=_T1.isoformat(),
        valid_until=_T2.isoformat(),
    )
    before_cut = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        valid_at=_T0 + timedelta(hours=1),
    )
    after_cut = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        valid_at=_T1 + timedelta(hours=1),
    )
    assert any(item.object_value.casefold() == "m" for item in before_cut)
    assert all(item.object_value.casefold() != "l" for item in before_cut)
    assert any(item.object_value.casefold() == "l" for item in after_cut)


@pytest.mark.asyncio
async def test_explicit_overlapping_open_windows_mark_contested() -> None:
    memory = _memory()
    await _observe_size(
        memory,
        source_record_id="size-m",
        object_value="M",
        observed_at=_T0,
        valid_from=_T0.isoformat(),
    )
    await _observe_size(
        memory,
        source_record_id="size-l",
        object_value="L",
        observed_at=_T1,
        valid_from=_T1.isoformat(),
    )
    statuses = await _status_by_value(memory)
    assert statuses["m"] is SemanticMemoryStatus.CONTESTED
    assert statuses["l"] is SemanticMemoryStatus.CONTESTED


@pytest.mark.asyncio
async def test_implicit_supersession_closes_predecessor_valid_until() -> None:
    memory = _memory()
    await _observe_size(memory, source_record_id="size-m", object_value="M", observed_at=_T0)
    await _observe_size(memory, source_record_id="size-l", object_value="L", observed_at=_T1)
    superseded_list = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        status=SemanticMemoryStatus.SUPERSEDED,
    )
    superseded = next(item for item in superseded_list if item.object_value.casefold() == "m")
    assert superseded.valid_until is not None
    assert superseded.valid_until <= _T1


@pytest.mark.asyncio
async def test_lower_confidence_later_evidence_still_supersedes() -> None:
    memory = _memory()
    await _observe_size(
        memory,
        source_record_id="size-m",
        object_value="M",
        observed_at=_T0,
        confidence=0.9,
    )
    await _observe_size(
        memory,
        source_record_id="size-l",
        object_value="L",
        observed_at=_T1,
        confidence=0.2,
    )
    statuses = await _status_by_value(memory)
    assert statuses["m"] is SemanticMemoryStatus.SUPERSEDED
    assert statuses["l"] is SemanticMemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_implicit_supersession_is_deterministic() -> None:
    async def run_once() -> dict[str, SemanticMemoryStatus]:
        memory = _memory()
        await _observe_size(memory, source_record_id="size-m", object_value="M", observed_at=_T0)
        await _observe_size(memory, source_record_id="size-l", object_value="L", observed_at=_T1)
        return await _status_by_value(memory)

    first = await run_once()
    second = await run_once()
    assert first == second
