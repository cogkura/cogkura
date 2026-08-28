"""Regression tests for 0.15.3 semantic state and associative recall."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    MemoryKind,
    RecallInspectionDisposition,
    RetrievalEligibility,
    SemanticMemoryStatus,
)

_TENANT = "shop"
_SUBJECT = "customer_42"
_JACKET_QUERY = (
    "I'm looking for a waterproof jacket for a hiking trip next month. What would you recommend?"
)


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


def _semantic_observation(
    *,
    source_record_id: str,
    conversation_id: str,
    semantic_fact: dict,
    observed_at: datetime,
    content: str,
) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        actor_id=_SUBJECT,
        source_namespace="chat.messages",
        source_record_id=source_record_id,
        event_type="message",
        content=content,
        observed_at=observed_at,
        metadata={
            "conversation_id": conversation_id,
            "entity_ids": [_SUBJECT],
            "semantic_facts": [semantic_fact],
        },
    )


def _browse_observation(
    *,
    source_record_id: str,
    conversation_id: str,
    observed_at: datetime,
    content: str,
) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        actor_id=_SUBJECT,
        source_namespace="commerce.events",
        source_record_id=source_record_id,
        event_type="browse",
        content=content,
        observed_at=observed_at,
        metadata={
            "conversation_id": conversation_id,
            "entity_ids": [_SUBJECT],
        },
    )


async def _semantic_statuses(memory: Memory) -> dict[str, SemanticMemoryStatus]:
    active = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    superseded = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        status=SemanticMemoryStatus.SUPERSEDED,
    )
    return {
        f"{item.predicate}={item.object_value.casefold()}": item.status
        for item in (*active, *superseded)
    }


@pytest.mark.asyncio
async def test_cardinality_one_size_reconciles_to_single_authority() -> None:
    memory = _memory()
    t_l = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t_m = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    await memory.observe(
        _semantic_observation(
            source_record_id="size-l",
            conversation_id="conv-l",
            semantic_fact={
                "predicate": "jacket_size",
                "object_value": "L",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=t_l,
            content="Customer jacket size recorded as L.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_l)
    await memory.observe(
        _semantic_observation(
            source_record_id="size-m",
            conversation_id="conv-m",
            semantic_fact={
                "predicate": "jacket_size",
                "object_value": "M",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=t_m,
            content="Customer jacket size updated to M.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_m)

    statuses = await _semantic_statuses(memory)
    assert statuses["jacket_size=m"] is SemanticMemoryStatus.ACTIVE
    assert statuses["jacket_size=l"] is SemanticMemoryStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_many_cardinality_interests_coexist() -> None:
    memory = _memory()
    t1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)
    for source, value, content in (
        ("hike", "hiking", "Customer enjoys hiking."),
        ("ski", "skiing", "Customer enjoys skiing."),
    ):
        await memory.observe(
            _semantic_observation(
                source_record_id=source,
                conversation_id=f"conv-{source}",
                semantic_fact={
                    "predicate": "activity_interest",
                    "object_value": value,
                    "cardinality": "many",
                    "polarity": "affirm",
                    "qualifiers": {},
                },
                observed_at=t1 if source == "hike" else t2,
                content=content,
            )
        )
        await memory.process(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            as_of=t1 if source == "hike" else t2,
        )

    statuses = await _semantic_statuses(memory)
    assert statuses["activity_interest=hiking"] is SemanticMemoryStatus.ACTIVE
    assert statuses["activity_interest=skiing"] is SemanticMemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_ninety_day_current_size_admitted_without_floor() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    query_time = evidence_time + timedelta(days=90)
    await memory.observe(
        _semantic_observation(
            source_record_id="size-m",
            conversation_id="conv-size",
            semantic_fact={
                "predicate": "jacket_size",
                "object_value": "M",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer jacket size updated to M after fitting.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    results = await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    sizes = [
        result
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC and result.memory.predicate == "jacket_size"
    ]
    assert sizes
    assert sizes[0].memory.object_value.casefold() == "m"
    if sizes[0].diagnostics is not None:
        assert sizes[0].diagnostics.eligibility is RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION


@pytest.mark.asyncio
async def test_evidence_linked_product_fit_recall() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    query_time = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    await memory.observe(
        _semantic_observation(
            source_record_id="return",
            conversation_id="conv-return",
            semantic_fact={
                "predicate": "product_fit_issue",
                "object_value": "northpeak-alpine-shell:sleeves_too_short",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content=(
                "Customer returned the NorthPeak Alpine Shell waterproof jacket "
                "because the sleeves were too short."
            ),
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    results = await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    predicates = {
        result.memory.predicate for result in results if result.memory_kind is MemoryKind.SEMANTIC
    }
    assert "product_fit_issue" in predicates


@pytest.mark.asyncio
async def test_evidence_linked_lightweight_preference_recall() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    query_time = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    await memory.observe(
        _semantic_observation(
            source_record_id="light-jacket",
            conversation_id="conv-light",
            semantic_fact={
                "predicate": "outerwear_weight_preference",
                "object_value": "lightweight",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer loved the lightweight waterproof shell purchase.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    results = await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    predicates = {
        result.memory.predicate for result in results if result.memory_kind is MemoryKind.SEMANTIC
    }
    assert "outerwear_weight_preference" in predicates


@pytest.mark.asyncio
async def test_unrelated_payment_not_current_admitted() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    query_time = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    await memory.observe(
        _semantic_observation(
            source_record_id="payment",
            conversation_id="conv-payment",
            semantic_fact={
                "predicate": "preferred_payment_method",
                "object_value": "card",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer prefers card payments.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    inspection = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    payment = next(
        item
        for item in inspection.rejected
        if item.memory_kind is MemoryKind.SEMANTIC
        and item.memory.predicate == "preferred_payment_method"
    )
    assert payment.disposition in {
        RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
        RecallInspectionDisposition.BELOW_THRESHOLD,
    }
    assert not payment.soft_admitted


@pytest.mark.asyncio
async def test_old_browse_episode_not_authoritatively_admitted() -> None:
    memory = _memory()
    browse_time = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    query_time = browse_time + timedelta(days=90)
    await memory.observe(
        _browse_observation(
            source_record_id="backpack",
            conversation_id="conv-backpack",
            observed_at=browse_time,
            content="Customer browsed red backpacks online.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=browse_time)

    results = await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    assert not results
