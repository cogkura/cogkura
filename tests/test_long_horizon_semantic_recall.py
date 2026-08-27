"""Regression tests for 0.15.2 long-horizon semantic recall."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    CognitiveTraceOrigin,
    MemoryKind,
    RecallInspectionDisposition,
)

_TENANT = "shop"
_SUBJECT = "customer_42"
_JACKET_QUERY = (
    "I'm looking for a waterproof jacket for a hiking trip next month. What would you recommend?"
)
_GOAL = "Help the customer choose an appropriate waterproof hiking jacket."


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


@pytest.mark.asyncio
async def test_twenty_day_jacket_size_soft_admitted_for_string_query() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 7, 12, 10, 0, tzinfo=UTC)
    query_time = evidence_time + timedelta(days=20)
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
    semantic_predicates = {
        result.memory.predicate for result in results if result.memory_kind is MemoryKind.SEMANTIC
    }
    assert "jacket_size" in semantic_predicates

    historical = await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        valid_at=query_time,
        limit=10,
    )
    historical_predicates = {
        result.memory.predicate
        for result in historical
        if result.memory_kind is MemoryKind.SEMANTIC
    }
    assert "jacket_size" in historical_predicates


@pytest.mark.asyncio
async def test_unrelated_current_semantic_not_soft_admitted() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 7, 12, 10, 0, tzinfo=UTC)
    query_time = evidence_time + timedelta(days=20)
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
            content="Customer jacket size updated to M.",
        )
    )
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
    payment_rejected = [
        item
        for item in inspection.rejected
        if item.memory_kind is MemoryKind.SEMANTIC
        and getattr(item.memory, "predicate", None) == "preferred_payment_method"
    ]
    assert payment_rejected
    assert payment_rejected[0].disposition in {
        RecallInspectionDisposition.BELOW_THRESHOLD,
        RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
    }
    assert not payment_rejected[0].soft_admitted


@pytest.mark.asyncio
async def test_superseded_size_not_soft_admitted_after_update() -> None:
    memory = _memory()
    t_l = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    t_m = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    query_time = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
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

    results = await memory.recall(
        "recommend a jacket",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    active_sizes = [
        result
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC and result.memory.predicate == "jacket_size"
    ]
    assert len(active_sizes) == 1
    assert active_sizes[0].memory.object_value.casefold() == "m"


@pytest.mark.asyncio
async def test_old_browse_episode_stays_weak_for_jacket_query() -> None:
    memory = _memory()
    browse_time = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)
    query_time = browse_time + timedelta(days=120)
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


@pytest.mark.asyncio
async def test_repeated_lightweight_support_strengthens_recall() -> None:
    memory = _memory()
    t1 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t2 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)
    query_time = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
    fact = {
        "predicate": "outerwear_weight_preference",
        "object_value": "lightweight",
        "cardinality": "one",
        "polarity": "affirm",
        "qualifiers": {},
    }
    await memory.observe(
        _semantic_observation(
            source_record_id="light-1",
            conversation_id="conv-light-1",
            semantic_fact=fact,
            observed_at=t1,
            content="Customer prefers lightweight outerwear.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t1)
    await memory.observe(
        _semantic_observation(
            source_record_id="light-2",
            conversation_id="conv-light-2",
            semantic_fact=fact,
            observed_at=t2,
            content="Customer praised the lightweight shell purchase.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t2)

    inspection = await memory.inspect_recall(
        "recommend a lightweight jacket",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    lightweight = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC
        and item.memory.predicate == "outerwear_weight_preference"
    )
    supported_traces = [
        trace
        for trace in lightweight.cognitive_traces
        if trace.origin is CognitiveTraceOrigin.SUPPORTED
    ]
    assert len(supported_traces) >= 2

    results = await memory.recall(
        "recommend a lightweight jacket",
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
async def test_repeated_process_does_not_duplicate_support_traces() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    query_time = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    fact = {
        "predicate": "hiking_interest",
        "object_value": "high",
        "cardinality": "one",
        "polarity": "affirm",
        "qualifiers": {},
    }
    await memory.observe(
        _semantic_observation(
            source_record_id="hike-1",
            conversation_id="conv-hike-1",
            semantic_fact=fact,
            observed_at=evidence_time,
            content="Customer confirmed hiking interest.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    inspection = await memory.inspect_recall(
        "hiking jacket recommendation",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=query_time,
        limit=10,
    )
    hiking = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "hiking_interest"
    )
    supported = [
        trace for trace in hiking.cognitive_traces if trace.origin is CognitiveTraceOrigin.SUPPORTED
    ]
    assert len(supported) == 1


@pytest.mark.asyncio
async def test_prepare_context_returns_non_empty_for_jacket_query() -> None:
    memory = _memory()
    evidence_time = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    query_time = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
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
            content="Customer jacket size updated to M.",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="hike",
            conversation_id="conv-hike",
            semantic_fact={
                "predicate": "hiking_interest",
                "object_value": "high",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer enjoys weekend hiking.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    context = await memory.prepare_context(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        goal=_GOAL,
        as_of=query_time,
        valid_at=query_time,
        prompt_budget_tokens=750,
    )
    assert context.working_memory.items
