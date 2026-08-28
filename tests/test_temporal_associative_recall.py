"""Regression tests for 0.15.4 temporal semantic recall and associative reach."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.activation import (
    TemporalRetrievalMode,
    _temporal_retrieval_mode,
)
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    ActivationConfig,
    MemoryKind,
    RecallInspectionDisposition,
    RetrievalCue,
    RetrievalEligibility,
)

_TENANT = "shop"
_SUBJECT = "customer_42"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_JACKET_QUERY = (
    "I'm looking for a waterproof jacket for a hiking trip to Scotland next month. "
    "What would you recommend?"
)
_RECOMMEND_JACKET = "Recommend a waterproof hiking jacket."


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
    entity_ids: list[str] | None = None,
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
            "entity_ids": entity_ids or [_SUBJECT],
            "semantic_facts": [semantic_fact],
        },
    )


def _episode_observation(
    *,
    source_record_id: str,
    conversation_id: str,
    observed_at: datetime,
    content: str,
    entity_ids: list[str] | None = None,
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
            "entity_ids": entity_ids or [_SUBJECT],
        },
    )


@pytest.mark.asyncio
async def test_valid_at_equals_as_of_uses_current_snapshot_mode() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=90)
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

    inspection = await memory.inspect_recall(
        "Recommend a jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    size = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "jacket_size"
    )
    assert size.diagnostics is not None
    assert size.diagnostics.temporal_mode == TemporalRetrievalMode.CURRENT.value


def test_demo_query_does_not_trigger_historical_intent() -> None:
    mode = _temporal_retrieval_mode(
        RetrievalCue(text=_JACKET_QUERY),
        valid_at=_T,
        config=ActivationConfig(),
    )
    assert mode is not TemporalRetrievalMode.HISTORICAL


def test_historical_intent_with_same_valid_at_clock() -> None:
    mode = _temporal_retrieval_mode(
        RetrievalCue(text="What jacket size did I use before M?"),
        valid_at=_T,
        config=ActivationConfig(),
    )
    assert mode is TemporalRetrievalMode.HISTORICAL


@pytest.mark.asyncio
async def test_old_hiking_interest_current_admitted_with_hiking_in_query() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=120)
    await memory.observe(
        _semantic_observation(
            source_record_id="hike",
            conversation_id="conv-hike",
            semantic_fact={
                "predicate": "activity_interest",
                "object_value": "hiking",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer enjoys hiking in the highlands.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    results = await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    hiking = [
        result
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC
        and result.memory.predicate == "activity_interest"
        and result.memory.object_value.casefold() == "hiking"
    ]
    assert hiking
    assert hiking[0].diagnostics is not None
    assert hiking[0].diagnostics.eligibility is RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION


@pytest.mark.asyncio
async def test_colour_preference_via_jacket_evidence() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=90)
    await memory.observe(
        _semantic_observation(
            source_record_id="colour",
            conversation_id="conv-colour",
            semantic_fact={
                "predicate": "colour_preference",
                "object_value": "neutral",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content=(
                "Customer prefers black, navy and grey waterproof jacket colours for outdoor wear."
            ),
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    results = await memory.recall(
        _RECOMMEND_JACKET,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    predicates = {
        result.memory.predicate for result in results if result.memory_kind is MemoryKind.SEMANTIC
    }
    assert "colour_preference" in predicates


@pytest.mark.asyncio
async def test_northpeak_fit_via_entity_associative_reach() -> None:
    memory = _memory()
    t_shell = _T - timedelta(days=60)
    t_return = _T - timedelta(days=30)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-browse",
            conversation_id="conv-shell",
            observed_at=t_shell,
            content="Customer compared waterproof hiking shell jacket options online.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_shell)
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
            observed_at=t_return,
            content="Return processed for northpeak-alpine-shell due to sleeve length.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_return)

    results = await memory.recall(
        _RECOMMEND_JACKET,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit = [
        result
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC
        and result.memory.predicate == "product_fit_issue"
    ]
    assert fit
    diagnostics = fit[0].diagnostics
    assert diagnostics is not None
    assert diagnostics.associative_fit > 0.0


@pytest.mark.asyncio
async def test_unrelated_shoe_fit_not_admitted() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=30)
    await memory.observe(
        _semantic_observation(
            source_record_id="shoe-fit",
            conversation_id="conv-shoe",
            semantic_fact={
                "predicate": "product_fit_issue",
                "object_value": "city-shoe:heel_too_loose",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer returned city-shoe because the heel was too loose.",
            entity_ids=[_SUBJECT, "city-shoe"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    inspection = await memory.inspect_recall(
        _RECOMMEND_JACKET,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    shoe = next(
        item
        for item in inspection.rejected
        if item.memory_kind is MemoryKind.SEMANTIC and "city-shoe" in item.memory.object_value
    )
    assert shoe.disposition in {
        RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
        RecallInspectionDisposition.BELOW_THRESHOLD,
    }


@pytest.mark.asyncio
async def test_hiking_query_does_not_admit_skiing() -> None:
    memory = _memory()
    t1 = _T - timedelta(days=120)
    t2 = _T - timedelta(days=100)
    for source, value, content, observed_at in (
        ("hike", "hiking", "Customer enjoys hiking.", t1),
        ("ski", "skiing", "Customer enjoys skiing.", t2),
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
                observed_at=observed_at,
                content=content,
            )
        )
        await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)

    results = await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    recalled = {
        result.memory.object_value.casefold()
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC
        and result.memory.predicate == "activity_interest"
    }
    assert "hiking" in recalled
    assert "skiing" not in recalled


@pytest.mark.asyncio
async def test_unrelated_payment_guardrail_with_valid_at() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=30)
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
        as_of=_T,
        valid_at=_T,
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
