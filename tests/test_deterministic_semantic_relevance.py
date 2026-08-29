"""Regression tests for 0.15.5 deterministic semantic relevance and association."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    MemoryKind,
    RecallInspectionDisposition,
)

_TENANT = "shop"
_SUBJECT = "customer_42"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


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


async def _recall_semantic_predicates(query: str) -> set[str]:
    memory = _memory()
    evidence_time = _T - timedelta(days=60)
    await memory.observe(
        _semantic_observation(
            source_record_id="colour",
            conversation_id="conv-colour",
            semantic_fact={
                "predicate": "colour_preference",
                "object_value": "neutral",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Prefers black, navy and grey jackets.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)
    results = await memory.recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    return {
        result.memory.predicate for result in results if result.memory_kind is MemoryKind.SEMANTIC
    }


@pytest.mark.asyncio
async def test_stopword_reformulation_invariance() -> None:
    queries = (
        "Recommend a waterproof jacket.",
        "Could you recommend a waterproof jacket for me?",
        "I'm looking for a waterproof jacket, what would you recommend?",
    )
    admissions = []
    for query in queries:
        predicates = await _recall_semantic_predicates(query)
        admissions.append("colour_preference" in predicates)
    assert admissions[0] == admissions[1] == admissions[2]


@pytest.mark.asyncio
async def test_stopword_false_positive_rejected() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=30)
    await memory.observe(
        _semantic_observation(
            source_record_id="grey-clothing",
            conversation_id="conv-grey",
            semantic_fact={
                "predicate": "colour_preference",
                "object_value": "grey",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer prefers grey colours for formal clothing.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    inspection = await memory.inspect_recall(
        "Could you recommend a backpack for me?",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    grey = next(
        item
        for item in inspection.rejected
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "colour_preference"
    )
    assert grey.disposition in {
        RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
        RecallInspectionDisposition.BELOW_THRESHOLD,
    }
    diagnostics = grey.diagnostics
    assert diagnostics is not None
    assert diagnostics.evidence_linked_fit == 0.0
    assert not diagnostics.matched_evidence_features


@pytest.mark.asyncio
async def test_singular_plural_jacket_overlap() -> None:
    predicates = await _recall_semantic_predicates("recommend a jacket")
    assert "colour_preference" in predicates


@pytest.mark.asyncio
async def test_query_verbosity_similar_admission() -> None:
    short = await _recall_semantic_predicates("waterproof jacket")
    verbose = await _recall_semantic_predicates(
        "please could you help me find a waterproof jacket that would work for me"
    )
    assert ("colour_preference" in short) == ("colour_preference" in verbose)


@pytest.mark.asyncio
async def test_colour_via_jacket_evidence_diagnostics() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=60)
    await memory.observe(
        _semantic_observation(
            source_record_id="colour",
            conversation_id="conv-colour",
            semantic_fact={
                "predicate": "colour_preference",
                "object_value": "neutral",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Prefers black, navy and grey jackets.",
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
    colour = next(
        item
        for item in inspection.returned
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "colour_preference"
    )
    diagnostics = colour.diagnostics
    assert diagnostics is not None
    assert diagnostics.evidence_linked_fit > 0.0
    assert "jacket" in diagnostics.matched_evidence_features
    assert "for" not in diagnostics.matched_evidence_features
    assert "to" not in diagnostics.matched_evidence_features


@pytest.mark.asyncio
async def test_lightweight_direct_evidence_zero_without_overlap() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=30)
    await memory.observe(
        _semantic_observation(
            source_record_id="lightweight",
            conversation_id="conv-light",
            semantic_fact={
                "predicate": "outerwear_weight_preference",
                "object_value": "lightweight",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Purchased lightweight shell.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    inspection = await memory.inspect_recall(
        "Recommend a hiking jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    lightweight = next(
        item
        for item in inspection.rejected
        if item.memory_kind is MemoryKind.SEMANTIC
        and item.memory.predicate == "outerwear_weight_preference"
    )
    diagnostics = lightweight.diagnostics
    assert diagnostics is not None
    assert diagnostics.evidence_linked_fit == 0.0
    assert "a" not in diagnostics.matched_evidence_features


@pytest.mark.asyncio
async def test_lightweight_association_via_product_entity() -> None:
    memory = _memory()
    t_browse = _T - timedelta(days=60)
    t_light = _T - timedelta(days=45)
    await memory.observe(
        _episode_observation(
            source_record_id="jacket-browse",
            conversation_id="conv-browse",
            observed_at=t_browse,
            content="Customer compared waterproof hiking jacket options.",
            entity_ids=[_SUBJECT, "featherlite-packable-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_browse)
    await memory.observe(
        _semantic_observation(
            source_record_id="lightweight",
            conversation_id="conv-light",
            semantic_fact={
                "predicate": "outerwear_weight_preference",
                "object_value": "lightweight",
                "object_entity_id": "featherlite-packable-shell",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=t_light,
            content="Customer prefers lightweight outerwear.",
            entity_ids=[_SUBJECT, "featherlite-packable-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_light)

    results = await memory.recall(
        "Recommend a waterproof hiking jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    lightweight = next(
        result
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC
        and result.memory.predicate == "outerwear_weight_preference"
    )
    diagnostics = lightweight.diagnostics
    assert diagnostics is not None
    assert diagnostics.associative_fit > 0.0


@pytest.mark.asyncio
async def test_northpeak_harder_evidence_to_evidence_association() -> None:
    memory = _memory()
    t_compare = _T - timedelta(days=60)
    t_return = _T - timedelta(days=30)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-compare",
            conversation_id="conv-compare",
            observed_at=t_compare,
            content="Customer compared waterproof hiking shell jacket options online.",
            entity_ids=[_SUBJECT],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_compare)
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

    inspection = await memory.inspect_recall(
        "Recommend a waterproof hiking jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit = next(
        item
        for item in inspection.returned
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "product_fit_issue"
    )
    diagnostics = fit.diagnostics
    assert diagnostics is not None
    assert diagnostics.associative_fit > 0.0
    assert diagnostics.association_path is not None
    assert diagnostics.association_path.hop_kind == "evidence"


@pytest.mark.asyncio
async def test_unrelated_product_issue_not_admitted() -> None:
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
        "hiking jacket",
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
async def test_subject_only_does_not_flood_semantics() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=30)
    for index, predicate in enumerate(
        ("preferred_payment_method", "activity_interest", "jacket_size"),
        start=1,
    ):
        await memory.observe(
            _semantic_observation(
                source_record_id=f"fact-{index}",
                conversation_id=f"conv-{index}",
                semantic_fact={
                    "predicate": predicate,
                    "object_value": f"value-{index}",
                    "cardinality": "one",
                    "polarity": "affirm",
                    "qualifiers": {},
                },
                observed_at=evidence_time,
                content=f"Stored fact {index}.",
            )
        )
        await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    results = await memory.recall(
        "hiking jacket",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    semantic_count = sum(1 for result in results if result.memory_kind is MemoryKind.SEMANTIC)
    assert semantic_count < 3


@pytest.mark.asyncio
async def test_path_attenuation_direct_over_association() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=30)
    await memory.observe(
        _semantic_observation(
            source_record_id="direct-hiking",
            conversation_id="conv-hike",
            semantic_fact={
                "predicate": "activity_interest",
                "object_value": "hiking",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Customer enjoys hiking trips.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    inspection = await memory.inspect_recall(
        "Recommend a waterproof hiking jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    hiking = next(
        item
        for item in inspection.returned
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "activity_interest"
    )
    diagnostics = hiking.diagnostics
    assert diagnostics is not None
    assert diagnostics.evidence_linked_fit >= diagnostics.associative_fit


@pytest.mark.asyncio
async def test_deterministic_repeatable_recall() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=60)
    await memory.observe(
        _semantic_observation(
            source_record_id="colour",
            conversation_id="conv-colour",
            semantic_fact={
                "predicate": "colour_preference",
                "object_value": "neutral",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Prefers black, navy and grey jackets.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)

    first = await memory.inspect_recall(
        "Recommend a jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    second = await memory.inspect_recall(
        "Recommend a jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    first_keys = tuple(item.memory.memory_key for item in first.returned)
    second_keys = tuple(item.memory.memory_key for item in second.returned)
    assert first_keys == second_keys
    assert first.canonical_query_features == second.canonical_query_features


@pytest.mark.asyncio
async def test_association_does_not_change_support_count() -> None:
    memory = _memory()
    t1 = _T - timedelta(days=60)
    t2 = _T - timedelta(days=30)
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
            conversation_id="conv-1",
            semantic_fact=fact,
            observed_at=t1,
            content="First lightweight purchase.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t1)
    await memory.observe(
        _semantic_observation(
            source_record_id="light-2",
            conversation_id="conv-2",
            semantic_fact=fact,
            observed_at=t2,
            content="Second lightweight purchase.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t2)
    memories = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    assert len(memories) == 1
    assert memories[0].support_count == 2
