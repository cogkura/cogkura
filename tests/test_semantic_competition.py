"""Regression tests for 0.15.6 semantic competition and recall specificity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    MemoryKind,
    RecallInspectionDisposition,
    RelevanceTier,
)

_TENANT = "shop"
_SUBJECT = "customer_42"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_HIKING_QUERY = "Recommend a waterproof hiking jacket."
_SKIING_QUERY = "Recommend a waterproof skiing jacket."
_BOTH_QUERY = "Recommend hiking and skiing equipment."


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


async def _load_hiking_skiing_fixture(memory: Memory) -> None:
    """Hiking interest plus skiing with recent ski-jacket evidence (Demo-shaped)."""
    t_hike = _T - timedelta(days=120)
    t_ski_ep = _T - timedelta(days=30)
    t_ski_sem = _T - timedelta(days=25)
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
            observed_at=t_hike,
            content="Customer enjoys hiking in the highlands.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_hike)
    await memory.observe(
        _episode_observation(
            source_record_id="ski-jacket",
            conversation_id="conv-ski",
            observed_at=t_ski_ep,
            content="Customer browsed waterproof ski jackets online.",
            entity_ids=[_SUBJECT, "ski-jacket-pro"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_ski_ep)
    await memory.observe(
        _semantic_observation(
            source_record_id="ski",
            conversation_id="conv-ski-sem",
            semantic_fact={
                "predicate": "activity_interest",
                "object_value": "skiing",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=t_ski_sem,
            content="Customer enjoys skiing and ski jackets.",
            entity_ids=[_SUBJECT, "ski-jacket-pro"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_ski_sem)


def _activity_interest_items(inspection) -> dict[str, object]:
    items = {}
    for item in (*inspection.returned, *inspection.rejected):
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "activity_interest":
            items[item.memory.object_value.casefold()] = item
    return items


@pytest.mark.asyncio
async def test_hiking_and_skiing_remain_distinct_under_collapse() -> None:
    memory = _memory()
    await _load_hiking_skiing_fixture(memory)
    inspection = await memory.inspect_recall(
        _HIKING_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    items = _activity_interest_items(inspection)
    assert "hiking" in items
    assert "skiing" in items
    assert items["hiking"].disposition is not RecallInspectionDisposition.COLLAPSED
    assert items["skiing"].disposition is not RecallInspectionDisposition.COLLAPSED


@pytest.mark.asyncio
async def test_hiking_query_ranks_hiking_above_skiing() -> None:
    memory = _memory()
    await _load_hiking_skiing_fixture(memory)
    inspection = await memory.inspect_recall(
        _HIKING_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    items = _activity_interest_items(inspection)
    hiking = items["hiking"]
    skiing = items["skiing"]
    assert hiking.diagnostics is not None
    assert skiing.diagnostics is not None
    assert hiking.diagnostics.direct_value_fit > 0.0
    assert skiing.diagnostics.direct_value_fit == 0.0
    assert skiing.diagnostics.evidence_linked_fit > 0.0
    assert hiking.diagnostics.relevance_tier == RelevanceTier.DIRECT_VALUE.value
    assert skiing.diagnostics.relevance_tier == RelevanceTier.EVIDENCE_ASSOCIATION.value

    results = await memory.recall(
        _HIKING_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    activity = [
        result
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC
        and result.memory.predicate == "activity_interest"
    ]
    assert len(activity) >= 2
    assert activity[0].memory.object_value.casefold() == "hiking"


@pytest.mark.asyncio
async def test_skiing_query_ranks_skiing_above_hiking() -> None:
    memory = _memory()
    await _load_hiking_skiing_fixture(memory)
    results = await memory.recall(
        _SKIING_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    activity = [
        result
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC
        and result.memory.predicate == "activity_interest"
    ]
    assert activity
    assert activity[0].memory.object_value.casefold() == "skiing"


@pytest.mark.asyncio
async def test_explicit_both_activities_survive() -> None:
    memory = _memory()
    await _load_hiking_skiing_fixture(memory)
    results = await memory.recall(
        _BOTH_QUERY,
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
    assert "skiing" in recalled


@pytest.mark.asyncio
async def test_many_valued_product_fit_issues_remain_distinct() -> None:
    memory = _memory()
    t1 = _T - timedelta(days=60)
    t2 = _T - timedelta(days=30)
    for source, value, content in (
        ("shell-fit", "northpeak-alpine-shell:sleeves_too_short", "Sleeves too short."),
        ("shoe-fit", "city-shoe:heel_too_loose", "Heel too loose."),
    ):
        await memory.observe(
            _semantic_observation(
                source_record_id=source,
                conversation_id=f"conv-{source}",
                semantic_fact={
                    "predicate": "product_fit_issue",
                    "object_value": value,
                    "cardinality": "many",
                    "polarity": "affirm",
                    "qualifiers": {},
                },
                observed_at=t1 if "shell" in source else t2,
                content=content,
                entity_ids=[_SUBJECT, value.split(":")[0]],
            )
        )
        await memory.process(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            as_of=t1 if "shell" in source else t2,
        )

    inspection = await memory.inspect_recall(
        _HIKING_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit_items = [
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "product_fit_issue"
    ]
    assert len(fit_items) == 2
    dispositions = {item.disposition for item in fit_items}
    assert RecallInspectionDisposition.COLLAPSED not in dispositions


@pytest.mark.asyncio
async def test_cardinality_one_jacket_size_keeps_authoritative_value() -> None:
    memory = _memory()
    t_l = _T - timedelta(days=120)
    t_m = _T - timedelta(days=90)
    for source, value, observed_at in (
        ("size-l", "L", t_l),
        ("size-m", "M", t_m),
    ):
        await memory.observe(
            _semantic_observation(
                source_record_id=source,
                conversation_id=f"conv-{source}",
                semantic_fact={
                    "predicate": "jacket_size",
                    "object_value": value,
                    "cardinality": "one",
                    "polarity": "affirm",
                    "qualifiers": {},
                },
                observed_at=observed_at,
                content=f"Customer jacket size is {value}.",
            )
        )
        await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)

    results = await memory.recall(
        "Recommend a jacket.",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    sizes = [
        result.memory.object_value.casefold()
        for result in results
        if result.memory_kind is MemoryKind.SEMANTIC and result.memory.predicate == "jacket_size"
    ]
    assert sizes == ["m"]


@pytest.mark.asyncio
async def test_unknown_query_does_not_flood_same_predicate() -> None:
    memory = _memory()
    await _load_hiking_skiing_fixture(memory)
    inspection = await memory.inspect_recall(
        "What is the weather in London?",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    admitted = [
        item
        for item in inspection.returned
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "activity_interest"
    ]
    assert not admitted


@pytest.mark.asyncio
async def test_competition_ranking_is_deterministic() -> None:
    memory = _memory()
    await _load_hiking_skiing_fixture(memory)
    first = await memory.inspect_recall(
        _HIKING_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    second = await memory.inspect_recall(
        _HIKING_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    first_keys = tuple(item.memory.memory_key for item in first.returned)
    second_keys = tuple(item.memory.memory_key for item in second.returned)
    assert first_keys == second_keys


@pytest.mark.asyncio
async def test_colour_preference_still_admits_on_jacket_query() -> None:
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
        _HIKING_QUERY,
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
