"""Regression tests for 0.15.8 structured entity relationship recall."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    MemoryKind,
    SemanticMemoryStatus,
)

_TENANT = "shop"
_SUBJECT = "customer_42"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_JACKET_QUERY = "Recommend a waterproof hiking jacket."


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


def _relationship_observation(
    *,
    source_record_id: str,
    relationships: list[dict[str, str]],
    observed_at: datetime | None = None,
) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        actor_id=_SUBJECT,
        source_namespace="catalog.taxonomy",
        source_record_id=source_record_id,
        event_type="taxonomy",
        content="Catalog relationship import",
        observed_at=observed_at or _T - timedelta(days=90),
        metadata={"relationships": relationships},
    )


def _episode_observation(
    *,
    source_record_id: str,
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
            "conversation_id": source_record_id,
            "entity_ids": entity_ids or [_SUBJECT],
        },
    )


def _semantic_observation(
    *,
    source_record_id: str,
    observed_at: datetime,
    semantic_fact: dict,
    entity_ids: list[str] | None = None,
    content: str = "Semantic memory",
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
            "conversation_id": source_record_id,
            "entity_ids": entity_ids or [_SUBJECT],
            "semantic_facts": [semantic_fact],
        },
    )


async def _load_northpeak_fixture(
    memory: Memory,
    *,
    with_graph: bool,
) -> None:
    if with_graph:
        await memory.observe(
            _relationship_observation(
                source_record_id="taxonomy-shells",
                relationships=[
                    {
                        "source_entity_id": "northpeak-alpine-shell",
                        "relation_type": "is_a",
                        "target_entity_id": "waterproof-shell",
                        "provenance": "catalog-import",
                    },
                    {
                        "source_entity_id": "waterproof-shell",
                        "relation_type": "is_a",
                        "target_entity_id": "jacket",
                        "provenance": "catalog-import",
                    },
                ],
            )
        )
    t_compare = _T - timedelta(days=60)
    t_return = _T - timedelta(days=30)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-compare",
            observed_at=t_compare,
            content=(
                "Customer compared waterproof shell jackets online, "
                "including the NorthPeak Alpine Shell."
            ),
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_compare)
    await memory.observe(
        _semantic_observation(
            source_record_id="return",
            observed_at=t_return,
            semantic_fact={
                "predicate": "product_fit_issue",
                "object_value": "northpeak-alpine-shell:sleeves_too_short",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            content="Returned NorthPeak Alpine Shell because sleeves were too short.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_return)


@pytest.mark.asyncio
async def test_northpeak_reachable_with_graph_unreachable_without() -> None:
    with_graph = _memory()
    await _load_northpeak_fixture(with_graph, with_graph=True)
    with_inspection = await with_graph.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    assert with_inspection.relationship_seed_count >= 1
    assert with_inspection.relationship_paths_used >= 1
    fit = next(
        item
        for item in (*with_inspection.returned, *with_inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "product_fit_issue"
    )
    assert fit.diagnostics is not None
    assert fit.diagnostics.structured_association_fit > 0.0
    assert fit.diagnostics.association_path is not None
    assert fit.diagnostics.association_path.seed_entity_id == "jacket"
    assert fit.diagnostics.association_path.hop_kind == "relationship"
    assert len(fit.diagnostics.association_path.relationship_edges) == 2

    without_graph = _memory()
    await _load_northpeak_fixture(without_graph, with_graph=False)
    without_inspection = await without_graph.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    assert without_inspection.relationship_paths_used == 0
    unreachable = next(
        item
        for item in (*without_inspection.returned, *without_inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "product_fit_issue"
    )
    assert unreachable.diagnostics is not None
    assert unreachable.diagnostics.structured_association_fit == 0.0


@pytest.mark.asyncio
async def test_two_hop_reverse_reach_and_hop_limit() -> None:
    memory = _memory()
    await memory.observe(
        _relationship_observation(
            source_record_id="taxonomy-chain",
            relationships=[
                {
                    "source_entity_id": "product-a",
                    "relation_type": "is_a",
                    "target_entity_id": "product-b",
                },
                {
                    "source_entity_id": "product-b",
                    "relation_type": "is_a",
                    "target_entity_id": "product-c",
                },
                {
                    "source_entity_id": "product-d",
                    "relation_type": "is_a",
                    "target_entity_id": "product-e",
                },
                {
                    "source_entity_id": "product-e",
                    "relation_type": "is_a",
                    "target_entity_id": "product-f",
                },
                {
                    "source_entity_id": "product-f",
                    "relation_type": "is_a",
                    "target_entity_id": "product-g",
                },
            ],
        )
    )
    observed_at = _T - timedelta(days=10)
    await memory.observe(
        _semantic_observation(
            source_record_id="fact-a",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "stock_level",
                "object_value": "low",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "product-a"],
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="fact-d",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "stock_level",
                "object_value": "high",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "product-d"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)

    inspection = await memory.inspect_recall(
        "product c availability",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    reachable = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.object_value == "low"
    )
    assert reachable.diagnostics is not None
    assert reachable.diagnostics.structured_association_fit > 0.0

    unreachable = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.object_value == "high"
    )
    assert unreachable.diagnostics is not None
    assert unreachable.diagnostics.structured_association_fit == 0.0


@pytest.mark.asyncio
async def test_cycle_terminates() -> None:
    memory = _memory()
    await memory.observe(
        _relationship_observation(
            source_record_id="taxonomy-cycle",
            relationships=[
                {
                    "source_entity_id": "entity-a",
                    "relation_type": "related_to",
                    "target_entity_id": "entity-b",
                },
                {
                    "source_entity_id": "entity-b",
                    "relation_type": "related_to",
                    "target_entity_id": "entity-a",
                },
            ],
        )
    )
    observed_at = _T - timedelta(days=5)
    await memory.observe(
        _semantic_observation(
            source_record_id="fact-b",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "availability",
                "object_value": "in_stock",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "entity-b"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)
    inspection = await memory.inspect_recall(
        "entity a recommendation",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC
    )
    assert fit.diagnostics is not None
    assert fit.diagnostics.structured_association_fit > 0.0


@pytest.mark.asyncio
async def test_is_a_beats_related_to_for_same_target() -> None:
    memory = _memory()
    await memory.observe(
        _relationship_observation(
            source_record_id="taxonomy-weights",
            relationships=[
                {
                    "source_entity_id": "strong-product",
                    "relation_type": "is_a",
                    "target_entity_id": "jacket",
                },
                {
                    "source_entity_id": "weak-product",
                    "relation_type": "related_to",
                    "target_entity_id": "jacket",
                },
            ],
        )
    )
    observed_at = _T - timedelta(days=5)
    await memory.observe(
        _semantic_observation(
            source_record_id="strong-fact",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "preferred_color",
                "object_value": "blue",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "strong-product"],
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="weak-fact",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "preferred_color",
                "object_value": "red",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "weak-product"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)
    inspection = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    blue = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.object_value == "blue"
    )
    red = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.object_value == "red"
    )
    assert blue.diagnostics is not None and red.diagnostics is not None
    assert blue.diagnostics.structured_association_fit > red.diagnostics.structured_association_fit


@pytest.mark.asyncio
async def test_lightweight_shell_reachable_from_hiking_jacket_query() -> None:
    memory = _memory()
    await memory.observe(
        _relationship_observation(
            source_record_id="taxonomy-light",
            relationships=[
                {
                    "source_entity_id": "featherlite-packable-shell",
                    "relation_type": "is_a",
                    "target_entity_id": "jacket",
                }
            ],
        )
    )
    observed_at = _T - timedelta(days=5)
    await memory.observe(
        _semantic_observation(
            source_record_id="light-pref",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "outerwear_weight_preference",
                "object_value": "lightweight",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "featherlite-packable-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)
    inspection = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC
        and item.memory.predicate == "outerwear_weight_preference"
    )
    assert fit.diagnostics is not None
    assert fit.diagnostics.structured_association_fit > 0.0


@pytest.mark.asyncio
async def test_city_shoe_isolated_from_jacket_query() -> None:
    memory = _memory()
    await memory.observe(
        _relationship_observation(
            source_record_id="taxonomy-footwear",
            relationships=[
                {
                    "source_entity_id": "city-shoe",
                    "relation_type": "is_a",
                    "target_entity_id": "footwear",
                }
            ],
        )
    )
    observed_at = _T - timedelta(days=5)
    await memory.observe(
        _semantic_observation(
            source_record_id="shoe-fact",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "style",
                "object_value": "urban",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "city-shoe"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)
    inspection = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC
    )
    assert fit.diagnostics is not None
    assert fit.diagnostics.structured_association_fit == 0.0


@pytest.mark.asyncio
async def test_superseded_semantic_blocked_from_structured_admission() -> None:
    memory = _memory()
    await memory.observe(
        _relationship_observation(
            source_record_id="taxonomy-superseded",
            relationships=[
                {
                    "source_entity_id": "northpeak-alpine-shell",
                    "relation_type": "is_a",
                    "target_entity_id": "jacket",
                }
            ],
        )
    )
    observed_at = _T - timedelta(days=5)
    await memory.observe(
        _semantic_observation(
            source_record_id="old-fact",
            observed_at=observed_at,
            semantic_fact={
                "predicate": "product_fit_issue",
                "object_value": "northpeak-alpine-shell:sleeves_too_short",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)
    semantics = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert len(semantics) == 1
    assert semantics[0].status is SemanticMemoryStatus.ACTIVE

    inspection = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    active_fit = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC
    )
    assert active_fit.diagnostics is not None
    assert active_fit.diagnostics.structured_association_fit > 0.0


@pytest.mark.asyncio
async def test_recall_does_not_mutate_relationship_graph_or_semantics() -> None:
    memory = _memory()
    await _load_northpeak_fixture(memory, with_graph=True)
    before = await memory.list_entity_relationships(tenant_id=_TENANT)
    semantics_before = await memory.list_semantic_memories(tenant_id=_TENANT)
    await memory.recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    after = await memory.list_entity_relationships(tenant_id=_TENANT)
    semantics_after = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert before == after
    assert semantics_before == semantics_after


@pytest.mark.asyncio
async def test_structured_recall_is_deterministic() -> None:
    memory = _memory()
    await _load_northpeak_fixture(memory, with_graph=True)
    first = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    second = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    assert first.relationship_seed_count == second.relationship_seed_count
    assert first.relationship_paths_used == second.relationship_paths_used
    assert [item.memory.id for item in first.returned] == [
        item.memory.id for item in second.returned
    ]
