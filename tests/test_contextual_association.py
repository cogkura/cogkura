"""Regression tests for 0.15.7 contextual association."""

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
_JACKET_QUERY = "Recommend a waterproof hiking jacket."
_DEMO_QUERY = (
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
    semantic_facts: list[dict] | None = None,
) -> ObservationInput:
    metadata: dict[str, object] = {
        "conversation_id": conversation_id,
        "entity_ids": entity_ids or [_SUBJECT],
    }
    if semantic_facts is not None:
        metadata["semantic_facts"] = semantic_facts
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        actor_id=_SUBJECT,
        source_namespace="commerce.events",
        source_record_id=source_record_id,
        event_type="browse",
        content=content,
        observed_at=observed_at,
        metadata=metadata,
    )


async def _load_demo_shaped_northpeak(memory: Memory) -> None:
    """Compare episode names NorthPeak in text but has no product entity_ids."""
    t_compare = _T - timedelta(days=60)
    t_return = _T - timedelta(days=30)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-compare",
            conversation_id="conv-compare",
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
            conversation_id="conv-return",
            semantic_fact={
                "predicate": "product_fit_issue",
                "object_value": "northpeak-alpine-shell:sleeves_too_short",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=t_return,
            content=(
                "Returned NorthPeak Alpine Shell because sleeves were too short "
                "when reaching overhead."
            ),
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_return)


@pytest.mark.asyncio
async def test_demo_shaped_northpeak_gains_contextual_association() -> None:
    memory = _memory()
    await _load_demo_shaped_northpeak(memory)
    inspection = await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit = next(
        (
            item
            for item in (*inspection.returned, *inspection.rejected)
            if item.memory_kind is MemoryKind.SEMANTIC
            and item.memory.predicate == "product_fit_issue"
            and "northpeak" in item.memory.object_value
        ),
        None,
    )
    assert fit is not None
    assert fit.diagnostics is not None
    assert fit.diagnostics.associative_fit > 0.0
    assert fit.diagnostics.association_path is not None
    assert fit.diagnostics.association_path.bridge_entity_id == "northpeak-alpine-shell"
    assert fit.disposition is RecallInspectionDisposition.RETURNED


@pytest.mark.asyncio
async def test_direct_entity_hop_when_seed_has_product_entity() -> None:
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
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    fit = next(
        r
        for r in results
        if r.memory_kind is MemoryKind.SEMANTIC and r.memory.predicate == "product_fit_issue"
    )
    assert fit.diagnostics is not None
    assert fit.diagnostics.associative_fit > 0.0
    assert fit.diagnostics.association_path is not None
    assert fit.diagnostics.association_path.hop_kind == "entity"


@pytest.mark.asyncio
async def test_episode_to_episode_entity_bridge() -> None:
    memory = _memory()
    t_compare = _T - timedelta(days=60)
    t_browse = _T - timedelta(days=45)
    await memory.observe(
        _episode_observation(
            source_record_id="context-compare",
            conversation_id="conv-compare",
            observed_at=t_compare,
            content="Customer compared waterproof hiking jacket options online.",
            entity_ids=[_SUBJECT, "compare-session-7"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_compare)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-browse",
            conversation_id="conv-browse",
            observed_at=t_browse,
            content="Reviewed northpeak alpine shell fit details.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell", "compare-session-7"],
            semantic_facts=[
                {
                    "predicate": "product_fit_issue",
                    "object_value": "northpeak-alpine-shell:sleeves_too_short",
                    "cardinality": "many",
                    "polarity": "affirm",
                    "qualifiers": {},
                }
            ],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_browse)

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
        for item in inspection.returned
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "product_fit_issue"
    )
    path = fit.diagnostics.association_path if fit.diagnostics else None
    assert path is not None
    assert path.hop_count == 2
    assert path.related_episode_id is not None
    assert path.bridge_entity_id == "compare-session-7"


@pytest.mark.asyncio
async def test_below_threshold_seed_can_bridge_semantic() -> None:
    memory = _memory()
    await _load_demo_shaped_northpeak(memory)
    inspection = await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    seeds = [
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.EPISODE
        and "compared waterproof shell" in item.memory.statement.lower()
    ]
    assert seeds
    seed = seeds[0]
    assert seed.disposition is RecallInspectionDisposition.BELOW_THRESHOLD
    fit_returned = any(
        item.memory_kind is MemoryKind.SEMANTIC
        and item.memory.predicate == "product_fit_issue"
        and item.disposition is RecallInspectionDisposition.RETURNED
        for item in inspection.returned
    )
    assert fit_returned


@pytest.mark.asyncio
async def test_unrelated_city_shoe_not_admitted() -> None:
    memory = _memory()
    await _load_demo_shaped_northpeak(memory)
    evidence_time = _T - timedelta(days=20)
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
        _JACKET_QUERY,
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
async def test_shared_subject_only_does_not_flood() -> None:
    memory = _memory()
    t1 = _T - timedelta(days=90)
    t2 = _T - timedelta(days=60)
    for source, content, observed_at in (
        ("payment", "Customer prefers card payments.", t1),
        ("colour", "Customer prefers navy for outerwear.", t2),
    ):
        await memory.observe(
            _semantic_observation(
                source_record_id=source,
                conversation_id=f"conv-{source}",
                semantic_fact={
                    "predicate": "preferred_payment_method"
                    if source == "payment"
                    else "colour_preference",
                    "object_value": "card" if source == "payment" else "navy",
                    "cardinality": "many" if source == "colour" else "one",
                    "polarity": "affirm",
                    "qualifiers": {},
                },
                observed_at=observed_at,
                content=content,
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
    admitted_predicates = {
        item.memory.predicate
        for item in inspection.returned
        if item.memory_kind is MemoryKind.SEMANTIC
    }
    assert "preferred_payment_method" not in admitted_predicates


@pytest.mark.asyncio
async def test_single_jacket_token_does_not_admit_unrelated_fit() -> None:
    memory = _memory()
    evidence_time = _T - timedelta(days=30)
    await memory.observe(
        _semantic_observation(
            source_record_id="formal-fit",
            conversation_id="conv-formal",
            semantic_fact={
                "predicate": "product_fit_issue",
                "object_value": "formalwear:button_issue",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=evidence_time,
            content="Formal blazer button came loose.",
            entity_ids=[_SUBJECT, "formal-blazer-pro"],
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
    formal = next(
        item
        for item in inspection.rejected
        if item.memory_kind is MemoryKind.SEMANTIC and "formalwear" in item.memory.object_value
    )
    assert formal.disposition in {
        RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
        RecallInspectionDisposition.BELOW_THRESHOLD,
    }


@pytest.mark.asyncio
async def test_unknown_query_does_not_flood() -> None:
    memory = _memory()
    await _load_demo_shaped_northpeak(memory)
    inspection = await memory.inspect_recall(
        "What is the weather in London?",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    semantics = [item for item in inspection.returned if item.memory_kind is MemoryKind.SEMANTIC]
    assert not semantics


@pytest.mark.asyncio
async def test_association_paths_are_deterministic() -> None:
    memory = _memory()
    await _load_demo_shaped_northpeak(memory)
    first = await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    second = await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    first_paths = tuple(
        item.diagnostics.association_path
        for item in (*first.returned, *first.rejected)
        if item.diagnostics and item.diagnostics.association_path is not None
    )
    second_paths = tuple(
        item.diagnostics.association_path
        for item in (*second.returned, *second.rejected)
        if item.diagnostics and item.diagnostics.association_path is not None
    )
    assert first_paths == second_paths


@pytest.mark.asyncio
async def test_lightweight_negative_without_product_relationship() -> None:
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
        _JACKET_QUERY,
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
    assert lightweight.diagnostics is not None
    assert lightweight.diagnostics.associative_fit == 0.0


@pytest.mark.asyncio
async def test_distinctive_seed_overlap_beats_generic_jacket_only_seed() -> None:
    memory = _memory()
    t_generic = _T - timedelta(days=90)
    t_distinctive = _T - timedelta(days=60)
    t_return = _T - timedelta(days=30)
    await memory.observe(
        _episode_observation(
            source_record_id="generic-jacket",
            conversation_id="conv-generic",
            observed_at=t_generic,
            content="Customer mentioned jackets briefly.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_generic)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-compare",
            conversation_id="conv-compare",
            observed_at=t_distinctive,
            content=(
                "Customer compared waterproof shell jackets online, "
                "including the NorthPeak Alpine Shell."
            ),
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_distinctive)
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
            content="Returned NorthPeak Alpine Shell because sleeves were too short.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_return)

    inspection = await memory.inspect_recall(
        _DEMO_QUERY,
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
    path = fit.diagnostics.association_path if fit.diagnostics else None
    assert path is not None
    assert len(path.matched_features) >= 2


@pytest.mark.asyncio
async def test_association_attenuation_direct_entity_beats_contextual() -> None:
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

    direct = await memory.inspect_recall(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    direct_fit = next(
        item
        for item in direct.returned
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "product_fit_issue"
    )
    await _load_demo_shaped_northpeak(memory)
    contextual = await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    contextual_fit = next(
        item
        for item in (*contextual.returned, *contextual.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC
        and item.memory.predicate == "product_fit_issue"
        and "northpeak" in item.memory.object_value
    )
    assert direct_fit.diagnostics is not None
    assert contextual_fit.diagnostics is not None
    assert direct_fit.diagnostics.associative_fit > contextual_fit.diagnostics.associative_fit


@pytest.mark.asyncio
async def test_bounded_association_expansion_with_many_related_memories() -> None:
    memory = _memory()
    t_compare = _T - timedelta(days=90)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-compare",
            conversation_id="conv-compare",
            observed_at=t_compare,
            content=(
                "Customer compared waterproof shell jackets online, "
                "including the NorthPeak Alpine Shell."
            ),
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_compare)
    for index in range(12):
        observed_at = _T - timedelta(days=80 - index)
        await memory.observe(
            _episode_observation(
                source_record_id=f"noise-{index}",
                conversation_id=f"conv-noise-{index}",
                observed_at=observed_at,
                content=f"Customer browsed unrelated item {index} online.",
                entity_ids=[_SUBJECT, f"noise-entity-{index}"],
            )
        )
        await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=observed_at)
    t_return = _T - timedelta(days=30)
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
            content="Returned NorthPeak Alpine Shell because sleeves were too short.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_return)

    inspection = await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=20,
    )
    assert inspection.association_seed_count <= 5
    assert inspection.association_paths_used <= 5
    fit = next(
        item
        for item in inspection.returned
        if item.memory_kind is MemoryKind.SEMANTIC and item.memory.predicate == "product_fit_issue"
    )
    assert fit.diagnostics is not None
    assert fit.diagnostics.associative_fit > 0.0


@pytest.mark.asyncio
async def test_association_does_not_mutate_semantic_memory() -> None:
    memory = _memory()
    await _load_demo_shaped_northpeak(memory)
    before = await memory.list_semantic_memories(tenant_id=_TENANT, valid_at=_T)
    target = next(memory for memory in before if memory.predicate == "product_fit_issue")
    snapshot = (
        target.support_count,
        target.confidence,
        target.status,
        tuple(target.derivations),
    )
    await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    after = await memory.list_semantic_memories(tenant_id=_TENANT, valid_at=_T)
    updated = next(memory for memory in after if memory.predicate == "product_fit_issue")
    assert (
        updated.support_count,
        updated.confidence,
        updated.status,
        tuple(updated.derivations),
    ) == snapshot


@pytest.mark.asyncio
async def test_inspect_marks_seed_and_bridge_roles() -> None:
    memory = _memory()
    await _load_demo_shaped_northpeak(memory)
    inspection = await memory.inspect_recall(
        _DEMO_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=10,
    )
    assert inspection.association_seed_count > 0
    assert inspection.association_paths_used > 0
    roles = {
        item.association_role
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.EPISODE
    }
    assert "seed" in roles
