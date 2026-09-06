"""Investigation-only tests for cardinality-many semantic currentness (0.15.10).

These tests document stored state, admission, forgetting, and working-memory
selection. They do not assert that a weak old interest must be dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    ActivationConfig,
    MemoryIdentity,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RetrievalEligibility,
    SemanticCardinality,
    SemanticMemoryStatus,
    StoredSemanticMemory,
    WorkingMemoryConfig,
)

_TENANT = "lab"
_SUBJECT = "customer-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_T_WEAK = _T - timedelta(days=150)
_T_STRONG = _T - timedelta(days=20)
_QUERY = "Recommend outerwear for the customer."
_WEAK_OBJECT = "activity-a"
_STRONG_OBJECT = "activity-b"
_WEAK_PRODUCT = "activity-a-product"
_CATEGORY = "outerwear"


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


def _interest_fact(object_value: str, **extra: Any) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "predicate": "activity_interest",
        "object_value": object_value,
        "cardinality": "many",
        "polarity": "affirm",
        "qualifiers": {},
    }
    fact.update(extra)
    return fact


def _observe_semantic(
    memory: Memory,
    *,
    source_record_id: str,
    observed_at: datetime,
    content: str,
    semantic_fact: dict[str, Any],
    entity_ids: list[str] | None = None,
    event_type: str = "message",
) -> Any:
    return memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id=source_record_id,
            event_type=event_type,
            content=content,
            observed_at=observed_at,
            metadata={
                "conversation_id": source_record_id,
                "entity_ids": entity_ids or [_SUBJECT],
                "semantic_facts": [semantic_fact],
            },
        )
    )


def _observe_relationships(memory: Memory) -> Any:
    return memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="catalog.taxonomy",
            source_record_id="taxonomy-outerwear",
            event_type="taxonomy",
            content="Catalog relationship import",
            observed_at=_T - timedelta(days=180),
            metadata={
                "relationships": [
                    {
                        "source_entity_id": _WEAK_PRODUCT,
                        "relation_type": "is_a",
                        "target_entity_id": _CATEGORY,
                        "provenance": "catalog-import",
                    }
                ]
            },
        )
    )


async def _load_weak_and_strong(
    memory: Memory,
    *,
    with_graph: bool,
    extra_strong_support: bool = False,
    weak_valid_until: datetime | None = None,
) -> None:
    if with_graph:
        await _observe_relationships(memory)
        await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T_WEAK)

    weak_fact = _interest_fact(_WEAK_OBJECT)
    if weak_valid_until is not None:
        weak_fact["valid_until"] = weak_valid_until.isoformat()
    await _observe_semantic(
        memory,
        source_record_id="weak-browse",
        observed_at=_T_WEAK,
        content="Customer briefly browsed activity-a-product listings.",
        semantic_fact=weak_fact,
        entity_ids=[_SUBJECT, _WEAK_PRODUCT],
        event_type="browse",
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T_WEAK)

    await _observe_semantic(
        memory,
        source_record_id="strong-purchase",
        observed_at=_T_STRONG,
        content="Customer purchased activity-b equipment after repeated use.",
        semantic_fact=_interest_fact(_STRONG_OBJECT),
        entity_ids=[_SUBJECT],
        event_type="purchase",
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T_STRONG)
    if extra_strong_support:
        await _observe_semantic(
            memory,
            source_record_id="strong-repeat",
            observed_at=_T_STRONG + timedelta(days=5),
            content="Customer confirmed continued activity-b use.",
            semantic_fact=_interest_fact(_STRONG_OBJECT),
            entity_ids=[_SUBJECT],
            event_type="message",
        )
        await memory.process(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            as_of=_T_STRONG + timedelta(days=5),
        )


def _interest(memories: list[StoredSemanticMemory], object_value: str) -> StoredSemanticMemory:
    return next(item for item in memories if item.object_value == object_value)


def _inspect_item(inspection: Any, object_value: str) -> RecallInspectionCandidate | None:
    for item in (*inspection.returned, *inspection.rejected):
        if (
            item.memory_kind is MemoryKind.SEMANTIC
            and isinstance(item.memory, StoredSemanticMemory)
            and item.memory.object_value == object_value
        ):
            return item
    return None


def _admission_gates(
    item: RecallInspectionCandidate,
    *,
    config: ActivationConfig,
) -> tuple[str, ...]:
    diagnostics = item.diagnostics
    if diagnostics is None:
        return ()
    gates: list[str] = []
    if diagnostics.semantic_relevance >= config.semantic_current_min_relevance:
        gates.append("combined_relevance")
    if len(diagnostics.matched_direct_features) >= config.lexical_slot_min_overlap:
        gates.append("lexical_direct")
    if len(diagnostics.matched_evidence_features) >= config.lexical_slot_min_overlap:
        gates.append("lexical_evidence")
    if diagnostics.associative_fit >= config.contextual_association_min_relevance:
        gates.append("associative_fit")
    if diagnostics.structured_association_fit >= config.semantic_relationship_min_relevance:
        gates.append("structured_fit")
    path = diagnostics.association_path
    if path is not None and path.hop_kind == "relationship":
        gates.append("structured_path")
    return tuple(gates)


@dataclass(frozen=True)
class SemanticLifecycleSnapshot:
    object_value: str
    stored: bool
    status: SemanticMemoryStatus | None
    cardinality: SemanticCardinality | None
    support_count: int | None
    confidence: float | None
    importance: float | None
    first_supported_at: datetime | None
    last_supported_at: datetime | None
    valid_from: datetime | None
    valid_until: datetime | None
    created_at: datetime | None
    derivation_episode_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    retrieved: bool
    current_admitted: bool
    eligibility: str | None
    admission_reason: str | None
    relevance_tier: str | None
    semantic_relevance: float | None
    structured_association_fit: float | None
    activation: float | None
    base_level: float | None
    spreading: float | None
    rank_activation: float | None
    retrieval_threshold: float | None
    structured_adjustment: float | None
    soft_admitted: bool | None
    retention_state: str | None
    last_base_level: float | None
    last_retention_score: float | None
    episode_ended_at: datetime | None
    episode_created_at: datetime | None
    learning_count: int
    contradiction_count: int | None
    relationship_paths_used: int
    admission_gates: tuple[str, ...]
    selected: bool


async def _snapshot_for(
    memory: Memory,
    *,
    object_value: str,
    query: str = _QUERY,
    with_wm: bool = True,
) -> SemanticLifecycleSnapshot:
    semantics = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        include_inactive=True,
    )
    stored = next((item for item in semantics if item.object_value == object_value), None)
    inspection = await memory.inspect_recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=20,
    )
    item = _inspect_item(inspection, object_value)
    selected = False
    if with_wm:
        snapshot = await memory.select_working_memory(
            query,
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            as_of=_T,
        )
        selected = any(
            recall.memory_kind is MemoryKind.SEMANTIC
            and isinstance(recall.memory, StoredSemanticMemory)
            and recall.memory.object_value == object_value
            for recall in snapshot.recall_results
        )
    dynamics = None
    learning_count = 0
    episode_ended_at = None
    episode_created_at = None
    if stored is not None:
        identity = MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key=stored.memory_key)
        dynamics_map = await memory._dynamics_store.get_many(
            tenant_id=_TENANT,
            identities=[identity],
        )
        dynamics = dynamics_map.get(identity)
        learning_states = await memory.list_learning_state(
            tenant_id=_TENANT,
            identities=[identity],
        )
        learning_count = sum(
            state.helpful_count + state.unhelpful_count + state.incorrect_count
            for state in learning_states
        )
        derivation_ids = {d.episode_id for d in stored.derivations}
        episodes = await memory.list_episodes(tenant_id=_TENANT, subject_id=_SUBJECT)
        support = next(
            (
                episode
                for episode in episodes
                if episode.id in derivation_ids or episode.memory_key in derivation_ids
            ),
            None,
        )
        if support is not None:
            episode_ended_at = support.ended_at
            episode_created_at = support.created_at
    diagnostics = item.diagnostics if item is not None else None
    retrieved = item is not None and item.disposition is RecallInspectionDisposition.RETURNED
    current_admitted = (
        item is not None
        and diagnostics is not None
        and diagnostics.eligibility is RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION
    )
    return SemanticLifecycleSnapshot(
        object_value=object_value,
        stored=stored is not None,
        status=stored.status if stored is not None else None,
        cardinality=stored.cardinality if stored is not None else None,
        support_count=stored.support_count if stored is not None else None,
        confidence=stored.confidence if stored is not None else None,
        importance=stored.importance if stored is not None else None,
        first_supported_at=stored.first_supported_at if stored is not None else None,
        last_supported_at=stored.last_supported_at if stored is not None else None,
        valid_from=stored.valid_from if stored is not None else None,
        valid_until=stored.valid_until if stored is not None else None,
        created_at=stored.created_at if stored is not None else None,
        derivation_episode_ids=tuple(d.episode_id for d in stored.derivations)
        if stored is not None
        else (),
        observation_ids=tuple(e.observation_id for e in stored.observation_evidence)
        if stored is not None
        else (),
        retrieved=retrieved,
        current_admitted=current_admitted,
        eligibility=diagnostics.eligibility.value if diagnostics is not None else None,
        admission_reason=diagnostics.admission_reason if diagnostics is not None else None,
        relevance_tier=diagnostics.relevance_tier if diagnostics is not None else None,
        semantic_relevance=diagnostics.semantic_relevance if diagnostics is not None else None,
        structured_association_fit=diagnostics.structured_association_fit
        if diagnostics is not None
        else None,
        activation=item.activation if item is not None else None,
        base_level=diagnostics.base_level if diagnostics is not None else None,
        spreading=diagnostics.spreading if diagnostics is not None else None,
        rank_activation=diagnostics.rank_activation if diagnostics is not None else None,
        retrieval_threshold=item.retrieval_threshold if item is not None else None,
        structured_adjustment=diagnostics.structured_adjustment
        if diagnostics is not None
        else None,
        soft_admitted=item.soft_admitted if item is not None else None,
        retention_state=dynamics.retention_state.value if dynamics is not None else None,
        last_base_level=dynamics.last_base_level if dynamics is not None else None,
        last_retention_score=dynamics.last_retention_score if dynamics is not None else None,
        episode_ended_at=episode_ended_at,
        episode_created_at=episode_created_at,
        learning_count=learning_count,
        contradiction_count=stored.contradiction_count if stored is not None else None,
        relationship_paths_used=inspection.relationship_paths_used,
        admission_gates=_admission_gates(item, config=ActivationConfig())
        if item is not None
        else (),
        selected=selected,
    )


@pytest.mark.asyncio
async def test_weak_and_strong_interests_both_remain_active() -> None:
    memory = _memory()
    await _load_weak_and_strong(memory, with_graph=True, extra_strong_support=True)
    semantics = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    weak = _interest(semantics, _WEAK_OBJECT)
    strong = _interest(semantics, _STRONG_OBJECT)
    assert weak.status is SemanticMemoryStatus.ACTIVE
    assert strong.status is SemanticMemoryStatus.ACTIVE
    assert weak.cardinality is SemanticCardinality.MANY
    assert strong.cardinality is SemanticCardinality.MANY
    assert weak.support_count == 1
    assert strong.support_count >= 2
    assert weak.last_supported_at < strong.last_supported_at
    assert weak.created_at is not None
    assert weak.first_supported_at == _T_WEAK


@pytest.mark.asyncio
async def test_weak_old_structured_interest_is_current_admitted() -> None:
    memory = _memory()
    await _load_weak_and_strong(memory, with_graph=True)
    await memory.apply_forgetting(tenant_id=_TENANT, as_of=_T)
    weak = await _snapshot_for(memory, object_value=_WEAK_OBJECT)
    assert weak.stored
    assert weak.status is SemanticMemoryStatus.ACTIVE
    assert weak.retrieved
    assert weak.current_admitted
    assert weak.eligibility == RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION.value
    assert "structured_fit" in weak.admission_gates or "structured_path" in weak.admission_gates
    assert weak.activation is not None
    assert weak.activation < ActivationConfig().retrieval_threshold
    assert weak.retention_state != "forgotten"


@pytest.mark.asyncio
async def test_strong_interest_is_also_current_admitted() -> None:
    memory = _memory()
    await _load_weak_and_strong(memory, with_graph=True, extra_strong_support=True)
    strong = await _snapshot_for(memory, object_value=_STRONG_OBJECT)
    assert strong.stored
    assert strong.status is SemanticMemoryStatus.ACTIVE
    assert strong.support_count is not None and strong.support_count >= 2
    assert strong.retrieved
    assert strong.selected
    assert strong.eligibility in {
        RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION.value,
        RetrievalEligibility.THRESHOLD.value,
        RetrievalEligibility.ENTITY_SLOT_ADMISSION.value,
        RetrievalEligibility.SEMANTIC_SLOT_ADMISSION.value,
    }
    if (
        strong.activation is not None
        and strong.activation >= ActivationConfig().retrieval_threshold
    ):
        assert not strong.current_admitted
    else:
        assert strong.current_admitted


@pytest.mark.asyncio
async def test_weak_interest_without_graph_is_not_relationship_admitted() -> None:
    memory = _memory()
    await _load_weak_and_strong(memory, with_graph=False)
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=20,
    )
    weak_item = _inspect_item(inspection, _WEAK_OBJECT)
    if weak_item is None:
        return
    diagnostics = weak_item.diagnostics
    assert diagnostics is not None
    assert diagnostics.structured_association_fit == 0.0
    if weak_item.disposition is RecallInspectionDisposition.RETURNED:
        assert diagnostics.eligibility is RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION
        assert "structured_fit" not in _admission_gates(weak_item, config=ActivationConfig())
    else:
        assert weak_item.disposition in {
            RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
            RecallInspectionDisposition.BELOW_THRESHOLD,
        }


@pytest.mark.asyncio
async def test_weak_interest_unrelated_query_does_not_use_structured_path() -> None:
    memory = _memory()
    await _load_weak_and_strong(memory, with_graph=True)
    inspection = await memory.inspect_recall(
        "What payment method does the customer prefer?",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=20,
    )
    weak_item = _inspect_item(inspection, _WEAK_OBJECT)
    if weak_item is None:
        return
    diagnostics = weak_item.diagnostics
    assert diagnostics is not None
    assert diagnostics.structured_association_fit == 0.0
    # Shared subject/evidence features can still satisfy current relevance.
    if weak_item.disposition is RecallInspectionDisposition.RETURNED:
        assert diagnostics.eligibility is RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION
    else:
        assert weak_item.disposition in {
            RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
            RecallInspectionDisposition.BELOW_THRESHOLD,
            RecallInspectionDisposition.FILTERED_BELOW_SOFT_FLOOR,
        }


@pytest.mark.asyncio
async def test_explicit_valid_until_excludes_weak_interest() -> None:
    memory = _memory()
    await _load_weak_and_strong(
        memory,
        with_graph=True,
        weak_valid_until=_T - timedelta(days=30),
    )
    semantics = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        valid_at=_T,
    )
    assert all(item.object_value != _WEAK_OBJECT for item in semantics)
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=20,
    )
    weak_item = _inspect_item(inspection, _WEAK_OBJECT)
    if weak_item is not None:
        assert weak_item.disposition in {
            RecallInspectionDisposition.FILTERED_VALID_TIME,
            RecallInspectionDisposition.FILTERED_SEMANTIC_STATUS,
            RecallInspectionDisposition.BELOW_THRESHOLD,
            RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
        }
        assert weak_item.disposition is not RecallInspectionDisposition.RETURNED


@pytest.mark.asyncio
async def test_working_memory_selects_admitted_weak_interest_under_capacity() -> None:
    memory = _memory()
    memory._working_memory_config = WorkingMemoryConfig(max_items=8, candidate_pool_size=20)
    await _load_weak_and_strong(memory, with_graph=True)
    snapshot = await memory.select_working_memory(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
    )
    weak = await _snapshot_for(memory, object_value=_WEAK_OBJECT)
    assert snapshot.selected_count <= 8
    if weak.current_admitted:
        assert weak.selected
    assert memory._working_memory_config.minimum_selection_score == 0.0
    assert memory._working_memory_config.minimum_goal_relevance == 0.0


@pytest.mark.asyncio
async def test_predicate_agnostic_durable_and_interest_share_admission_rules() -> None:
    memory = _memory()
    await _observe_semantic(
        memory,
        source_record_id="db-postgres",
        observed_at=_T_WEAK,
        content="Production database is postgres.",
        semantic_fact={
            "predicate": "database",
            "object_value": "postgres",
            "cardinality": "one",
            "polarity": "affirm",
            "qualifiers": {},
        },
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T_WEAK)
    await _load_weak_and_strong(memory, with_graph=True)
    db = next(
        item
        for item in await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
        if item.predicate == "database"
    )
    weak = _interest(
        await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT),
        _WEAK_OBJECT,
    )
    assert db.status is SemanticMemoryStatus.ACTIVE
    assert weak.status is SemanticMemoryStatus.ACTIVE


@pytest.mark.asyncio
async def test_forgetting_does_not_change_semantic_status() -> None:
    memory = _memory()
    await _load_weak_and_strong(memory, with_graph=True)
    before = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    await memory.apply_forgetting(tenant_id=_TENANT, as_of=_T)
    after = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    assert {item.object_value: item.status for item in before} == {
        item.object_value: item.status for item in after
    }
    weak = await _snapshot_for(memory, object_value=_WEAK_OBJECT)
    assert weak.status is SemanticMemoryStatus.ACTIVE
    assert weak.retention_state in {None, "active", "fading"}


@pytest.mark.asyncio
async def test_lifecycle_snapshot_fields_are_visible() -> None:
    memory = _memory()
    await _load_weak_and_strong(memory, with_graph=True, extra_strong_support=True)
    await memory.apply_forgetting(tenant_id=_TENANT, as_of=_T)
    weak = await _snapshot_for(memory, object_value=_WEAK_OBJECT)
    strong = await _snapshot_for(memory, object_value=_STRONG_OBJECT)
    for snap in (weak, strong):
        assert snap.stored
        assert snap.derivation_episode_ids
        assert snap.observation_ids
        assert snap.eligibility is not None
        assert snap.admission_reason is not None
        assert snap.relevance_tier is not None
        assert snap.semantic_relevance is not None
        assert snap.activation is not None
        assert snap.base_level is not None
    assert (weak.support_count or 0) < (strong.support_count or 0)
    assert weak.last_supported_at is not None
    assert strong.last_supported_at is not None
    assert weak.last_supported_at < strong.last_supported_at
    assert weak.confidence is not None
    assert strong.confidence is not None
    assert weak.first_supported_at == _T_WEAK
    assert weak.episode_ended_at == _T_WEAK
    assert weak.created_at == _T_WEAK
    assert weak.learning_count == 0
    assert strong.learning_count == 0
    assert weak.valid_from is None
    assert weak.valid_until is None
    assert not hasattr(StoredSemanticMemory, "rehearsal_count")
    assert not hasattr(StoredSemanticMemory, "reinforcement_count")
    assert not hasattr(ObservationInput, "strength")


@pytest.mark.asyncio
async def test_cardinality_one_superseded_value_is_excluded() -> None:
    memory = _memory()
    await _observe_semantic(
        memory,
        source_record_id="size-l",
        observed_at=_T_WEAK,
        content="Customer jacket size recorded as L.",
        semantic_fact={
            "predicate": "jacket_size",
            "object_value": "L",
            "cardinality": "one",
            "polarity": "affirm",
            "qualifiers": {},
        },
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T_WEAK)
    await _observe_semantic(
        memory,
        source_record_id="size-m",
        observed_at=_T_STRONG,
        content="Customer jacket size updated to M.",
        semantic_fact={
            "predicate": "jacket_size",
            "object_value": "M",
            "cardinality": "one",
            "polarity": "affirm",
            "qualifiers": {},
        },
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T_STRONG)
    listed = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
    )
    superseded_listed = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        status=SemanticMemoryStatus.SUPERSEDED,
    )
    by_value = {
        item.object_value.casefold(): item.status
        for item in (*listed, *superseded_listed)
        if item.predicate == "jacket_size"
    }
    assert by_value["l"] is SemanticMemoryStatus.SUPERSEDED
    assert by_value["m"] is SemanticMemoryStatus.ACTIVE
    inspection = await memory.inspect_recall(
        "What jacket size does the customer wear?",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        valid_at=_T,
        limit=20,
    )
    superseded = next(
        (
            item
            for item in (*inspection.returned, *inspection.rejected)
            if item.memory_kind is MemoryKind.SEMANTIC
            and isinstance(item.memory, StoredSemanticMemory)
            and item.memory.predicate == "jacket_size"
            and item.memory.object_value.casefold() == "l"
        ),
        None,
    )
    if superseded is not None:
        assert superseded.disposition is RecallInspectionDisposition.FILTERED_SEMANTIC_STATUS
        assert superseded.disposition is not RecallInspectionDisposition.RETURNED
