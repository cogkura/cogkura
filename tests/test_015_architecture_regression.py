"""Canonical 0.15 architecture end-to-end regression fixture.

Freezes semantic state, relationships, broad recall, chunk construction,
bounded working-memory selection, rendering/provenance, and record_context_use.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    MemoryKind,
    SemanticCardinality,
    SemanticMemoryStatus,
    StoredSemanticMemory,
    WorkingMemoryChunkType,
    WorkingMemoryConfig,
)

_TENANT = "arch"
_SUBJECT = "customer-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_QUERY = "Recommend outerwear for the customer."
_GOAL = "Help choose outerwear."


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


@pytest.mark.asyncio
async def test_015_architecture_regression_fixture() -> None:
    memory = _memory()
    memory._working_memory_config = WorkingMemoryConfig(max_items=4, candidate_pool_size=50)

    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="catalog",
            source_record_id="rel-1",
            event_type="taxonomy",
            content="Catalog import",
            observed_at=_T - timedelta(days=200),
            metadata={
                "conversation_id": "conv-catalog",
                "relationships": [
                    {
                        "source_entity_id": "shell-a",
                        "relation_type": "is_a",
                        "target_entity_id": "outerwear",
                        "provenance": "catalog",
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T - timedelta(days=200))

    t_old = _T - timedelta(days=120)
    t_new = _T - timedelta(days=10)
    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id="size-m",
            event_type="message",
            content="Size recorded as M.",
            observed_at=t_old,
            metadata={
                "conversation_id": "conv-size-m",
                "entity_ids": [_SUBJECT],
                "semantic_facts": [
                    {
                        "predicate": "size",
                        "object_value": "M",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_old)

    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id="colours",
            event_type="message",
            content="Customer prefers black, navy and grey jackets.",
            observed_at=_T - timedelta(days=60),
            metadata={
                "conversation_id": "conv-colour",
                "entity_ids": [_SUBJECT],
                "semantic_facts": [
                    {
                        "predicate": "colour_preference",
                        "object_value": colour,
                        "cardinality": "many",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                    for colour in ("black", "navy", "grey")
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T - timedelta(days=60))

    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id="interest-a",
            event_type="browse",
            content="Customer browsed activity-a gear.",
            observed_at=_T - timedelta(days=90),
            metadata={
                "conversation_id": "conv-interest-a",
                "entity_ids": [_SUBJECT, "shell-a"],
                "semantic_facts": [
                    {
                        "predicate": "activity_interest",
                        "object_value": "activity-a",
                        "cardinality": "many",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
            },
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id="interest-b",
            event_type="purchase",
            content="Customer purchased activity-b equipment.",
            observed_at=t_new,
            metadata={
                "conversation_id": "conv-interest-b",
                "entity_ids": [_SUBJECT],
                "semantic_facts": [
                    {
                        "predicate": "activity_interest",
                        "object_value": "activity-b",
                        "cardinality": "many",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_new)

    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id="size-l",
            event_type="message",
            content="Size updated to L.",
            observed_at=t_new,
            metadata={
                "conversation_id": "conv-size-l",
                "entity_ids": [_SUBJECT],
                "semantic_facts": [
                    {
                        "predicate": "size",
                        "object_value": "L",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_new)

    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="events",
            source_record_id="pref-light",
            event_type="purchase",
            content="Customer purchased a lightweight shell in size L.",
            observed_at=t_new,
            metadata={
                "conversation_id": "conv-pref",
                "entity_ids": [_SUBJECT],
                "semantic_facts": [
                    {
                        "predicate": "outerwear_weight_preference",
                        "object_value": "lightweight",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_new)

    sizes = {
        item.object_value.casefold(): item.status
        for item in (
            *await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT),
            *await memory.list_semantic_memories(
                tenant_id=_TENANT,
                subject_id=_SUBJECT,
                status=SemanticMemoryStatus.SUPERSEDED,
            ),
        )
        if item.predicate == "size"
    }
    assert sizes["m"] is SemanticMemoryStatus.SUPERSEDED
    assert sizes["l"] is SemanticMemoryStatus.ACTIVE

    historical = await memory.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        valid_at=t_old + timedelta(days=1),
    )
    historical_size = next(item for item in historical if item.predicate == "size")
    assert historical_size.object_value.casefold() == "m"

    many_interests = [
        item
        for item in await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
        if item.predicate == "activity_interest"
    ]
    assert len(many_interests) >= 2
    assert all(item.cardinality is SemanticCardinality.MANY for item in many_interests)
    assert len({item.memory_key for item in many_interests}) >= 2

    context = await memory.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        goal=_GOAL,
        as_of=_T,
    )
    rendered = context.render()
    assert rendered.strip()
    assert "size L" not in rendered

    support_chunks = [
        item
        for item in context.working_memory.items
        if item.chunk and item.chunk.chunk_type is WorkingMemoryChunkType.SEMANTIC_WITH_SUPPORT
    ]
    assert len(support_chunks) >= 1
    support_item = support_chunks[0]
    support_chunk = support_item.chunk
    assert support_chunk is not None
    semantic_member = next(
        recall
        for recall in support_item.member_recalls
        if recall.memory_kind is MemoryKind.SEMANTIC
        and isinstance(recall.memory, StoredSemanticMemory)
    )
    support_episode_keys = {
        recall.memory.memory_key
        for recall in support_item.member_recalls
        if recall.memory_kind is MemoryKind.EPISODE
    }
    assert support_episode_keys
    assert support_chunk.serialized_text == semantic_member.memory.statement
    assert any(
        identity.memory_kind is MemoryKind.EPISODE for identity in support_chunk.member_identities
    )

    collection_chunks = [
        item
        for item in context.working_memory.items
        if item.chunk and item.chunk.chunk_type is WorkingMemoryChunkType.SEMANTIC_COLLECTION
    ]
    assert len(collection_chunks) >= 1
    colour_chunk = collection_chunks[0].chunk
    assert colour_chunk is not None
    assert colour_chunk.members_total == 3

    assert context.working_memory.candidate_count > context.working_memory.selected_chunk_count
    assert context.working_memory.selected_chunk_count <= 4

    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        limit=20,
    )
    assert inspection.relationship_paths_used > 0
    relationship_semantic = next(
        item
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC
        and isinstance(item.memory, StoredSemanticMemory)
        and item.memory.predicate == "activity_interest"
        and item.diagnostics is not None
        and item.diagnostics.association_path is not None
        and item.diagnostics.association_path.hop_kind == "relationship"
    )
    path = relationship_semantic.diagnostics.association_path
    assert path is not None
    assert any(edge.relation_type == "is_a" for edge in path.relationship_edges)
    assert "shell-a" in {edge.source_entity_id for edge in path.relationship_edges}
    assert "outerwear" in {edge.target_entity_id for edge in path.relationship_edges}

    await memory.record_context_use(context, referenced_at=_T)
