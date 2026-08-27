"""Regression tests for 0.15.1 processing-cadence recall stability."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import Memory, ObservationInput
from cogkura.models import (
    CognitiveTraceOrigin,
    RecallInspectionDisposition,
)

_TENANT = "company_123"
_SUBJECT = "customer_42"
_JAN = datetime(2025, 1, 15, 10, 0, tzinfo=UTC)
_MAR = datetime(2025, 3, 14, 10, 0, tzinfo=UTC)
_JUN = datetime(2025, 6, 2, 10, 0, tzinfo=UTC)
_QUERY_TIME = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

_LIGHTWEIGHT_FACT = {
    "predicate": "outerwear_weight_preference",
    "object_value": "lightweight",
    "cardinality": "one",
    "polarity": "affirm",
    "qualifiers": {},
}

_HIKING_FACT = {
    "predicate": "hiking_interest",
    "object_value": "high",
    "cardinality": "one",
    "polarity": "affirm",
    "qualifiers": {},
}


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
            "entity_ids": [_SUBJECT, "hiking-boots"],
        },
    )


async def _build_incremental_memory() -> Memory:
    memory = Memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="msg-jan",
            conversation_id="conv-jan",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_JAN,
            content="Customer prefers lightweight outerwear.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JAN)
    await memory.observe(
        _browse_observation(
            source_record_id="browse-mar",
            conversation_id="conv-mar",
            observed_at=_MAR,
            content="Customer browsed hiking boots in March.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_MAR)
    await memory.observe(
        _semantic_observation(
            source_record_id="msg-jun",
            conversation_id="conv-jun",
            semantic_fact=_HIKING_FACT,
            observed_at=_JUN,
            content="Customer confirmed strong hiking interest.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JUN)
    return memory


async def _build_deferred_memory() -> Memory:
    memory = Memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="msg-jan",
            conversation_id="conv-jan",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_JAN,
            content="Customer prefers lightweight outerwear.",
        )
    )
    await memory.observe(
        _browse_observation(
            source_record_id="browse-mar",
            conversation_id="conv-mar",
            observed_at=_MAR,
            content="Customer browsed hiking boots in March.",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="msg-jun",
            conversation_id="conv-jun",
            semantic_fact=_HIKING_FACT,
            observed_at=_JUN,
            content="Customer confirmed strong hiking interest.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JUN)
    return memory


def _memory_keys(memories: list) -> set[str]:
    return {memory.memory_key for memory in memories}


@pytest.mark.asyncio
async def test_incremental_and_deferred_produce_same_active_memories() -> None:
    incremental = await _build_incremental_memory()
    deferred = await _build_deferred_memory()

    incremental_episodes = await incremental.list_episodes(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
    )
    deferred_episodes = await deferred.list_episodes(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
    )
    assert _memory_keys(incremental_episodes) == _memory_keys(deferred_episodes)

    incremental_semantics = await incremental.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
    )
    deferred_semantics = await deferred.list_semantic_memories(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
    )
    assert _memory_keys(incremental_semantics) == _memory_keys(deferred_semantics)


@pytest.mark.asyncio
async def test_processing_cadence_invariance_for_recall() -> None:
    incremental = await _build_incremental_memory()
    deferred = await _build_deferred_memory()
    query = "lightweight hiking preference"

    incremental_results = await incremental.recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=10,
    )
    deferred_results = await deferred.recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=10,
    )

    incremental_keys = [result.memory.memory_key for result in incremental_results]
    deferred_keys = [result.memory.memory_key for result in deferred_results]
    assert incremental_keys == deferred_keys

    incremental_by_key = {result.memory.memory_key: result for result in incremental_results}
    deferred_by_key = {result.memory.memory_key: result for result in deferred_results}
    for key in incremental_by_key:
        left = incremental_by_key[key]
        right = deferred_by_key[key]
        assert abs(left.activation - right.activation) < 1e-12


@pytest.mark.asyncio
async def test_repeated_unchanged_process_does_not_change_recall() -> None:
    memory = await _build_incremental_memory()
    query = "lightweight hiking preference"
    first = await memory.recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=10,
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_MAR)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JUN)
    second = await memory.recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=10,
    )
    assert [item.memory.memory_key for item in first] == [item.memory.memory_key for item in second]
    for left, right in zip(first, second, strict=True):
        assert abs(left.activation - right.activation) < 1e-12


@pytest.mark.asyncio
async def test_historical_backfill_uses_evidence_time_not_materialization() -> None:
    memory = Memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="historical-a",
            conversation_id="conv-hist-a",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_JAN,
            content="Historical lightweight preference.",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="historical-b",
            conversation_id="conv-hist-b",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_MAR,
            content="Repeated lightweight preference.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JUN)
    semantics = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=_SUBJECT)
    assert len(semantics) == 1
    assert semantics[0].created_at == _JUN
    assert semantics[0].first_supported_at == _JAN

    inspection = await memory.inspect_recall(
        "lightweight preference",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=5,
    )
    semantic_candidates = [
        *inspection.returned,
        *inspection.rejected,
    ]
    semantic = next(item for item in semantic_candidates if item.memory_kind.value == "semantic")
    assert any(
        trace.origin is CognitiveTraceOrigin.SUPPORTED and trace.referenced_at == _JAN
        for trace in semantic.cognitive_traces
    )


@pytest.mark.asyncio
async def test_inspect_recall_matches_recall_returned_identities() -> None:
    memory = await _build_incremental_memory()
    query = "lightweight hiking preference"
    recalled = await memory.recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=5,
    )
    inspection = await memory.inspect_recall(
        query,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=5,
    )
    assert [item.memory.memory_key for item in recalled] == [
        item.memory.memory_key for item in inspection.returned
    ]
    assert all(
        item.disposition is RecallInspectionDisposition.RETURNED for item in inspection.returned
    )


@pytest.mark.asyncio
async def test_growing_grouped_episode_cadence_invariance() -> None:
    group_id = "trip-planning"
    incremental = Memory()
    await incremental.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="commerce.events",
            source_record_id="grp-1",
            content="Started trip planning browse.",
            observed_at=_JAN,
            metadata={"conversation_id": group_id, "entity_ids": [_SUBJECT]},
        )
    )
    await incremental.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JAN)
    await incremental.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="commerce.events",
            source_record_id="grp-2",
            content="Continued trip planning browse.",
            observed_at=_MAR,
            metadata={"conversation_id": group_id, "entity_ids": [_SUBJECT]},
        )
    )
    await incremental.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_MAR)

    deferred = Memory()
    await deferred.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="commerce.events",
            source_record_id="grp-1",
            content="Started trip planning browse.",
            observed_at=_JAN,
            metadata={"conversation_id": group_id, "entity_ids": [_SUBJECT]},
        )
    )
    await deferred.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="commerce.events",
            source_record_id="grp-2",
            content="Continued trip planning browse.",
            observed_at=_MAR,
            metadata={"conversation_id": group_id, "entity_ids": [_SUBJECT]},
        )
    )
    await deferred.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_MAR)

    incremental_episode = (await incremental.list_episodes(tenant_id=_TENANT))[0]
    deferred_episode = (await deferred.list_episodes(tenant_id=_TENANT))[0]
    assert incremental_episode.ended_at == deferred_episode.ended_at == _MAR
    assert incremental_episode.created_at != deferred_episode.created_at

    incremental_results = await incremental.recall(
        "trip planning",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=5,
    )
    deferred_results = await deferred.recall(
        "trip planning",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=5,
    )
    assert [item.memory.memory_key for item in incremental_results] == [
        item.memory.memory_key for item in deferred_results
    ]


@pytest.mark.asyncio
async def test_new_support_can_strengthen_semantic_recall() -> None:
    single = Memory()
    await single.observe(
        _semantic_observation(
            source_record_id="support-a",
            conversation_id="conv-a",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_JAN,
            content="First lightweight preference.",
        )
    )
    await single.observe(
        _semantic_observation(
            source_record_id="support-a2",
            conversation_id="conv-a2",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_MAR,
            content="Second lightweight preference for single.",
        )
    )
    await single.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_MAR)

    reinforced = Memory()
    await reinforced.observe(
        _semantic_observation(
            source_record_id="support-a",
            conversation_id="conv-a",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_JAN,
            content="First lightweight preference.",
        )
    )
    await reinforced.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JAN)
    await reinforced.observe(
        _semantic_observation(
            source_record_id="support-b",
            conversation_id="conv-b",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_MAR,
            content="Second lightweight preference.",
        )
    )
    await reinforced.observe(
        _semantic_observation(
            source_record_id="support-c",
            conversation_id="conv-c",
            semantic_fact=_LIGHTWEIGHT_FACT,
            observed_at=_JUN,
            content="Third lightweight preference.",
        )
    )
    await reinforced.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_JUN)

    single_inspection = await single.inspect_recall(
        "lightweight preference",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=10,
    )
    reinforced_inspection = await reinforced.inspect_recall(
        "lightweight preference",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_QUERY_TIME,
        limit=10,
    )

    def _semantic_activation(inspection) -> float:
        candidates = [*inspection.returned, *inspection.rejected]
        semantics = [item for item in candidates if item.memory_kind.value == "semantic"]
        assert semantics, "expected semantic candidate in inspection"
        return max(item.activation for item in semantics)

    assert _semantic_activation(reinforced_inspection) >= _semantic_activation(single_inspection)
