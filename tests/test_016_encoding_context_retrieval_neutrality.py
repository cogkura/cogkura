"""Mandatory retrieval-neutrality gate for encoding context in 0.16.0."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.models import MemoryKind
from cogkura.observations.encoding_context import ObservationContext

_T = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "neutral"
_SUBJECT = "developer-1"
_QUERY = "Why did we decide not to use Redis?"
_GOAL = "reduce-operational-complexity"


def _base_observations() -> list[ObservationInput]:
    return [
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="1",
            source_type="discussion",
            content=(
                "We decided not to use Redis because another stateful dependency "
                "would increase operational complexity."
            ),
            observed_at=_T,
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("redis", "payments-api"),
            },
        ),
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="2",
            source_type="discussion",
            content="PostgreSQL remains the coordination store for payments-api.",
            observed_at=_T + timedelta(minutes=5),
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("postgresql", "payments-api"),
            },
        ),
    ]


def _context_observations() -> list[ObservationInput]:
    return [
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="1",
            source_type="discussion",
            content=(
                "We decided not to use Redis because another stateful dependency "
                "would increase operational complexity."
            ),
            observed_at=_T,
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("redis", "payments-api"),
            },
            context=ObservationContext(
                conversation_id="arch-42",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="payments-api",
                temporal_context=("queue-redesign",),
            ),
        ),
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="2",
            source_type="discussion",
            content="PostgreSQL remains the coordination store for payments-api.",
            observed_at=_T + timedelta(minutes=5),
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("postgresql", "payments-api"),
            },
            context=ObservationContext(
                conversation_id="arch-42",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="payments-api",
                temporal_context=("queue-redesign",),
            ),
        ),
    ]


async def _build_store(observations: list[ObservationInput]) -> Memory:
    memory = Memory()
    for observation in observations:
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))
    return memory


def _recall_snapshot(results: list) -> list[tuple[str, MemoryKind, float]]:
    return [
        (result.memory.memory_key, result.memory_kind, round(result.score, 6)) for result in results
    ]


@pytest.mark.asyncio
async def test_recall_is_identical_with_or_without_encoding_context() -> None:
    baseline = await _build_store(_base_observations())
    contextual = await _build_store(_context_observations())

    baseline_recall = await baseline.recall(_QUERY, tenant_id=_TENANT)
    contextual_recall = await contextual.recall(_QUERY, tenant_id=_TENANT)
    assert _recall_snapshot(baseline_recall) == _recall_snapshot(contextual_recall)

    baseline_inspection = await baseline.inspect_recall(_QUERY, tenant_id=_TENANT)
    contextual_inspection = await contextual.inspect_recall(_QUERY, tenant_id=_TENANT)
    baseline_returned = [
        (item.memory.memory_key, item.disposition.value, round(item.score, 6))
        for item in baseline_inspection.returned
    ]
    contextual_returned = [
        (item.memory.memory_key, item.disposition.value, round(item.score, 6))
        for item in contextual_inspection.returned
    ]
    assert baseline_returned == contextual_returned


@pytest.mark.asyncio
async def test_working_memory_selection_is_identical_with_or_without_encoding_context() -> None:
    baseline = await _build_store(_base_observations())
    contextual = await _build_store(_context_observations())

    baseline_wm = await baseline.select_working_memory(
        _QUERY,
        tenant_id=_TENANT,
        goal=_GOAL,
        prompt_budget_tokens=512,
    )
    contextual_wm = await contextual.select_working_memory(
        _QUERY,
        tenant_id=_TENANT,
        goal=_GOAL,
        prompt_budget_tokens=512,
    )
    baseline_items = [
        (item.memory_kind, item.memory.memory_key, round(item.recall.score, 6))
        for item in baseline_wm.items
    ]
    contextual_items = [
        (item.memory_kind, item.memory.memory_key, round(item.recall.score, 6))
        for item in contextual_wm.items
    ]
    assert baseline_items == contextual_items

    baseline_context = await baseline.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        goal=_GOAL,
        prompt_budget_tokens=512,
    )
    contextual_context = await contextual.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        goal=_GOAL,
        prompt_budget_tokens=512,
    )
    assert baseline_context.render() == contextual_context.render()


@pytest.mark.asyncio
async def test_redis_payments_fixture_round_trips_encoding_context() -> None:
    memory = await _build_store(_context_observations())
    episodes = await memory.list_episodes(tenant_id=_TENANT)
    assert len(episodes) == 1
    signature = episodes[0].encoding_context
    assert signature.conversation_ids == ("arch-42",)
    assert signature.thread_ids == ("queue-selection",)
    assert signature.goals == ("reduce-operational-complexity",)
    assert signature.domains == ("payments-api",)
    assert "redis" in signature.entity_ids
    assert not signature.is_empty()

    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT)
    assert inspection.returned
    episode = inspection.returned[0].memory
    assert episode.encoding_context.to_canonical_dict() == signature.to_canonical_dict()
