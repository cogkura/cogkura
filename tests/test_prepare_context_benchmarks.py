"""Reproducible prepare_context / recall performance baselines (0.15.11)."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator

_TENANT = "bench"
_SUBJECT = "subject-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_QUERY = "Recommend durable outerwear for the customer."
_GOAL = "Help choose appropriate outerwear."
_WARM_ITERATIONS = 20


@dataclass(frozen=True)
class BenchStats:
    label: str
    cold_ms: float
    median_ms: float
    p95_ms: float
    candidate_count: int | None = None
    returned_count: int | None = None
    chunk_count: int | None = None
    estimated_tokens: int | None = None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


async def _timed_async_ms(coro) -> float:
    start = time.perf_counter()
    await coro
    end = time.perf_counter()
    return (end - start) * 1000.0


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


async def _build_fixture(memory: Memory, *, scale: str) -> None:
    episode_counts = {"small": 8, "medium": 40, "large": 120}
    count = episode_counts[scale]
    for index in range(count):
        observed_at = _T - timedelta(days=count - index)
        predicate = "activity_interest" if index % 3 == 0 else None
        metadata: dict = {
            "conversation_id": f"conv-{index}",
            "entity_ids": [_SUBJECT, f"product-{index % 5}"],
        }
        if predicate is not None:
            metadata["semantic_facts"] = [
                {
                    "predicate": predicate,
                    "object_value": f"activity-{index % 4}",
                    "cardinality": "many",
                    "polarity": "affirm",
                    "qualifiers": {},
                }
            ]
        if index % 7 == 0:
            metadata["relationships"] = [
                {
                    "source_entity_id": f"product-{index % 5}",
                    "relation_type": "is_a",
                    "target_entity_id": "outerwear",
                    "provenance": "catalog",
                }
            ]
        await memory.observe(
            ObservationInput(
                tenant_id=_TENANT,
                subject_id=_SUBJECT,
                actor_id=_SUBJECT,
                source_namespace="events",
                source_record_id=f"evt-{index}",
                event_type="browse",
                content=f"Customer browsed product-{index % 5} listings.",
                observed_at=observed_at,
                metadata=metadata,
            )
        )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T)


async def _bench_prepare_context(memory: Memory) -> BenchStats:
    cold_ms = await _timed_async_ms(
        memory.prepare_context(
            _QUERY,
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            goal=_GOAL,
            as_of=_T,
        )
    )
    warm_samples = [
        await _timed_async_ms(
            memory.prepare_context(
                _QUERY,
                tenant_id=_TENANT,
                subject_id=_SUBJECT,
                goal=_GOAL,
                as_of=_T,
            )
        )
        for _ in range(_WARM_ITERATIONS)
    ]
    snapshot = await memory.select_working_memory(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        goal=_GOAL,
        as_of=_T,
    )
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
        limit=20,
    )
    context = await memory.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        goal=_GOAL,
        as_of=_T,
    )
    return BenchStats(
        label="prepare_context",
        cold_ms=cold_ms,
        median_ms=statistics.median(warm_samples),
        p95_ms=_percentile(warm_samples, 95.0),
        candidate_count=snapshot.candidate_count,
        returned_count=len(inspection.returned),
        chunk_count=snapshot.selected_chunk_count,
        estimated_tokens=context.estimated_tokens,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scale", ["small", "medium", "large"])
async def test_prepare_context_benchmark_records_stats(scale: str) -> None:
    memory = _memory()
    await _build_fixture(memory, scale=scale)
    stats = await _bench_prepare_context(memory)
    assert stats.cold_ms >= 0.0
    assert stats.median_ms >= 0.0
    assert stats.p95_ms >= stats.median_ms
    assert stats.candidate_count is not None and stats.candidate_count >= 0
    assert stats.chunk_count is not None and stats.chunk_count >= 0
    assert stats.estimated_tokens is not None and stats.estimated_tokens >= 0
