"""Reproducible prepare_context performance measurement (0.15.12)."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator

_TENANT = "bench"
_SUBJECT = "subject-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_QUERY = "Recommend durable outerwear for the customer."
_GOAL = "Help choose appropriate outerwear."
_WARM_ITERATIONS = 20
_FIXTURE_SCALES = ("small", "medium", "large")
_EPISODE_COUNTS = {"small": 8, "medium": 40, "large": 120}


@dataclass(frozen=True)
class BenchStats:
    fixture: str
    memory_count: int
    relationship_count: int
    cold_ms: float
    warm_median_ms: float
    warm_p95_ms: float
    candidate_count: int
    returned_count: int
    chunk_count: int
    token_count: int
    warmup_count: int
    run_count: int


@dataclass(frozen=True)
class BenchEnvironment:
    package_version: str
    git_commit: str | None
    python_version: str
    platform: str


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


def memory_factory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


async def build_fixture(memory: Memory, *, scale: str) -> tuple[int, int]:
    count = _EPISODE_COUNTS[scale]
    relationship_count = 0
    for index in range(count):
        observed_at = _T - timedelta(days=count - index)
        predicate = "activity_interest" if index % 3 == 0 else None
        metadata: dict[str, object] = {
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
            relationship_count += 1
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
    return count, relationship_count


async def measure_workload_counts(memory: Memory) -> BenchStats:
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
        fixture="prepare_context",
        memory_count=0,
        relationship_count=0,
        cold_ms=0.0,
        warm_median_ms=0.0,
        warm_p95_ms=0.0,
        candidate_count=snapshot.candidate_count,
        returned_count=len(inspection.returned),
        chunk_count=snapshot.selected_chunk_count,
        token_count=context.estimated_tokens,
        warmup_count=0,
        run_count=0,
    )


async def measure_timings(
    memory: Memory, *, scale: str, memory_count: int, relationship_count: int
) -> BenchStats:
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
    counts = await measure_workload_counts(memory)
    return BenchStats(
        fixture=scale,
        memory_count=memory_count,
        relationship_count=relationship_count,
        cold_ms=cold_ms,
        warm_median_ms=statistics.median(warm_samples),
        warm_p95_ms=_percentile(warm_samples, 95.0),
        candidate_count=counts.candidate_count,
        returned_count=counts.returned_count,
        chunk_count=counts.chunk_count,
        token_count=counts.token_count,
        warmup_count=0,
        run_count=_WARM_ITERATIONS,
    )


async def run_all_benchmarks() -> list[BenchStats]:
    results: list[BenchStats] = []
    for scale in _FIXTURE_SCALES:
        memory = memory_factory()
        memory_count, relationship_count = await build_fixture(memory, scale=scale)
        results.append(
            await measure_timings(
                memory,
                scale=scale,
                memory_count=memory_count,
                relationship_count=relationship_count,
            )
        )
    return results


def package_version() -> str:
    try:
        from importlib.metadata import version

        return version("cogkura")
    except Exception:
        from cogkura import __version__

        return __version__


def git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip() or None


def environment_metadata() -> BenchEnvironment:
    return BenchEnvironment(
        package_version=package_version(),
        git_commit=git_commit(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )


def stats_to_dict(stats: BenchStats, env: BenchEnvironment) -> dict[str, Any]:
    payload = asdict(stats)
    payload.update(asdict(env))
    return payload


def format_markdown_table(results: list[BenchStats], env: BenchEnvironment) -> str:
    lines = [
        "# prepare_context benchmark results",
        "",
        "> Timings are observational, machine-dependent evidence — not acceptance thresholds.",
        "",
        f"- CogKura version: `{env.package_version}`",
        f"- Git commit: `{env.git_commit or 'unknown'}`",
        f"- Python: `{env.python_version}`",
        f"- Platform: `{env.platform}`",
        f"- Warm runs per fixture: `{_WARM_ITERATIONS}`",
        "",
        "| fixture | memories | relationships | cold_ms | warm_median_ms | "
        "warm_p95_ms | candidates | returned | chunks | tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {fixture} | {memory_count} | {relationship_count} | {cold_ms:.2f} | "
            "{warm_median_ms:.2f} | {warm_p95_ms:.2f} | {candidate_count} | "
            "{returned_count} | {chunk_count} | {token_count} |".format(**asdict(row))
        )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(results: list[BenchStats], output_dir: Path) -> None:
    env = environment_metadata()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": asdict(env),
        "warm_iterations": _WARM_ITERATIONS,
        "fixtures": [stats_to_dict(row, env) for row in results],
    }
    json_path = output_dir / "benchmark-results.json"
    md_path = output_dir / "benchmark-results.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown_table(results, env), encoding="utf-8")


def print_table(results: list[BenchStats], env: BenchEnvironment) -> None:
    print(format_markdown_table(results, env))
