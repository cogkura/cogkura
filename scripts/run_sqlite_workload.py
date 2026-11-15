"""Observational, logical-parity workload for the embedded SQLite backend.

Run from the repository root with ``uv run python scripts/run_sqlite_workload.py``.
Results are measurements, not pass/fail performance thresholds.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import sqlite3
import statistics
import tempfile
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    ObservationInput,
    ObservationRevisionIdentity,
)
from cogkura.storage.sqlite import SQLiteStorage

_T = datetime(2026, 1, 1, tzinfo=UTC)


def _observation(index: int, version: int) -> ObservationInput:
    return ObservationInput(
        tenant_id="benchmark",
        subject_id=f"service-{index:04d}",
        source_namespace="workload",
        source_record_id=f"source-{index:04d}",
        source_version=str(version),
        content=f"Service {index:04d} deploys to environment env-{index:04d}, revision {version}.",
        observed_at=_T + timedelta(seconds=version),
        metadata={
            "conversation_id": f"thread-{index:04d}",
            "semantic_facts": [
                {
                    "predicate": "deployment_environment",
                    "object_value": f"env-{index:04d}",
                    "cardinality": "one",
                    "authority": {"basis": "application_asserted"},
                }
            ],
        },
    )


def _percentile(samples: list[float], quantile: float) -> float:
    values = sorted(samples)
    return values[max(0, math.ceil(len(values) * quantile) - 1)]


async def _run(
    memory: Memory,
    count: int,
    revisions: int,
    repetitions: int,
    storage: SQLiteStorage | None = None,
) -> dict[str, Any]:
    ingest = []
    for version in range(1, revisions + 1):
        start = time.perf_counter()
        for index in range(count):
            await memory.observe(_observation(index, version))
        ingest.append(time.perf_counter() - start)
    start = time.perf_counter()
    processed = await memory.process(tenant_id="benchmark", as_of=_T + timedelta(days=1))
    process_seconds = time.perf_counter() - start
    episodes = await memory.list_episodes(tenant_id="benchmark")
    semantics = await memory.list_semantic_memories(tenant_id="benchmark")
    history = await memory.list_semantic_revisions(tenant_id="benchmark")
    observations = await memory._observation_store.list(tenant_id="benchmark")
    references = [
        ObservationRevisionIdentity(item.id, number)
        for item in observations
        for number in range(1, revisions + 1)
    ]
    historical = await memory._observation_store.get_revisions_many(
        tenant_id="benchmark", references=references
    )
    historical_values = sorted(
        (value.source_record_id, value.identity.observation_revision, value.source_version)
        for value in historical.values()
    )
    latencies = []
    outputs = []
    statements: list[str] = []
    if storage is not None:
        await storage._run(lambda: storage._conn().set_trace_callback(statements.append))
    for n in range(repetitions):
        query = f"service-{n % count:04d} deployment environment env-{n % count:04d}"
        start = time.perf_counter()
        results = await memory.recall(
            query, tenant_id="benchmark", as_of=_T + timedelta(days=2), limit=5
        )
        latencies.append(time.perf_counter() - start)
        outputs.append(
            [
                (item.memory_kind.value, item.memory.statement, round(item.score, 8))
                for item in results
            ]
        )
    if storage is not None:
        await storage._run(lambda: storage._conn().set_trace_callback(None))
    return {
        "recall_sql_statements_total": len(statements) if storage is not None else None,
        "ingest_seconds_per_pass": ingest,
        "process_seconds": process_seconds,
        "recall_median_ms": statistics.median(latencies) * 1000,
        "recall_p95_ms": _percentile(latencies, 0.95) * 1000,
        "counts": {
            "observations": len(observations),
            "observation_revisions": len(historical),
            "episodes": len(episodes),
            "semantic_memories": len(semantics),
            "semantic_revisions": len(history),
        },
        "process_created_episodes": processed.episodes.created,
        "historical_values": historical_values,
        "semantic_values": sorted(
            (item.subject_id, item.predicate, item.object_value, item.status.value)
            for item in semantics
        ),
        "recall_outputs": outputs,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=int, default=120)
    parser.add_argument("--revisions", type=int, default=4)
    parser.add_argument("--recalls", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.observations, args.revisions, args.recalls) < 1:
        parser.error("workload arguments must be positive")
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-workload-") as directory:
        path = Path(directory) / "memory.db"
        tracemalloc.start()
        owner = await EmbeddedMemory.open(path, **options)
        sqlite_result = await _run(
            owner.memory, args.observations, args.revisions, args.recalls, owner._storage
        )
        _, sqlite_peak = tracemalloc.get_traced_memory()
        sizes = {
            suffix: (
                Path(str(path) + suffix).stat().st_size if Path(str(path) + suffix).exists() else 0
            )
            for suffix in ("", "-wal", "-shm")
        }
        await owner.close()
        post_close_bytes = path.stat().st_size
        tracemalloc.stop()
        tracemalloc.start()
        memory_result = await _run(
            Memory(**options), args.observations, args.revisions, args.recalls
        )
        _, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    parity = {
        "counts": sqlite_result["counts"] == memory_result["counts"],
        "semantic_values": sqlite_result["semantic_values"] == memory_result["semantic_values"],
        "historical_values": sqlite_result["historical_values"]
        == memory_result["historical_values"],
        "recall_outputs": sqlite_result["recall_outputs"] == memory_result["recall_outputs"],
    }
    comparison = {}
    for backend, result in (("sqlite", sqlite_result), ("in_memory", memory_result)):
        comparison[backend] = {
            "ranked_recall_outputs": result.pop("recall_outputs"),
            "historical_values_sha256": hashlib.sha256(
                json.dumps(
                    result.pop("historical_values"), separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest(),
            "semantic_values_sha256": hashlib.sha256(
                json.dumps(
                    result.pop("semantic_values"), separators=(",", ":"), ensure_ascii=False
                ).encode()
            ).hexdigest(),
        }
    report = {
        "workload": {
            "observations": args.observations,
            "revisions_per_observation": args.revisions,
            "recall_queries": args.recalls,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        },
        "sqlite": {
            **sqlite_result,
            "peak_python_traced_bytes": sqlite_peak,
            "file_bytes": sizes,
            "database_bytes_after_close": post_close_bytes,
        },
        "in_memory": {**memory_result, "peak_python_traced_bytes": memory_peak},
        "logical_parity": parity,
        "logical_comparisons": comparison,
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if not all(parity.values()):
        raise SystemExit("Logical parity failed; investigate before accepting a baseline")


if __name__ == "__main__":
    asyncio.run(main())
