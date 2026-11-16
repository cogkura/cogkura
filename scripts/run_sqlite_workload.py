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
import sys
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
    SemanticCorrectionConfig,
    SemanticSlotQuery,
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
    *,
    trace_sql: bool = True,
    extended: bool = False,
    mixed: bool = False,
) -> dict[str, Any]:
    ingest = []
    observe_latencies = []
    for version in range(1, revisions + 1):
        start = time.perf_counter()
        for index in range(count):
            start_observation = time.perf_counter()
            await memory.observe(_observation(index, version))
            observe_latencies.append(time.perf_counter() - start_observation)
        ingest.append(time.perf_counter() - start)
    start = time.perf_counter()
    processed = await memory.process(tenant_id="benchmark", as_of=_T + timedelta(days=1))
    process_seconds = time.perf_counter() - start
    process_latencies = [process_seconds]
    mixed_counts: dict[str, int] = {}
    if mixed:
        initial = sorted(
            await memory.list_semantic_memories(tenant_id="benchmark"),
            key=lambda item: item.subject_id or "",
        )[: min(6, count)]
        at = _T + timedelta(days=1, hours=1)
        for index, item in enumerate(initial):
            metadata: dict[str, Any] = {"conversation_id": f"correction-{index}"}
            if index % 2:
                metadata["semantic_withdrawals"] = [
                    {
                        "target_revision_key": item.revision_key,
                        "effective_at": at.isoformat(),
                        "authority": {"basis": "application_asserted"},
                    }
                ]
                content = f"Service {item.subject_id} deployment target was withdrawn."
            else:
                metadata["semantic_facts"] = [
                    {
                        "predicate": "deployment_environment",
                        "object_value": f"revised-{index}",
                        "cardinality": "one",
                        "authority": {"basis": "application_asserted"},
                        "correction": {
                            "replaces_revision_key": item.revision_key,
                            "effective_at": at.isoformat(),
                        },
                    }
                ]
                content = f"Service {item.subject_id} deployment target was corrected."
            start_observation = time.perf_counter()
            await memory.observe(
                ObservationInput(
                    tenant_id="benchmark",
                    subject_id=item.subject_id,
                    source_namespace="workload-correction",
                    source_record_id=f"correction-{index}",
                    source_version="1",
                    content=content,
                    observed_at=at,
                    metadata=metadata,
                )
            )
            observe_latencies.append(time.perf_counter() - start_observation)
        started = time.perf_counter()
        correction_result = await memory.process(tenant_id="benchmark", as_of=at)
        correction_process_seconds = time.perf_counter() - started
        process_latencies.append(correction_process_seconds)
        process_seconds += correction_process_seconds
        mixed_counts = {
            "corrections": correction_result.semantics.authoritative_corrections_applied,
            "withdrawals": correction_result.semantics.authoritative_withdrawals_applied,
        }
        if mixed_counts != {
            "corrections": (len(initial) + 1) // 2,
            "withdrawals": len(initial) // 2,
        }:
            raise RuntimeError(f"Mixed workload failed to apply its corrections: {mixed_counts}")
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
    if storage is not None and trace_sql:
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
    if storage is not None and trace_sql:
        await storage._run(lambda: storage._conn().set_trace_callback(None))
    extended_latencies: dict[str, list[float]] = {
        "prepare_context": [],
        "assess_source_memory": [],
        "inspect_semantic_slot": [],
        "record_context_use": [],
    }
    if extended:
        for n in range(repetitions):
            index = n % count
            now = _T + timedelta(days=2)
            query = f"service-{index:04d} deployment environment env-{index:04d}"
            start = time.perf_counter()
            context = await memory.prepare_context(query, tenant_id="benchmark", as_of=now)
            extended_latencies["prepare_context"].append(time.perf_counter() - start)
            start = time.perf_counter()
            await memory.assess_source_memory(context)
            extended_latencies["assess_source_memory"].append(time.perf_counter() - start)
            start = time.perf_counter()
            await memory.inspect_semantic_slot(
                SemanticSlotQuery(f"service-{index:04d}", "deployment_environment"),
                tenant_id="benchmark",
                as_of=now,
            )
            extended_latencies["inspect_semantic_slot"].append(time.perf_counter() - start)
            start = time.perf_counter()
            await memory.record_context_use(context, referenced_at=now, request_id=f"workload-{n}")
            extended_latencies["record_context_use"].append(time.perf_counter() - start)
    latency_metrics = {
        "observe": observe_latencies,
        "process": process_latencies,
        **extended_latencies,
    }
    operation_latencies = {
        name: {
            "count": len(samples),
            "median_ms": statistics.median(samples) * 1000,
            "p95_ms": _percentile(samples, 0.95) * 1000 if len(samples) >= 20 else None,
        }
        for name, samples in latency_metrics.items()
        if samples
    }
    return {
        "operation_latencies": operation_latencies,
        "mixed_operations": mixed_counts,
        "recall_sql_statements_total": len(statements)
        if storage is not None and trace_sql
        else None,
        "ingest_seconds_per_pass": ingest,
        "process_seconds": process_seconds,
        "recall_median_ms": statistics.median(latencies) * 1000,
        "recall_p95_ms": _percentile(latencies, 0.95) * 1000 if len(latencies) >= 20 else None,
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
    parser.add_argument(
        "--mixed", action="store_true", help="include authoritative corrections and withdrawals"
    )
    parser.add_argument(
        "--extended", action="store_true", help="time context, source, slot and use operations"
    )
    parser.add_argument(
        "--uninstrumented", action="store_true", help="disable tracemalloc and SQL tracing"
    )
    args = parser.parse_args()
    if min(args.observations, args.revisions, args.recalls) < 1:
        parser.error("workload arguments must be positive")
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    }
    if args.mixed:
        options["semantic_correction_config"] = SemanticCorrectionConfig(enabled=True)
    cpu_start = time.process_time()
    with tempfile.TemporaryDirectory(prefix="cogkura-workload-") as directory:
        path = Path(directory) / "memory.db"
        if not args.uninstrumented:
            tracemalloc.start()
        opening = time.perf_counter()
        owner = await EmbeddedMemory.open(path, **options)
        open_ms = (time.perf_counter() - opening) * 1000
        sqlite_result = await _run(
            owner.memory,
            args.observations,
            args.revisions,
            args.recalls,
            owner._storage,
            trace_sql=not args.uninstrumented,
            extended=args.extended,
            mixed=args.mixed,
        )
        sqlite_peak = tracemalloc.get_traced_memory()[1] if not args.uninstrumented else None
        sizes = {
            suffix: (
                Path(str(path) + suffix).stat().st_size if Path(str(path) + suffix).exists() else 0
            )
            for suffix in ("", "-wal", "-shm")
        }
        await owner.close()
        post_close_bytes = path.stat().st_size
        opening = time.perf_counter()
        reopened = await EmbeddedMemory.open(path, **options)
        reopen_ms = (time.perf_counter() - opening) * 1000
        await reopened.close()
        if not args.uninstrumented:
            tracemalloc.stop()
            tracemalloc.start()
        memory_result = await _run(
            Memory(**options),
            args.observations,
            args.revisions,
            args.recalls,
            extended=args.extended,
            mixed=args.mixed,
        )
        memory_peak = tracemalloc.get_traced_memory()[1] if not args.uninstrumented else None
        if not args.uninstrumented:
            tracemalloc.stop()
    parity = {
        "counts": sqlite_result["counts"] == memory_result["counts"],
        "mixed_operations": sqlite_result["mixed_operations"] == memory_result["mixed_operations"],
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
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    except ImportError:
        peak_rss_bytes = None
    report = {
        "workload": {
            "observations": args.observations,
            "revisions_per_observation": args.revisions,
            "recall_queries": args.recalls,
            "instrumented": not args.uninstrumented,
            "extended": args.extended,
            "mixed": args.mixed,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        },
        "process_resources": {
            "cpu_seconds_total": time.process_time() - cpu_start,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "sqlite": {
            **sqlite_result,
            "open_ms": open_ms,
            "reopen_ms": reopen_ms,
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
