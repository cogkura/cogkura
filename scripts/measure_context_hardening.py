"""Measure complete context operations on the mixed retrieval fixture.

Each invocation builds a fresh fixture. Read operations use a fixed cognitive
clock and run before any recorded use. Recorded-use samples mutate state and
are reported separately; they are not independent samples of unchanged state.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import sqlite3
import statistics
import subprocess
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from run_sqlite_workload import _T, _run

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    LearningConfig,
    Memory,
    RetrievalContext,
    SemanticCorrectionConfig,
    WorkingMemoryConfig,
)


def summary(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


async def measure(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    measured_paths = (Path(__file__), root / "scripts/run_sqlite_workload.py")
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in measured_paths]
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", "src/cogkura"]
    )
    if subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "src/cogkura"]
    ):
        raise RuntimeError("Untracked Core source cannot be identified by the Git diff")
    try:
        load = os.getloadavg()
    except (AttributeError, OSError):
        load = None
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
        "working_memory_config": WorkingMemoryConfig(candidate_pool_size=args.pool),
        "learning_config": LearningConfig(enabled=args.learning),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-context-hardening-") as directory:
        owner = None
        if args.backend == "sqlite":
            owner = await EmbeddedMemory.open(Path(directory) / "memory.db", **options)
            memory = owner.memory
        else:
            memory = Memory(**options)
        try:
            fixture = await _run(memory, args.observations, 5, 1, mixed=True, trace_sql=False)
            query = (
                "service-0000 deployment environment env-0000"
                if args.query == "narrow"
                else "deployment environment"
                if args.query == "broad"
                else "unrelated arctic wildlife"
            )
            now = _T + timedelta(days=2)
            shared = {"tenant_id": "benchmark", "as_of": now}
            contextual = (
                {
                    "goal": "Identify deployment environment",
                    "retrieval_context": RetrievalContext(
                        domain="deployment", activity="configuration review"
                    ),
                }
                if args.structured
                else {}
            )
            previous = None
            if args.previous:
                previous = await memory.select_working_memory(query, **shared, **contextual)
            samples: dict[str, list[float]] = {
                name: []
                for name in (
                    "recall",
                    "prepare_context",
                    "select_working_memory",
                    "assess_memory",
                    "record_context_use",
                )
            }
            contexts = []
            for _ in range(args.repetitions):
                if not args.prepare_only:
                    started = time.perf_counter()
                    await memory.recall(
                        query, **shared, retrieval_context=contextual.get("retrieval_context")
                    )
                    samples["recall"].append(time.perf_counter() - started)
                started = time.perf_counter()
                context = await memory.prepare_context(
                    query,
                    **shared,
                    **contextual,
                    previous=previous,
                    prompt_budget_tokens=args.budget,
                )
                samples["prepare_context"].append(time.perf_counter() - started)
                contexts.append(context)
                if not args.prepare_only:
                    started = time.perf_counter()
                    await memory.select_working_memory(
                        query,
                        **shared,
                        **contextual,
                        previous=previous,
                        prompt_budget_tokens=args.budget,
                    )
                    samples["select_working_memory"].append(time.perf_counter() - started)
                    started = time.perf_counter()
                    await memory.assess_memory(query, **shared, **contextual)
                    samples["assess_memory"].append(time.perf_counter() - started)
            identical = all(context == contexts[0] for context in contexts[1:])
            if not identical:
                raise RuntimeError("context changed during repeated fixed-state preparation")
            if not args.prepare_only:
                for index, context in enumerate(contexts):
                    started = time.perf_counter()
                    await memory.record_context_use(
                        context, referenced_at=now, request_id=f"context-hardening-{index}"
                    )
                    samples["record_context_use"].append(time.perf_counter() - started)
            if hashes != [hashlib.sha256(path.read_bytes()).hexdigest() for path in measured_paths]:
                raise RuntimeError("Measured scripts changed during the run")
            if (
                source_diff
                != subprocess.check_output(
                    ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", "src/cogkura"]
                )
                or commit
                != subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip()
            ):
                raise RuntimeError("Core source changed during the run")
            if subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "src/cogkura",
                ]
            ):
                raise RuntimeError("Untracked Core source appeared during the run")
            return {
                "workload": vars(args) | {"output": str(args.output) if args.output else None},
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "sqlite": sqlite3.sqlite_version,
                    "pid": os.getpid(),
                    "host_load_average_before_fixture": load,
                    "commit": commit,
                    "core_source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
                    "harness_sha256": hashes[0],
                    "fixture_sha256": hashes[1],
                },
                "fixture_counts": fixture["counts"],
                "mixed_operations": fixture["mixed_operations"],
                "repeated_contexts_equal": identical,
                "rendered_sha256": hashlib.sha256(contexts[0].render().encode()).hexdigest(),
                "selected_recall_count": len(contexts[0].recall_results),
                "timings": {name: summary(values) for name, values in samples.items() if values},
                "note": (
                    "Only fixed-state context preparation is timed."
                    if args.prepare_only
                    else "Use samples mutate state and follow all fixed-state read samples."
                ),
            }
        finally:
            if owner is not None:
                await owner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--backend", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--query", choices=("narrow", "broad", "irrelevant"), default="narrow")
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--previous", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="measure only prepare_context")
    parser.add_argument("--learning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.observations, args.repetitions, args.pool, args.budget) < 1:
        parser.error("counts, pool and budget must be positive")
    output = json.dumps(asyncio.run(measure(args)), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
