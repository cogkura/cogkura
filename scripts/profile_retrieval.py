"""Profile recalls in the unchanged 0.19.6 mixed workload.

Run in a fresh process: uv run python scripts/profile_retrieval.py --observations 100
Profiler timings are diagnostic, not uninstrumented latency measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import hashlib
import json
import platform
import pstats
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

from run_sqlite_workload import _run

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    SemanticCorrectionConfig,
)


def profile_rows(profile: cProfile.Profile) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), (
        primitive,
        calls,
        exclusive,
        inclusive,
        _callers,
    ) in stats.stats.items():
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}:{name}",
                "primitive_calls": primitive,
                "calls": calls,
                "exclusive_seconds": exclusive,
                "inclusive_seconds": inclusive,
            }
        )
    return sorted(rows, key=lambda row: row["exclusive_seconds"], reverse=True)


async def run(
    count: int, revisions: int, recalls: int, backend: str = "memory", *, profile: bool = True
) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    original = Memory.recall

    async def measured(self: Memory, *args: Any, **kwargs: Any) -> Any:
        profiler = cProfile.Profile() if profile else None
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        if profiler is not None:
            profiler.enable()
        try:
            return await original(self, *args, **kwargs)
        finally:
            if profiler is not None:
                profiler.disable()
            profiles.append(
                {
                    "wall_seconds": time.perf_counter() - wall_start,
                    "process_cpu_seconds": time.process_time() - cpu_start,
                    "functions": profile_rows(profiler) if profiler is not None else None,
                }
            )

    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-retrieval-profile-") as directory:
        owner = None
        if backend == "sqlite":
            owner = await EmbeddedMemory.open(Path(directory) / "memory.db", **options)
            memory = owner.memory
        else:
            memory = Memory(**options)
        try:
            with patch.object(Memory, "recall", measured):
                result = await _run(memory, count, revisions, recalls, mixed=True, trace_sql=False)
        finally:
            if owner is not None:
                await owner.close()
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "method": (
            "cProfile around each recall; timings include profiling overhead"
            if profile
            else "individual recalls without cProfile, tracemalloc or SQL tracing"
        ),
        "environment": {
            "commit": commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
            "activation_source_sha256": hashlib.sha256(
                (
                    Path(__file__).resolve().parents[1] / "src/cogkura/algorithms/activation.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "workload": {
            "observations": count,
            "revisions": revisions,
            "recalls": recalls,
            "backend": backend,
            "profiled": profile,
        },
        "counts": result["counts"],
        "mixed_operations": result["mixed_operations"],
        "recall_outputs": result["recall_outputs"],
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=25)
    parser.add_argument("--revisions", type=int, default=5)
    parser.add_argument("--recalls", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--uninstrumented", action="store_true")
    args = parser.parse_args()
    if min(args.observations, args.revisions, args.recalls) < 1:
        parser.error("workload arguments must be positive")
    report = asyncio.run(
        run(
            args.observations,
            args.revisions,
            args.recalls,
            args.backend,
            profile=not args.uninstrumented,
        )
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
