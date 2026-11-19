"""Compare SQLite recall in fresh processes with subsequent same-process recalls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from run_sqlite_workload import _T, _run

import cogkura
from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    SemanticCorrectionConfig,
)


def options() -> dict[str, Any]:
    return {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }


async def child(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    owner = await EmbeddedMemory.open(path, **options())
    opened = time.perf_counter() - started
    results = []
    try:
        for _ in range(2):
            wall, cpu = time.perf_counter(), time.process_time()
            recalled = await owner.memory.recall(
                "service-0000 deployment environment env-0000",
                tenant_id="benchmark",
                as_of=_T + timedelta(days=2),
                limit=5,
            )
            results.append(
                {
                    "wall_seconds": time.perf_counter() - wall,
                    "process_cpu_seconds": time.process_time() - cpu,
                    "outputs": [
                        [item.memory_kind.value, item.memory.statement, round(item.score, 8)]
                        for item in recalled
                    ],
                }
            )
    finally:
        await owner.close()
    return {"open_seconds": opened, "recalls": results}


async def run(count: int, revisions: int, processes: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cogkura-fresh-process-") as directory:
        database = Path(directory) / "fixture.db"
        owner = await EmbeddedMemory.open(database, **options())
        try:
            fixture = await _run(owner.memory, count, revisions, 1, mixed=True, trace_sql=False)
        finally:
            await owner.close()
        samples = []
        for _ in range(processes):
            started = time.perf_counter()
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(database)],
                check=True,
                capture_output=True,
                text=True,
            )
            elapsed = time.perf_counter() - started
            sample = json.loads(result.stdout)
            sample["process_wall_seconds"] = elapsed
            expected = [list(item) for item in fixture["recall_outputs"][0]]
            if any(item["outputs"] != expected for item in sample["recalls"]):
                raise RuntimeError("Restarted recall output differs from the fixture process")
            samples.append(sample)
    root = Path(__file__).resolve().parents[1]
    return {
        "method": "fresh Python processes reopen a prebuilt SQLite file; OS cache is uncontrolled",
        "environment": {
            "commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "activation_source_sha256": hashlib.sha256(
                (root / "src/cogkura/algorithms/activation.py").read_bytes()
            ).hexdigest(),
            "profile_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_runtime_version": cogkura.__version__,
            "core_distribution_version": importlib.metadata.version("cogkura"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        },
        "workload": {"observations": count, "revisions": revisions, "processes": processes},
        "counts": fixture["counts"],
        "fixture_output": fixture["recall_outputs"][0],
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=25)
    parser.add_argument("--revisions", type=int, default=5)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.observations, args.revisions, args.processes) < 1:
        parser.error("workload arguments must be positive")
    result = asyncio.run(
        child(args.child) if args.child else run(args.observations, args.revisions, args.processes)
    )
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
