"""Measure a first recall in a fresh process, with an uncleared OS page cache."""

import asyncio
import json
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from cogkura import ActivationConfig, EmbeddedMemory, SemanticCorrectionConfig
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from scripts.run_sqlite_workload import _T, _run


def options():
    return {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }


async def recall(path):
    start = time.perf_counter()
    owner = await EmbeddedMemory.open(path, **options())
    open_ms = (time.perf_counter() - start) * 1000
    latencies = []
    outputs = []
    for _ in range(6):
        start = time.perf_counter()
        result = await owner.memory.recall(
            "service-0000 deployment environment env-0000",
            tenant_id="benchmark",
            as_of=_T + timedelta(days=2),
            limit=5,
        )
        latencies.append((time.perf_counter() - start) * 1000)
        outputs.append(
            [(item.memory_kind.value, item.memory.statement, item.score) for item in result]
        )
    await owner.close()
    return {
        "open_ms": open_ms,
        "first_recall_ms": latencies[0],
        "subsequent_recalls_ms": latencies[1:],
        "results_identical": all(result == outputs[0] for result in outputs),
    }


async def main():
    with tempfile.TemporaryDirectory(prefix="cogkura-cold-process-") as folder:
        path = Path(folder) / "memory.db"
        owner = await EmbeddedMemory.open(path, **options())
        await _run(owner.memory, 100, 5, 1, owner._storage, trace_sql=False, mixed=True)
        await owner.close()
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        child_total_ms = (time.perf_counter() - start) * 1000
        report = {
            "scope": "fresh process; operating-system page cache not cleared",
            "workload": {
                "observations": 100,
                "revisions_per_observation": 5,
                "corrections": 3,
                "withdrawals": 3,
                "query_repeats": 6,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
            },
            "child_total_ms": child_total_ms,
            **json.loads(result.stdout),
        }
        Path("/tmp/cogkura-0196-cold-process-perf.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report))


if len(sys.argv) == 3 and sys.argv[1] == "--child":
    print(json.dumps(asyncio.run(recall(Path(sys.argv[2])))))
else:
    asyncio.run(main())
