import asyncio
import json
import platform
import sqlite3
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from cogkura import ActivationConfig, EmbeddedMemory, SemanticCorrectionConfig
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from scripts.run_sqlite_workload import _T, _run


async def main():
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-reopen-perf-") as folder:
        path = Path(folder) / "memory.db"
        owner = await EmbeddedMemory.open(path, **options)
        await _run(owner.memory, 100, 5, 1, owner._storage, trace_sql=False, mixed=True)
        await owner.close()
        start = time.perf_counter()
        owner = await EmbeddedMemory.open(path, **options)
        open_ms = (time.perf_counter() - start) * 1000
        latency = []
        outputs = []
        for _ in range(6):
            start = time.perf_counter()
            result = await owner.memory.recall(
                "service-0000 deployment environment env-0000",
                tenant_id="benchmark",
                as_of=_T + timedelta(days=2),
                limit=5,
            )
            latency.append((time.perf_counter() - start) * 1000)
            outputs.append([(x.memory_kind.value, x.memory.statement, x.score) for x in result])
        await owner.close()
        report = {
            "scope": (
                "first recall after close and reopen in same process; "
                "operating-system page cache not cleared"
            ),
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
            "reopen_ms": open_ms,
            "first_recall_ms": latency[0],
            "subsequent_recalls_ms": latency[1:],
            "results_identical": all(x == outputs[0] for x in outputs),
        }
        Path("/tmp/cogkura-0196-reopen-perf.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


asyncio.run(main())
