"""Measure scoped context on a prebuilt Agent-scale SQLite fixture.

Run independent processes sequentially on a closed copy of the fixture. This
diagnostic records every sample and does not itself approve a release candidate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import statistics
import time
from datetime import timedelta
from pathlib import Path

try:
    import resource
except ModuleNotFoundError:  # Windows has no resource module.
    resource = None  # type: ignore[assignment]

from run_software_project_workload import _T, _TENANT

import cogkura
from cogkura import (
    ActivationConfig,
    EmbeddedMemory,
    RetrievalContext,
    WorkingMemoryConfig,
)

_STATE_TABLES = (
    "observations",
    "episodes",
    "episode_history",
    "semantic_memories",
    "semantic_history",
    "semantic_revisions",
    "semantic_revision_history",
    "semantic_relations",
    "entity_relationships",
    "activation_references",
    "learning_events",
    "inhibition_traces",
)


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in _STATE_TABLES
    }


async def measure(database: Path, samples: int) -> dict[str, object]:
    if not database.is_file() or samples < 1:
        raise ValueError("A prebuilt database and at least one sample are required")
    with database.open("rb") as source:
        database_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    package_root = Path(cogkura.__file__).resolve().parent
    package_hash = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        package_hash.update(str(path.relative_to(package_root)).encode())
        package_hash.update(path.read_bytes())
    try:
        load_before = os.getloadavg()
    except (AttributeError, OSError):
        load_before = None
    with sqlite3.connect(database) as conn:
        counts = dict(
            conn.execute(
                "SELECT subject_id, count(*) FROM observations "
                "WHERE tenant_id=? GROUP BY subject_id",
                (_TENANT,),
            ).fetchall()
        )
        storage_before = (
            conn.execute("PRAGMA page_count").fetchone()[0]
            * conn.execute("PRAGMA page_size").fetchone()[0]
        )
        rows_before = _counts(conn)
    query = "project-000 recall deployment incident database connection failures"
    at = _T + timedelta(days=16)
    kwargs = {
        "tenant_id": _TENANT,
        "subject_id": "project-000",
        "as_of": at,
        "goal": "Identify the cause and response for the deployment incident",
        "retrieval_context": RetrievalContext(domain="deployment", activity="incident review"),
    }
    evidence: dict[str, object] = {}
    async with await EmbeddedMemory.open(
        database,
        activation_config=ActivationConfig(retrieval_threshold=-10),
        working_memory_config=WorkingMemoryConfig(candidate_pool_size=50),
    ) as memory:
        for name, text, valid_at, expected in (
            (
                "incident",
                query,
                None,
                "An expired database connection pool caused intermittent recall failures",
            ),
            (
                "migration",
                "project-000 migration rollout deployment pipeline",
                None,
                "deployment pipeline now verifies migration completion",
            ),
            (
                "historical",
                "project-000 storage engine architecture",
                _T + timedelta(days=5),
                "project-000 storage engine postgres_primary",
            ),
        ):
            results = await memory.recall(
                text,
                tenant_id=_TENANT,
                subject_id="project-000",
                as_of=at,
                valid_at=valid_at,
                limit=10,
            )
            statements = [item.memory.statement for item in results]
            evidence[name] = {
                "expected": expected,
                "found": any(expected in s for s in statements),
                "statements": statements,
            }
        previous = await memory.select_working_memory(query, **kwargs)
        times = []
        reference = None
        cpu_start = time.process_time()
        for _ in range(samples):
            start = time.perf_counter()
            context = await memory.prepare_context(
                query,
                **kwargs,
                previous=previous,
                prompt_budget_tokens=2048,
            )
            times.append(time.perf_counter() - start)
            if reference is None:
                reference = context
            elif context != reference:
                raise RuntimeError("Fixed-state prepared contexts changed across samples")
        sample_cpu_seconds = time.process_time() - cpu_start
        assert reference is not None
        rendered_hash = hashlib.sha256(reference.render().encode()).hexdigest()
        assessment = [flag.value for flag in reference.assessment.flags]
    with sqlite3.connect(database) as conn:
        storage_after = (
            conn.execute("PRAGMA page_count").fetchone()[0]
            * conn.execute("PRAGMA page_size").fetchone()[0]
        )
        rows_after = _counts(conn)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "load_average_before": load_before,
        "module": str(Path(cogkura.__file__).resolve()),
        "distribution_version": importlib.metadata.version("cogkura"),
        "package_source_sha256": package_hash.hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "database_sha256_before": database_sha256,
        "subject_counts": counts,
        "state_table_counts_before": rows_before,
        "state_table_counts_after": rows_after,
        "samples_seconds": times,
        "sample_process_cpu_seconds": sample_cpu_seconds,
        "process_max_rss_native_units": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
        ),
        "median_seconds": statistics.median(times),
        "p95_seconds": sorted(times)[math.ceil(0.95 * len(times)) - 1],
        "rendered_sha256": rendered_hash,
        "assessment_flags": assessment,
        "assessment_repr": repr(reference.assessment),
        "selected_count": len(reference.items),
        "evidence": evidence,
        "logical_database_bytes_before": storage_before,
        "logical_database_bytes_after": storage_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(measure(args.database, args.samples))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "median_seconds": result["median_seconds"],
                "p95_seconds": result["p95_seconds"],
                "evidence": {k: v["found"] for k, v in result["evidence"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
