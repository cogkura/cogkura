"""Exercise successive scoped project updates and historical context."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from run_software_project_workload import _T, _TENANT, _observation, workload

import cogkura
from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    MemoryKind,
    SemanticCorrectionConfig,
)


async def exercise(memory: Memory, projects: int) -> dict[str, Any]:
    await workload(memory, projects, heterogeneous=True, relationships=5)
    subject = "project-000"

    async def sizes() -> dict[str, int]:
        result = {
            "observations": len(await memory._observation_store.list(tenant_id=_TENANT)),
            "episodes": len(await memory.list_episodes(tenant_id=_TENANT, include_inactive=True)),
            "semantic_memories": len(await memory.list_semantic_memories(tenant_id=_TENANT)),
            "semantic_revisions": len(await memory.list_semantic_revisions(tenant_id=_TENANT)),
            "relationships": len(await memory.list_entity_relationships(tenant_id=_TENANT)),
        }
        storage = getattr(memory._observation_store, "_storage", None)
        if storage is not None:

            def sqlite_sizes() -> dict[str, int]:
                conn = storage._conn()
                return {
                    "sqlite_logical_bytes": conn.execute("PRAGMA page_count").fetchone()[0]
                    * conn.execute("PRAGMA page_size").fetchone()[0],
                    "sqlite_episode_history_rows": conn.execute(
                        "SELECT count(*) FROM episode_history"
                    ).fetchone()[0],
                    "sqlite_semantic_history_rows": conn.execute(
                        "SELECT count(*) FROM semantic_history"
                    ).fetchone()[0],
                    "sqlite_revision_history_rows": conn.execute(
                        "SELECT count(*) FROM semantic_revision_history"
                    ).fetchone()[0],
                }

            result.update(await storage._run(sqlite_sizes))
        return result

    before = await sizes()
    stages = []
    predecessor = None
    for day, value in ((11, "workers_4"), (12, "workers_8"), (13, None)):
        metadata: dict[str, Any]
        if value is None:
            metadata = {
                "semantic_withdrawals": [
                    {
                        "target_revision_key": predecessor,
                        "effective_at": (_T + timedelta(days=day)).isoformat(),
                        "authority": {"basis": "application_asserted"},
                    }
                ]
            }
        else:
            fact: dict[str, Any] = {
                "predicate": "worker_configuration",
                "object_value": value,
                "cardinality": "one",
                "authority": {"basis": "application_asserted"},
            }
            if predecessor is not None:
                fact["correction"] = {
                    "replaces_revision_key": predecessor,
                    "effective_at": (_T + timedelta(days=day)).isoformat(),
                }
            metadata = {"semantic_facts": [fact]}
        started = time.perf_counter()
        await memory.observe(
            _observation(
                0, f"config-{day}", f"Worker configuration: {value or 'withdrawn'}.", day, metadata
            )
        )
        processed = await memory.process(
            tenant_id=_TENANT, subject_id=subject, as_of=_T + timedelta(days=day)
        )
        elapsed = time.perf_counter() - started
        if value is not None:
            current = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=subject)
            config = [
                item
                for item in current
                if item.predicate == "worker_configuration" and item.object_value == value
            ]
            if len(config) != 1:
                raise RuntimeError(f"Expected one current configuration for {value}")
            predecessor = config[0].revision_key
        elif processed.semantics.authoritative_withdrawals_applied != 1:
            raise RuntimeError("Configuration withdrawal was not applied")
        if day == 12 and processed.semantics.authoritative_corrections_applied != 1:
            raise RuntimeError("Configuration correction was not applied")
        stages.append(
            {
                "day": day,
                "value": value,
                "observe_and_process_seconds": elapsed,
                "sizes": await sizes(),
            }
        )

    queries = []
    for day, expected in ((11, "workers_4"), (12, "workers_8"), (13, None)):
        at = _T + timedelta(days=day, hours=12)
        args = {"tenant_id": _TENANT, "subject_id": subject, "as_of": at, "valid_at": at}
        start = time.perf_counter()
        recalled = await memory.recall("worker configuration", **args, limit=10)
        elapsed = time.perf_counter() - start
        values = [
            item.memory.object_value
            for item in recalled
            if item.memory_kind is MemoryKind.SEMANTIC
            and item.memory.predicate == "worker_configuration"
        ]
        if values != ([expected] if expected else []):
            raise RuntimeError(f"Historical configuration mismatch at day {day}: {values}")
        inspected = await memory.inspect_recall("worker configuration", **args, limit=10)
        context = await memory.prepare_context("worker configuration", **args)
        queries.append(
            {
                "day": day,
                "expected": expected,
                "semantic_values": values,
                "wall_seconds": elapsed,
                "considered_count": inspected.considered_count,
                "returned_count": len(inspected.returned),
                "configuration_text_count": sum(
                    "worker configuration" in item.memory.statement.lower() for item in recalled
                ),
                "other_text_count": sum(
                    "worker configuration" not in item.memory.statement.lower() for item in recalled
                ),
                "ranked_outputs": [
                    (item.memory_kind.value, item.memory.statement, round(item.score, 8))
                    for item in recalled
                ],
                "assessment_flags": [flag.value for flag in context.assessment.flags],
            }
        )
    return {"before": before, "stages": stages, "queries": queries}


async def run(backend: str, projects: int) -> dict[str, Any]:
    try:
        load_before = os.getloadavg()
    except (AttributeError, OSError):
        load_before = None
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-project-updates-") as directory:
        if backend == "sqlite":
            async with await EmbeddedMemory.open(
                Path(directory) / "project.db", **options
            ) as memory:
                output = await exercise(memory, projects)
        else:
            output = await exercise(Memory(**options), projects)
    direct_url = importlib.metadata.distribution("cogkura").read_text("direct_url.json")
    return {
        "backend": backend,
        "projects": projects,
        "pid": os.getpid(),
        "host_load_average_before_fixture": load_before,
        "python": platform.python_version(),
        "module": str(Path(cogkura.__file__).resolve()),
        "direct_url": json.loads(direct_url) if direct_url else None,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(
            Path(__file__).with_name("run_software_project_workload.py").read_bytes()
        ).hexdigest(),
        **output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("memory", "sqlite"), required=True)
    parser.add_argument("--projects", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.projects < 1:
        parser.error("projects must be positive")
    args.output.write_text(
        json.dumps(asyncio.run(run(args.backend, args.projects)), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
