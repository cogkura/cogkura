"""Small, deterministic software-project retrieval workload.

Synthetic examples support investigation, not an Agent performance claim.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cogkura
from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    ObservationInput,
    SemanticCorrectionConfig,
)

_T = datetime(2026, 2, 1, tzinfo=UTC)
_TENANT = "project-workload"

# Events are deliberately distinct: decisions, operational incidents, migrations,
# rollout evidence and guidance. Structured facts supplement, not replace, prose.
_EVENTS: tuple[tuple[str, str, str | None, str | None], ...] = (
    (
        "adr-storage",
        "Architecture decision: use PostgreSQL as the primary project memory store.",
        "storage_engine",
        "postgres_primary",
    ),
    (
        "adr-migration",
        "Database migrations run in the deployment pipeline before API rollout.",
        "migration_strategy",
        "deployment_pipeline",
    ),
    (
        "incident",
        (
            "An expired database connection pool caused intermittent recall failures during "
            "deployment; restarting workers restored service."
        ),
        None,
        None,
    ),
    (
        "postmortem",
        (
            "The incident review asks for connection health checks and bounded retries "
            "before a worker is ready."
        ),
        "incident_action",
        "connection_health_checks",
    ),
    (
        "security",
        (
            "Security review requires tenant-scoped access and prohibits sharing memory rows "
            "across customers."
        ),
        "tenant_isolation",
        "tenant_scoped",
    ),
    (
        "migration",
        (
            "The storage migration added reversible schema changes and exercised recovery "
            "on a copy of the database."
        ),
        None,
        None,
    ),
    (
        "rollout",
        (
            "The deployment pipeline now verifies migration completion before starting workers; "
            "a failed migration stops rollout."
        ),
        None,
        None,
    ),
    (
        "integration",
        (
            "Agent integration needs an embedded local database while the hosted service "
            "still uses PostgreSQL."
        ),
        "agent_requirement",
        "embedded_local_and_hosted",
    ),
)

_INCIDENTS = (
    _EVENTS[2][1],
    "A connection pool exhausted its sessions after an import; worker restarts restored recall.",
    "A network partition left stale pool connections, delaying the agent's first recall.",
    "A database failover left a worker pool using closed connections until restart.",
    "An idle worker retained expired database sessions and failed rollout readiness checks.",
)


def _observation(
    index: int, event: str, text: str, day: int, metadata: dict[str, Any]
) -> ObservationInput:
    subject = f"project-{index:03d}"
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=subject,
        source_namespace="project-notes",
        source_record_id=f"{subject}-{event}",
        source_version="1",
        content=f"{subject}: {text}",
        observed_at=_T + timedelta(days=day),
        metadata={"conversation_id": f"{subject}-{event}", **metadata},
    )


async def workload(
    memory: Memory, projects: int, *, heterogeneous: bool = False, relationships: int = 1
) -> dict[str, Any]:
    for index in range(projects):
        for day, (event, content, predicate, value) in enumerate(_EVENTS):
            if heterogeneous and event == "incident":
                content = _INCIDENTS[index % len(_INCIDENTS)]
            metadata: dict[str, Any] = {"entity_ids": [f"project-{index:03d}", "memory-store"]}
            if predicate:
                metadata["semantic_facts"] = [
                    {
                        "predicate": predicate,
                        "object_value": value,
                        "cardinality": "one",
                        "authority": {"basis": "application_asserted"},
                    }
                ]
            if event == "integration":
                metadata["relationships"] = [
                    {
                        "source_entity_id": (
                            "agent-client"
                            if not heterogeneous and relationships == 1
                            else f"project-{index:03d}-client-{edge:03d}"
                        ),
                        "relation_type": "related_to",
                        "target_entity_id": "memory-store",
                        "provenance": "integration-design",
                    }
                    for edge in range(relationships)
                ]
            await memory.observe(_observation(index, event, content, day, metadata))
    await memory.process(tenant_id=_TENANT, as_of=_T + timedelta(days=8))
    memories = await memory.list_semantic_memories(tenant_id=_TENANT)
    previous = {
        item.subject_id: item.revision_key
        for item in memories
        if item.predicate == "storage_engine" and item.subject_id
    }
    if len(previous) != projects:
        raise RuntimeError("Missing a storage architecture decision")
    correction_at = _T + timedelta(days=10)
    for index in range(projects):
        subject = f"project-{index:03d}"
        await memory.observe(
            _observation(
                index,
                "adr-storage-update",
                "Revised architecture: SQLite for local agents, PostgreSQL for hosted deployments.",
                10,
                {
                    "entity_ids": [subject, "memory-store"],
                    "semantic_facts": [
                        {
                            "predicate": "storage_engine",
                            "object_value": "sqlite_local_postgres_hosted",
                            "cardinality": "one",
                            "authority": {"basis": "application_asserted"},
                            "correction": {
                                "replaces_revision_key": previous[subject],
                                "effective_at": correction_at.isoformat(),
                            },
                        }
                    ],
                },
            )
        )
    processed = await memory.process(tenant_id=_TENANT, as_of=correction_at)
    if processed.semantics.authoritative_corrections_applied != projects:
        raise RuntimeError("Expected one applied architecture correction per project")
    queries = (
        ("current_architecture", "project-000 storage engine architecture", None),
        (
            "historical_architecture",
            "project-000 storage engine architecture",
            _T + timedelta(days=5),
        ),
        ("incident", "project-000 recall deployment incident database connection failures", None),
        ("migration", "project-000 migration rollout deployment pipeline", None),
        ("broad", "agent local memory architecture database integration", None),
    )
    results = []
    for name, query, valid_at in queries:
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        retrieved = await memory.recall(
            query, tenant_id=_TENANT, as_of=_T + timedelta(days=11), valid_at=valid_at, limit=10
        )
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        results.append(
            {
                "name": name,
                "query": query,
                "valid_at": valid_at.isoformat() if valid_at else None,
                "wall_seconds": wall,
                "process_cpu_seconds": cpu,
                "ranked_outputs": [
                    (r.memory_kind.value, r.memory.statement, round(r.score, 8)) for r in retrieved
                ],
            }
        )
    return {
        "episodes": len(await memory.list_episodes(tenant_id=_TENANT)),
        "semantic_revisions": len(await memory.list_semantic_revisions(tenant_id=_TENANT)),
        "entity_relationships": len(await memory.list_entity_relationships(tenant_id=_TENANT)),
        "corrections_applied": processed.semantics.authoritative_corrections_applied,
        "queries": results,
    }


async def run(
    backend: str, projects: int, *, heterogeneous: bool = False, relationships: int = 1
) -> dict[str, Any]:
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-project-workload-") as directory:
        owner = None
        if backend == "sqlite":
            owner = await EmbeddedMemory.open(Path(directory) / "memory.db", **options)
            memory = owner.memory
        else:
            memory = Memory(**options)
        try:
            output = await workload(
                memory, projects, heterogeneous=heterogeneous, relationships=relationships
            )
        finally:
            if owner:
                await owner.close()
    root = Path(__file__).resolve().parents[1]
    return {
        "environment": {
            "commit": subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
            "activation_source_sha256": hashlib.sha256(
                (root / "src/cogkura/algorithms/activation.py").read_bytes()
            ).hexdigest(),
            "workload_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "core_runtime_version": cogkura.__version__,
            "core_distribution_version": importlib.metadata.version("cogkura"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        },
        "workload": {
            "backend": backend,
            "projects": projects,
            "heterogeneous": heterogeneous,
            "relationships_per_project": relationships,
        },
        **output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--projects", type=int, default=1)
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--relationships-per-project", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.projects < 1 or args.relationships_per_project < 0:
        parser.error("projects must be positive and relationships nonnegative")
    output = (
        json.dumps(
            asyncio.run(
                run(
                    args.backend,
                    args.projects,
                    heterogeneous=args.heterogeneous,
                    relationships=args.relationships_per_project,
                )
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
