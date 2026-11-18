"""Exercise successive semantic corrections and current/historical recall.

Run each backend separately in a fresh process. Timings are observations, not gates.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sqlite3
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    ObservationInput,
    SemanticCorrectionConfig,
)
from cogkura.algorithms import activation
from cogkura.exceptions import ValidationError
from cogkura.models import MemoryKind

_T = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = "history-workload"


async def run_workload(
    memory: Memory, count: int, revisions: int, *, withdraw: bool = False
) -> dict[str, Any]:
    previous: dict[str, str] = {}
    applied: list[int] = []
    for version in range(revisions):
        at = _T + timedelta(days=version)
        for index in range(count):
            subject = f"service-{index:04d}"
            fact: dict[str, Any] = {
                "predicate": "deployment_environment",
                "object_value": f"environment-{version:04d}",
                "cardinality": "one",
                "authority": {"basis": "application_asserted"},
            }
            if version:
                fact["correction"] = {
                    "replaces_revision_key": previous[subject],
                    "effective_at": at.isoformat(),
                }
            await memory.observe(
                ObservationInput(
                    tenant_id=_TENANT,
                    subject_id=subject,
                    source_namespace="history-workload",
                    source_record_id=f"{subject}-{version:04d}",
                    source_version="1",
                    content=f"{subject} deploys to environment-{version:04d}.",
                    observed_at=at,
                    metadata={
                        "conversation_id": f"{subject}-{version:04d}",
                        "semantic_facts": [fact],
                    },
                )
            )
        processed = await memory.process(tenant_id=_TENANT, as_of=at)
        corrections = processed.semantics.authoritative_corrections_applied
        if corrections != (count if version else 0):
            raise RuntimeError(f"Unexpected correction count at version {version}: {corrections}")
        applied.append(corrections)
        current = await memory.list_semantic_memories(tenant_id=_TENANT)
        previous = {item.subject_id: item.revision_key for item in current if item.subject_id}
        if len(previous) != count:
            raise RuntimeError("Expected one current semantic memory per subject")

    withdrawal_at = _T + timedelta(days=revisions)
    if withdraw:
        await memory.observe(
            ObservationInput(
                tenant_id=_TENANT,
                subject_id="service-0000",
                source_namespace="history-withdrawal",
                source_record_id="service-0000-withdrawal",
                source_version="1",
                content="The deployment environment is withdrawn.",
                observed_at=withdrawal_at,
                metadata={
                    "conversation_id": "service-0000-withdrawal",
                    "semantic_withdrawals": [
                        {
                            "target_revision_key": previous["service-0000"],
                            "effective_at": withdrawal_at.isoformat(),
                            "authority": {"basis": "application_asserted"},
                        }
                    ],
                },
            )
        )
        result = await memory.process(tenant_id=_TENANT, as_of=withdrawal_at)
        if result.semantics.authoritative_withdrawals_applied != 1:
            raise RuntimeError("Expected one authoritative withdrawal")

    now = withdrawal_at + timedelta(days=1)
    historical_at = _T + timedelta(hours=1)
    last = f"environment-{revisions - 1:04d}"
    cases: list[tuple[str, datetime, datetime | None, list[str]]] = [
        ("current", now, None, [] if withdraw else [last]),
        ("historical", now, historical_at, ["environment-0000"]),
    ]
    if revisions > 1:
        last_revision = _T + timedelta(days=revisions - 1)
        prior = f"environment-{revisions - 2:04d}"
        cases.extend(
            [
                (
                    "before_correction_effective",
                    now,
                    last_revision - timedelta(microseconds=1),
                    [prior],
                ),
                ("at_correction_effective", now, last_revision, [last]),
                (
                    "before_correction_known",
                    last_revision - timedelta(microseconds=1),
                    None,
                    [prior],
                ),
            ]
        )
    if withdraw:
        cases.extend(
            [
                (
                    "before_withdrawal_effective",
                    now,
                    withdrawal_at - timedelta(microseconds=1),
                    [last],
                ),
                ("at_withdrawal_effective", now, withdrawal_at, []),
                (
                    "before_withdrawal_known",
                    withdrawal_at - timedelta(microseconds=1),
                    None,
                    [last],
                ),
            ]
        )
    observations = []
    for name, as_of, valid_at, expected in cases:
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        try:
            results = await memory.recall(
                "service-0000 deployment environment",
                tenant_id=_TENANT,
                subject_id="service-0000",
                as_of=as_of,
                valid_at=valid_at,
                limit=10,
            )
        except ValidationError as exc:
            observations.append(
                {
                    "query": name,
                    "as_of": as_of.isoformat(),
                    "valid_at": valid_at.isoformat() if valid_at else None,
                    "expected_semantic_values": expected,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_seconds": time.perf_counter() - start_wall,
                    "process_cpu_seconds": time.process_time() - start_cpu,
                }
            )
            continue
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        values = [r.memory.object_value for r in results if r.memory_kind is MemoryKind.SEMANTIC]
        observations.append(
            {
                "query": name,
                "as_of": as_of.isoformat(),
                "valid_at": valid_at.isoformat() if valid_at else None,
                "wall_seconds": wall,
                "process_cpu_seconds": cpu,
                "semantic_values": values,
                "expected_semantic_values": expected,
                "status": "pass" if values == expected else "mismatch",
                "ranked_outputs": [
                    (r.memory_kind.value, r.memory.statement, round(r.score, 8)) for r in results
                ],
            }
        )
    history = await memory.list_semantic_revisions(tenant_id=_TENANT)
    return {
        "semantic_revision_count": len(history),
        "corrections_applied_per_pass": applied,
        "observations": observations,
        "passed": all(item["status"] == "pass" for item in observations),
    }


async def run(
    backend: str, count: int, revisions: int, *, withdraw: bool = False
) -> dict[str, Any]:
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-semantic-history-") as directory:
        owner = None
        if backend == "sqlite":
            owner = await EmbeddedMemory.open(Path(directory) / "memory.db", **options)
            memory = owner.memory
        else:
            memory = Memory(**options)
        try:
            result = await run_workload(memory, count, revisions, withdraw=withdraw)
        finally:
            if owner is not None:
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
            "activation_module": str(Path(activation.__file__).resolve()),
            "activation_source_sha256": hashlib.sha256(
                Path(activation.__file__).read_bytes()
            ).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
        },
        "workload": {
            "backend": backend,
            "subjects": count,
            "semantic_versions": revisions,
            "withdrawal": withdraw,
        },
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--versions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--withdraw", action="store_true")
    args = parser.parse_args()
    if min(args.subjects, args.versions) < 1:
        parser.error("subjects and versions must be positive")
    report = asyncio.run(run(args.backend, args.subjects, args.versions, withdraw=args.withdraw))
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    if not report["passed"]:
        raise SystemExit("Semantic history checks failed; see the report")


if __name__ == "__main__":
    main()
