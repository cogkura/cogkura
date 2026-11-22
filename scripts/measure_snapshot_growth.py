"""Measure SQLite snapshot rows and file sizes for the existing mixed fixture.

This is an observational fixture measurement, not a retention policy or a
prediction of production growth. Run one fixture size per fresh process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from typing import Any

from run_sqlite_workload import _T, _run

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    ObservationInput,
    SemanticCorrectionConfig,
)

_TABLES = {
    "episode_history": "episode_id",
    "semantic_history": "memory_id",
    "semantic_revision_history": "revision_key",
}


async def measure(
    count: int, reinforcements: int = 0, *, scope_subject: bool = False
) -> dict[str, Any]:
    if reinforcements and count < 7:
        raise ValueError("reinforcement needs seven observations to select an uncorrected memory")
    with tempfile.TemporaryDirectory(prefix="cogkura-snapshot-size-") as directory:
        path = Path(directory) / "memory.db"
        owner = await EmbeddedMemory.open(
            path,
            activation_config=ActivationConfig(retrieval_threshold=-10),
            semantic_consolidator=ComplementaryLearningSemanticConsolidator(
                allow_authoritative_single_episode=True
            ),
            semantic_correction_config=SemanticCorrectionConfig(enabled=True),
        )
        reinforcement_samples = []

        def history_counts() -> dict[str, int]:
            connection = owner._storage._conn()
            return {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in _TABLES
            }

        try:
            fixture = await _run(owner.memory, count, 5, 1, mixed=True, trace_sql=False)
            baseline_history_counts = await owner._storage._run(history_counts)
            target_index = min(6, count - 1)
            for index in range(reinforcements):
                at = _T + timedelta(days=2, seconds=index)
                started = time.perf_counter()
                await owner.memory.observe(
                    ObservationInput(
                        tenant_id="benchmark",
                        subject_id=f"service-{target_index:04d}",
                        source_namespace="reinforcement-size-probe",
                        source_record_id=f"reinforcement-{index}",
                        content=(
                            f"Service {target_index:04d} deployment target "
                            f"is env-{target_index:04d}."
                        ),
                        observed_at=at,
                        metadata={
                            "conversation_id": f"reinforcement-{index}",
                            "semantic_facts": [
                                {
                                    "predicate": "deployment_environment",
                                    "object_value": f"env-{target_index:04d}",
                                    "cardinality": "one",
                                    "authority": {"basis": "application_asserted"},
                                }
                            ],
                        },
                    )
                )
                processed = await owner.memory.process(
                    tenant_id="benchmark",
                    subject_id=f"service-{target_index:04d}" if scope_subject else None,
                    as_of=at,
                )
                reinforcement_samples.append(
                    {
                        "observe_and_process_seconds": time.perf_counter() - started,
                        "semantic_reinforced": processed.semantics.reinforced,
                        "semantic_updated": processed.semantics.updated,
                        "history_counts": await owner._storage._run(history_counts),
                    }
                )
        finally:
            await owner.close()
        backup = Path(directory) / "backup.db"
        with closing(sqlite3.connect(path)) as connection:
            rows = {}
            for table, key in _TABLES.items():
                total, available, payload_bytes = connection.execute(
                    "SELECT COUNT(*), COUNT(snapshot), "
                    f"COALESCE(SUM(LENGTH(CAST(snapshot AS BLOB))), 0) FROM {table}"
                ).fetchone()
                per_memory = [
                    count
                    for (count,) in connection.execute(
                        f"SELECT COUNT(*) FROM {table} GROUP BY tenant_id, {key}"
                    )
                ]
                rows[table] = {
                    "rows": total,
                    "available_snapshots": available,
                    "payload_bytes": payload_bytes,
                    "mean_payload_bytes": payload_bytes / available if available else None,
                    "records_per_identity": {
                        "count": len(per_memory),
                        "min": min(per_memory) if per_memory else None,
                        "max": max(per_memory) if per_memory else None,
                        "mean": sum(per_memory) / len(per_memory) if per_memory else None,
                    },
                }
            relation_rows = connection.execute(
                "SELECT COUNT(*) FROM semantic_relation_history"
            ).fetchone()[0]
            page_count = connection.execute("PRAGMA page_count").fetchone()[0]
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            try:
                physical = dict(
                    connection.execute(
                        "SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY name"
                    ).fetchall()
                )
            except sqlite3.OperationalError:
                physical = None
            with closing(sqlite3.connect(backup)) as destination:
                connection.backup(destination)
        return {
            "fixture": {
                "observations": count,
                "revisions": 5,
                "counts": fixture["counts"],
                "mixed_operations": fixture["mixed_operations"],
            },
            "environment": {
                "python": platform.python_version(),
                "sqlite": sqlite3.sqlite_version,
                "platform": platform.platform(),
            },
            "snapshots": rows,
            "history_counts_before_reinforcement": baseline_history_counts,
            "reinforcement_samples": reinforcement_samples,
            "reinforcement_scope": "subject" if scope_subject else "tenant",
            "semantic_relation_history_rows": relation_rows,
            "database_bytes": path.stat().st_size,
            "backup_bytes": backup.stat().st_size,
            "page_count": page_count,
            "page_size": page_size,
            "dbstat_object_bytes": physical,
            "note": "Snapshot payload totals exclude indexes, pages, WAL and backup overhead.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=100)
    parser.add_argument("--reinforcements", type=int, default=0)
    parser.add_argument("--scope-subject", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.observations < 1 or args.reinforcements < 0:
        parser.error("observations must be positive and reinforcements nonnegative")
    if args.reinforcements and args.observations < 7:
        parser.error("reinforcement requires at least seven observations")
    result = (
        json.dumps(
            asyncio.run(
                measure(args.observations, args.reinforcements, scope_subject=args.scope_subject)
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
