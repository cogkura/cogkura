"""Compare public historical reads after experimental timestamp-only row removal.

A disposable copy is altered; the input is never modified. This is a diagnostic,
not a safe compaction implementation or evidence for all historical workloads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sqlite3
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from run_software_project_workload import _T, _TENANT

from cogkura import ActivationConfig, EmbeddedMemory


def experimental_copy(source: Path, target: Path, subject: str) -> dict[str, int]:
    if not source.is_file() or target.exists():
        raise ValueError("Source must exist and target must be a new path")
    shutil.copyfile(source, target)
    removed: dict[str, int] = {}
    with sqlite3.connect(target) as conn:
        for table, key, field in (
            ("semantic_history", "memory_id", "memory"),
            ("semantic_revision_history", "revision_key", "revision"),
        ):
            if table == "semantic_history":
                eligible = {
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM semantic_memories WHERE tenant_id=? AND subject_id=?",
                        (_TENANT, subject),
                    )
                }
            else:
                eligible = {
                    row[0]
                    for row in conn.execute(
                        "SELECT r.revision_key FROM semantic_revisions r "
                        "JOIN semantic_memories m ON m.tenant_id=r.tenant_id "
                        "AND m.memory_key=r.memory_key WHERE m.tenant_id=? AND m.subject_id=?",
                        (_TENANT, subject),
                    )
                }
            previous: dict[str, dict] = {}
            to_remove = []
            for sequence, identity, payload in conn.execute(
                f"SELECT sequence,{key},snapshot FROM {table} ORDER BY sequence"
            ):
                if identity not in eligible:
                    continue
                if payload is None:
                    previous.pop(identity, None)
                    continue
                current = json.loads(payload)[field]
                old = previous.get(identity)
                if old is not None:
                    changed = {
                        k for k in old.keys() | current.keys() if old.get(k) != current.get(k)
                    }
                    if changed == {"updated_at"}:
                        to_remove.append((sequence,))
                previous[identity] = current
            conn.executemany(f"DELETE FROM {table} WHERE sequence=?", to_remove)
            removed[table] = len(to_remove)
    return removed


async def inspect(source: Path, altered: Path, subject: str) -> list[dict[str, object]]:
    output = []
    options = {"activation_config": ActivationConfig(retrieval_threshold=-10)}
    async with await EmbeddedMemory.open(source, **options) as original:
        async with await EmbeddedMemory.open(altered, **options) as pruned:
            for seconds in (-30, 0, 1, 5, 10, 20, 21, 30):
                at = _T + timedelta(days=15, minutes=50, seconds=seconds)
                for query in ("storage engine architecture", "unrelated revision probe"):
                    kwargs = {
                        "tenant_id": _TENANT,
                        "subject_id": subject,
                        "as_of": at,
                        "valid_at": at,
                    }
                    left = await original.recall(query, **kwargs, limit=20)
                    right = await pruned.recall(query, **kwargs, limit=20)
                    left_context = await original.prepare_context(query, **kwargs)
                    right_context = await pruned.prepare_context(query, **kwargs)
                    left_sources = await original.resolve_provenance(left_context)
                    right_sources = await pruned.resolve_provenance(right_context)
                    output.append(
                        {
                            "as_of": at.isoformat(),
                            "query": query,
                            "complete_recall_equal": left == right,
                            "recall_equal_ignoring_updated_at": [
                                replace(r, memory=replace(r.memory, updated_at=_T)) for r in left
                            ]
                            == [replace(r, memory=replace(r.memory, updated_at=_T)) for r in right],
                            "ranked_statements_scores_equal": [
                                (r.memory_kind.value, r.memory.statement, r.score) for r in left
                            ]
                            == [(r.memory_kind.value, r.memory.statement, r.score) for r in right],
                            "memory_updated_at_before": [
                                r.memory.updated_at.isoformat() for r in left
                            ],
                            "memory_updated_at_after": [
                                r.memory.updated_at.isoformat() for r in right
                            ],
                            "render_equal": left_context.render() == right_context.render(),
                            "assessment_equal": left_context.assessment == right_context.assessment,
                            "assessment_flags_equal": (
                                left_context.assessment.flags == right_context.assessment.flags
                            ),
                            "assessment_signals_equal": (
                                left_context.assessment.signals == right_context.assessment.signals
                            ),
                            "assessment_repr_equal_ignoring_updated_at": re.sub(
                                r"updated_at=datetime\.datetime\([^)]*\)",
                                "updated_at=<TIME>",
                                repr(left_context.assessment),
                            )
                            == re.sub(
                                r"updated_at=datetime\.datetime\([^)]*\)",
                                "updated_at=<TIME>",
                                repr(right_context.assessment),
                            ),
                            "provenance_equal": left_sources == right_sources,
                        }
                    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--copy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", default="project-049")
    args = parser.parse_args()
    removed = experimental_copy(args.source, args.copy, args.subject)
    cases = asyncio.run(inspect(args.source, args.copy, args.subject))
    result = {"removed_timestamp_only_rows": removed, "cases": cases}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "removed": removed,
                "cases": len(cases),
                "complete_recall_differences": sum(not c["complete_recall_equal"] for c in cases),
                "ranked_differences": sum(not c["ranked_statements_scores_equal"] for c in cases),
                "assessment_differences": sum(not c["assessment_equal"] for c in cases),
                "provenance_differences": sum(not c["provenance_equal"] for c in cases),
            }
        )
    )


if __name__ == "__main__":
    main()
