"""Compare append-only SQLite history against a baseline fixture copy."""

from __future__ import annotations

import argparse
import json
import sqlite3
from itertools import zip_longest
from pathlib import Path

_TABLES = {
    "episode_history": "JOIN episodes e ON e.tenant_id=h.tenant_id AND e.id=h.episode_id",
    "semantic_history": "JOIN semantic_memories e ON e.tenant_id=h.tenant_id AND e.id=h.memory_id",
    "semantic_revision_history": (
        "JOIN semantic_revisions r ON r.tenant_id=h.tenant_id AND r.revision_key=h.revision_key "
        "JOIN semantic_memories e ON e.tenant_id=r.tenant_id AND e.memory_key=r.memory_key"
    ),
}


def audit(baseline: Path, variant: Path) -> dict[str, object]:
    if not baseline.is_file() or not variant.is_file():
        raise FileNotFoundError("Both databases must exist")
    result: dict[str, object] = {}
    with sqlite3.connect(baseline) as before, sqlite3.connect(variant) as after:
        for table, joins in _TABLES.items():
            max_sequence = before.execute(
                f"SELECT COALESCE(MAX(sequence),0) FROM {table}"
            ).fetchone()[0]
            original = before.execute(f"SELECT * FROM {table} ORDER BY sequence")
            preserved = after.execute(
                f"SELECT * FROM {table} WHERE sequence<=? ORDER BY sequence", (max_sequence,)
            )
            preserved_equal = all(a == b for a, b in zip_longest(original, preserved))
            grouped = after.execute(
                f"SELECT e.subject_id,COUNT(*) FROM {table} h {joins} "
                "WHERE h.sequence>? GROUP BY e.subject_id ORDER BY e.subject_id",
                (max_sequence,),
            ).fetchall()
            total = after.execute(
                f"SELECT COUNT(*) FROM {table} WHERE sequence>?", (max_sequence,)
            ).fetchone()[0]
            changed_fields: dict[str, int] = {}
            if table in ("semantic_history", "semantic_revision_history"):
                key, field = (
                    ("memory_id", "memory")
                    if table == "semantic_history"
                    else ("revision_key", "revision")
                )
                previous: dict[str, dict] = {}
                for identity, snapshot in before.execute(
                    f"SELECT {key},snapshot FROM {table} ORDER BY sequence"
                ):
                    if snapshot is not None:
                        previous[identity] = json.loads(snapshot)[field]
                    else:
                        previous.pop(identity, None)
                for identity, snapshot in after.execute(
                    f"SELECT {key},snapshot FROM {table} WHERE sequence>? ORDER BY sequence",
                    (max_sequence,),
                ):
                    if snapshot is None:
                        previous.pop(identity, None)
                        continue
                    current = json.loads(snapshot)[field]
                    old = previous.get(identity)
                    if old is not None:
                        fields = sorted(
                            k for k in old.keys() | current.keys() if old.get(k) != current.get(k)
                        )
                        label = ",".join(fields) or "(none)"
                        changed_fields[label] = changed_fields.get(label, 0) + 1
                    previous[identity] = current
            result[table] = {
                "baseline_rows_preserved": preserved_equal,
                "new_rows": total,
                "new_rows_by_current_subject": dict(grouped),
                "unmatched_new_rows": total - sum(count for _, count in grouped),
                "changed_fields_for_previously_snapshotted_identities": changed_fields,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.baseline, args.variant), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
