"""The observational workload reports instrumentation and small-sample limits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_extended_uninstrumented_workload_reports_each_operation(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sqlite_workload.py"
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--observations",
            "2",
            "--revisions",
            "2",
            "--recalls",
            "2",
            "--extended",
            "--uninstrumented",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["workload"]["instrumented"] is False
    assert report["sqlite"]["recall_sql_statements_total"] is None
    assert report["sqlite"]["peak_python_traced_bytes"] is None
    assert report["sqlite"]["recall_p95_ms"] is None
    assert all(report["logical_parity"].values())
    for backend in ("sqlite", "in_memory"):
        ops = report[backend]["operation_latencies"]
        assert set(ops) == {
            "observe",
            "prepare_context",
            "assess_source_memory",
            "inspect_semantic_slot",
            "record_context_use",
        }
        assert ops["observe"]["count"] == 4
        assert all(v["p95_ms"] is None for v in ops.values())
