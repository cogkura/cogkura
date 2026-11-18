"""The history workload reports its checks and the known baseline failure honestly."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_history_workload_exercises_versions(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_semantic_history_workload.py"
    output = tmp_path / "history.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--subjects",
            "3",
            "--versions",
            "3",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    report = json.loads(output.read_text())
    assert report["semantic_revision_count"] == 9
    assert report["corrections_applied_per_pass"] == [0, 3, 3]
    assert report["passed"] is True


def test_history_workload_reports_baseline_withdrawal_failure(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_semantic_history_workload.py"
    output = tmp_path / "history-withdrawal.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--subjects",
            "3",
            "--versions",
            "3",
            "--withdraw",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    report = json.loads(output.read_text())
    assert result.returncode != 0
    assert report["passed"] is False
    failure = next(
        row for row in report["observations"] if row["query"] == "before_correction_known"
    )
    assert failure["status"] == "error"
    assert failure["error_type"] == "ValidationError"
