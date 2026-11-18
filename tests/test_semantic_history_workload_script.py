"""The performance fixture creates actual semantic history on both backends."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_semantic_history_fixture_processes_each_revision(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_semantic_history_workload.py"
    reports = []
    for backend in ("memory", "sqlite"):
        output = tmp_path / f"{backend}.json"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--backend",
                backend,
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
        reports.append(json.loads(output.read_text()))
    for report in reports:
        assert report["semantic_revision_count"] == 9
        assert report["corrections_applied_per_pass"] == [0, 3, 3]
        values = {sample["query"]: sample["semantic_values"] for sample in report["observations"]}
        assert values == {
            "current": ["environment-0002"],
            "historical": ["environment-0000"],
            "before_correction_effective": ["environment-0001"],
            "at_correction_effective": ["environment-0002"],
            "before_correction_known": ["environment-0001"],
        }
    assert [r["ranked_outputs"] for r in reports[0]["observations"]] == [
        r["ranked_outputs"] for r in reports[1]["observations"]
    ]


def test_semantic_history_withdrawal_boundaries(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_semantic_history_workload.py"
    for backend in ("memory", "sqlite"):
        output = tmp_path / f"withdraw-{backend}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--backend",
                backend,
                "--subjects",
                "3",
                "--versions",
                "3",
                "--withdraw",
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        report = json.loads(output.read_text())
        assert completed.returncode == 0
        assert report["passed"] is True
        assert all(sample["status"] == "pass" for sample in report["observations"])
        values = {
            sample["query"]: sample["semantic_values"]
            for sample in report["observations"]
            if sample["status"] == "pass"
        }
        assert values["before_correction_known"] == ["environment-0001"]
        assert values["current"] == []
        assert values["before_withdrawal_effective"] == ["environment-0002"]
        assert values["at_withdrawal_effective"] == []
        assert values["before_withdrawal_known"] == ["environment-0002"]
