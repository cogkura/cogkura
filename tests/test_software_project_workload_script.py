"""The project workload runs the same logical fixture across local backends."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_software_project_workload_revises_architecture(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_software_project_workload.py"
    reports = []
    for backend in ("memory", "sqlite"):
        output = tmp_path / f"{backend}.json"
        subprocess.run(
            [sys.executable, str(script), "--backend", backend, "--output", str(output)],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        reports.append(json.loads(output.read_text()))
    for report in reports:
        assert report["episodes"] == 9
        assert report["semantic_revisions"] == 6
        assert report["corrections_applied"] == 1
        results = {item["name"]: item["ranked_outputs"] for item in report["queries"]}
        assert any(
            "sqlite_local_postgres_hosted" in row[1] for row in results["current_architecture"]
        )
        assert any("postgres_primary" in row[1] for row in results["historical_architecture"])
    assert [q["ranked_outputs"] for q in reports[0]["queries"]] == [
        q["ranked_outputs"] for q in reports[1]["queries"]
    ]


def test_project_workload_varies_relationships_independently(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_software_project_workload.py"
    counts = []
    for relationships in (0, 2):
        output = tmp_path / f"relationships-{relationships}.json"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--projects",
                "2",
                "--heterogeneous",
                "--relationships-per-project",
                str(relationships),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        report = json.loads(output.read_text())
        counts.append(report["entity_relationships"])
        assert report["episodes"] == 18
        assert report["corrections_applied"] == 2
    assert counts == [0, 4]
