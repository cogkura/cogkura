"""The diagnostic runner preserves the fixture across backends and measurement modes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_profile_runner_backend_and_measurement_modes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "profile_retrieval.py"
    reports = []
    for backend, extra in (("memory", []), ("sqlite", ["--uninstrumented"])):
        output = tmp_path / f"{backend}.json"
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
                "--backend",
                backend,
                "--output",
                str(output),
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        reports.append(json.loads(output.read_text()))
    memory, sqlite = reports
    assert memory["workload"]["profiled"] is True
    assert sqlite["workload"]["profiled"] is False
    assert memory["counts"] == sqlite["counts"]
    assert (
        memory["mixed_operations"]
        == sqlite["mixed_operations"]
        == {
            "corrections": 1,
            "withdrawals": 1,
        }
    )
    assert memory["recall_outputs"] == sqlite["recall_outputs"]
    assert len(memory["profiles"]) == len(sqlite["profiles"]) == 2
    assert all(sample["functions"] for sample in memory["profiles"])
    assert all(sample["functions"] is None for sample in sqlite["profiles"])
