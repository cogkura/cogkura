"""The context workload times whole operations and separates recorded use."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_fixed_state_context_workload(backend: str, tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_context_hardening.py"
    report_path = tmp_path / f"{backend}.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            backend,
            "--observations",
            "2",
            "--repetitions",
            "2",
            "--structured",
            "--previous",
            "--no-learning",
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    assert report["repeated_contexts_equal"]
    assert report["mixed_operations"] == {"corrections": 1, "withdrawals": 1}
    assert set(report["timings"]) == {
        "recall",
        "prepare_context",
        "select_working_memory",
        "assess_memory",
        "record_context_use",
    }
    assert all(len(item["samples_seconds"]) == 2 for item in report["timings"].values())
    assert "mutate state" in report["note"]


def test_prepare_only_reports_independent_samples(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_context_hardening.py"
    path = tmp_path / "prepare.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--observations",
            "2",
            "--repetitions",
            "2",
            "--prepare-only",
            "--output",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(path.read_text())
    assert set(report["timings"]) == {"prepare_context"}
    assert len(report["timings"]["prepare_context"]["samples_seconds"]) == 2
    assert report["repeated_contexts_equal"]
    assert report["environment"]["pid"] > 0
    assert len(report["environment"]["harness_sha256"]) == 64
