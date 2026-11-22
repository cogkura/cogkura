"""Snapshot growth report counts stored payloads and the backup size."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_snapshot_growth_report(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_snapshot_growth.py"
    report = tmp_path / "sizes.json"
    subprocess.run(
        [sys.executable, str(script), "--observations", "2", "--output", str(report)],
        check=True,
        text=True,
    )
    result = json.loads(report.read_text())
    assert result["fixture"]["mixed_operations"] == {"corrections": 1, "withdrawals": 1}
    for item in result["snapshots"].values():
        assert item["rows"] >= item["available_snapshots"] > 0
        assert item["payload_bytes"] > 0
    assert result["database_bytes"] == result["page_count"] * result["page_size"]
    assert result["backup_bytes"] > 0


@pytest.mark.parametrize("scope_subject", [False, True])
def test_reinforcement_growth_reports_actual_semantic_reinforcement(
    tmp_path: Path, scope_subject: bool
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_snapshot_growth.py"
    report = tmp_path / "reinforcement.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--observations",
            "7",
            "--reinforcements",
            "2",
            *(["--scope-subject"] if scope_subject else []),
            "--output",
            str(report),
        ],
        check=True,
        text=True,
    )
    result = json.loads(report.read_text())
    assert result["reinforcement_scope"] == ("subject" if scope_subject else "tenant")
    samples = result["reinforcement_samples"]
    assert len(samples) == 2
    assert all(sample["semantic_reinforced"] == 1 for sample in samples)
    assert (
        samples[0]["history_counts"]["semantic_history"]
        > result["history_counts_before_reinforcement"]["semantic_history"]
    )
    assert (
        samples[1]["history_counts"]["semantic_history"]
        > samples[0]["history_counts"]["semantic_history"]
    )
