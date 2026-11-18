"""The diagnostic runner preserves the fixture across backends and measurement modes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_opt_in_diagnostics_keep_recall_results_and_report_measurements(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "profile_retrieval.py"
    reports = []
    for name, backend, mode in (
        ("plain", "memory", "--uninstrumented"),
        ("stages", "memory", "--stage-diagnostics"),
        ("allocations", "memory", "--allocations"),
        ("sql", "sqlite", "--sql-trace"),
    ):
        output = tmp_path / f"{name}.json"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--observations",
                "3",
                "--revisions",
                "2",
                "--backend",
                backend,
                mode,
                *(["--context"] if name == "stages" else []),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        reports.append(json.loads(output.read_text()))
    plain, stages, allocations, sql = reports
    assert all(report["recall_outputs"] == plain["recall_outputs"] for report in reports)
    assert plain["profiles"][0]["stages"] is None
    observed = stages["profiles"][0]["stages"]
    assert observed["episode_loading"]["returned_items"] > 0
    assert observed["candidate_construction"]["returned_items"] > 0
    assert observed["scoring_and_ranking"]["calls"] == 1
    assert stages["context_profiles"][0]["stages"]["scoring_and_ranking"]["calls"] == 1
    assert allocations["profiles"][0]["allocation"]["peak_traced_bytes"] > 0
    assert sql["profiles"][0]["sql_select_count"] > 0


@pytest.mark.postgres
def test_postgres_profile_runner_uses_disposable_database(tmp_path: Path) -> None:
    url = os.environ.get("COGKURA_POSTGRES_MEMORY_URL")
    if not url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL unset")
    script = Path(__file__).resolve().parents[1] / "scripts" / "profile_retrieval.py"
    output = tmp_path / "postgres.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            "postgres",
            "--postgres-url",
            url,
            "--observations",
            "3",
            "--revisions",
            "2",
            "--stage-diagnostics",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    report = json.loads(output.read_text())
    assert report["workload"]["backend"] == "postgres"
    assert report["mixed_operations"] == {"corrections": 2, "withdrawals": 1}
    assert report["profiles"][0]["stages"]["revision_loading"]["returned_items"] > 0
