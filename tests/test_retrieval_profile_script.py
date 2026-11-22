"""The diagnostic runner preserves the fixture across backends and measurement modes."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cognitive_audit_checks_state_without_claiming_benchmark_timings(
    tmp_path: Path, backend: str
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "profile_retrieval.py"
    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--observations",
            "3",
            "--revisions",
            "2",
            "--recalls",
            "2",
            "--backend",
            backend,
            "--uninstrumented",
            "--cognitive-state-audit",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    report = json.loads(output.read_text())
    assert report["workload"]["cognitive_state_audit"] is True
    assert report["cognitive_state_audit_matches"] is True


@pytest.mark.asyncio
async def test_cognitive_audit_detects_recorded_access(monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    profile = importlib.import_module("profile_retrieval")
    from types import MappingProxyType

    nested = {"metadata": {"items": [1]}}
    frozen = profile.freeze_state(MappingProxyType(nested))
    nested["metadata"]["items"].append(2)
    assert frozen != profile.freeze_state(MappingProxyType(nested))
    workload = importlib.import_module("run_sqlite_workload")
    from datetime import timedelta

    from cogkura import Memory

    memory = Memory()
    await memory.observe(workload._observation(0, 1))
    await memory.process(tenant_id="benchmark", as_of=workload._T + timedelta(days=1))
    before = await profile.cognitive_state(memory, 1)
    results = await memory.recall(
        "service-0000 deployment environment",
        tenant_id="benchmark",
        as_of=workload._T + timedelta(days=2),
    )
    assert results
    await memory.record_access(
        results, tenant_id="benchmark", referenced_at=workload._T + timedelta(days=2)
    )
    after = await profile.cognitive_state(memory, 1)
    assert before["activation"] != after["activation"]


@pytest.mark.asyncio
async def test_cognitive_audit_rejects_a_mutating_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    profile = importlib.import_module("profile_retrieval")
    from cogkura import Memory

    original = Memory.recall

    async def mutating_recall(self: Memory, *args: object, **kwargs: object) -> object:
        results = await original(self, *args, **kwargs)
        await self.record_access(results, tenant_id="benchmark", referenced_at=kwargs["as_of"])
        return results

    monkeypatch.setattr(Memory, "recall", mutating_recall)
    with pytest.raises(RuntimeError, match="Cognitive state changed during recall.*activation"):
        await profile.run(2, 2, 1, profile=False, cognitive_state_audit=True)


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
    assert memory["recorded_fixture_state_after_matches"] is True
    assert sqlite["recorded_fixture_state_after_matches"] is True
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
    assert observed["scoring_and_ranking"]["input_candidates"] > 0
    assert (
        observed["candidate_scoring"]["calls"]
        == observed["scoring_and_ranking"]["input_candidates"]
    )
    assert observed["semantic_relevance"]["calls"] == 1
    assert observed["historical_projection"]["input_revisions"] > 0
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
