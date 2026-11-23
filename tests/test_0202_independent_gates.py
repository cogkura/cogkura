"""The waiver does not turn snapshot timings into original timing evidence."""

from __future__ import annotations

import copy
import gzip
import importlib
import json
from pathlib import Path

import pytest


def test_retained_reports_satisfy_separate_candidate_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    reviewer = importlib.import_module("review_0202_independent_gates")
    result = reviewer.review(root / "docs/findings/data")
    assert result["timing_and_integrity_gates_passed"] is True
    assert result["original_runs_contained_complete_per_run_state_checks"] is False
    assert result["snapshot_timings_used_as_original_protocol"] is False
    assert len(result["groups"]) == 6
    assert all(group["independent_state_runs_passed"] == 3 for group in result["groups"])


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_snapshot_review_requires_exact_candidate_and_unchanged_state(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    reviewer = importlib.import_module("review_0202_independent_gates")
    data = root / "docs/findings/data"
    baseline = json.loads(
        gzip.decompress((data / f"0.20.2-exact-442b5d2-{backend}-100-run1.json.gz").read_bytes())
    )
    prefix = "fork-memory" if backend == "memory" else "clone-final-sqlite"
    observed = json.loads(
        gzip.decompress((data / f"0.20.2-experimental-{prefix}-100-run1.json.gz").read_bytes())
    )
    reviewer.validate_snapshot(observed, baseline, backend=backend, size=100)

    wrong_candidate = copy.deepcopy(observed)
    wrong_candidate["environment"]["core_direct_url"]["vcs_info"]["commit_id"] = "0" * 40
    with pytest.raises(ValueError, match="official candidate"):
        reviewer.validate_snapshot(wrong_candidate, baseline, backend=backend, size=100)

    changed = copy.deepcopy(observed)
    changed[
        "memory_state_fork_matches" if backend == "memory" else "sqlite_state_clone_matches"
    ] = False
    with pytest.raises(ValueError, match="state comparison"):
        reviewer.validate_snapshot(changed, baseline, backend=backend, size=100)

    different_results = copy.deepcopy(observed)
    different_results["recall_outputs"] = []
    with pytest.raises(ValueError, match="differ"):
        reviewer.validate_snapshot(different_results, baseline, backend=backend, size=100)
