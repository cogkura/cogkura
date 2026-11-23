"""Release review uses all observations and requires independent matching runs."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_retrieval_repeatability.py"
_SPEC = importlib.util.spec_from_file_location("review_retrieval_repeatability", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _reports(tmp_path: Path, *, count: int = 1000) -> list[Path]:
    environment = dict.fromkeys(
        (
            "commit",
            "core_source_diff_sha256",
            "python",
            "core_runtime_version",
            "core_distribution_version",
            "platform",
            "sqlite",
            "profiler_source_sha256",
            "diagnostics_source_sha256",
            "activation_source_sha256",
            "workload_source_sha256",
        ),
        "same",
    )
    paths = []
    for pid in range(1, 4):
        report = {
            "environment": {
                **environment,
                "pid": pid,
                "host_load_average_before_fixture": [1, 1, 1],
            },
            "workload": {
                "observations": count,
                "revisions": 5,
                "recalls": 100 if count == 1000 else 101,
                "backend": "memory",
                "profiled": False,
                "diagnostic": None,
                "context": False,
            },
            "counts": {},
            "mixed_operations": {},
            "semantic_values": [],
            "historical_values": [],
            "recall_outputs": [],
            "process_peak_rss": {},
            "profiles": [{"wall_seconds": 1.0}] * (100 if count == 1000 else 101),
        }
        path = tmp_path / f"{pid}.json"
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_review_includes_first_sample_in_nearest_rank_p95(tmp_path: Path) -> None:
    paths = _reports(tmp_path)
    report = json.loads(paths[0].read_text())
    for item in report["profiles"][:6]:
        item["wall_seconds"] = 2.1
    paths[0].write_text(json.dumps(report))
    result = _MODULE.review(paths)
    assert result["runs"][0]["p95_including_first_seconds"] == 2.1
    assert not result["all_runs_pass"]
    assert result["runs"][1]["passes_target"]
    assert result["post_measurement_state_checked"] is False


def test_review_rejects_changed_output_and_duplicate_process(tmp_path: Path) -> None:
    paths = _reports(tmp_path, count=100)
    report = json.loads(paths[1].read_text())
    report["semantic_values"] = ["different"]
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="logical output"):
        _MODULE.review(paths)
    report["semantic_values"] = []
    report["environment"]["pid"] = 1
    paths[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="process IDs"):
        _MODULE.review(paths)


def test_review_tracks_post_measurement_state(tmp_path: Path) -> None:
    paths = _reports(tmp_path)
    for path in paths:
        report = json.loads(path.read_text())
        report["recorded_fixture_state_after_matches"] = True
        path.write_text(json.dumps(report))
    assert _MODULE.review(paths)["post_measurement_state_checked"] is True
    report["recorded_fixture_state_after_matches"] = False
    paths[-1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="state changed"):
        _MODULE.review(paths)


@pytest.mark.parametrize(
    "mode", ["cognitive_state_audit", "sqlite_state_clone", "memory_state_fork"]
)
def test_review_rejects_experimental_timings(tmp_path: Path, mode: str) -> None:
    paths = _reports(tmp_path)
    for path in paths:
        report = json.loads(path.read_text())
        report["workload"][mode] = True
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="uninstrumented"):
        _MODULE.review(paths)


def test_review_accepts_compressed_raw_reports(tmp_path: Path) -> None:
    paths = _reports(tmp_path)
    compressed = []
    for path in paths:
        target = path.with_suffix(".json.gz")
        target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        compressed.append(target)
    assert _MODULE.review(compressed)["all_runs_pass"] is True
