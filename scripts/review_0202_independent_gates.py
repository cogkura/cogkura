"""Review the separately accepted 0.20.2 timing and state-integrity evidence.

This is candidate-specific. It does not reclassify snapshot timings as the
original protocol, validate release artifacts, or authorize publication.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from review_retrieval_repeatability import review as review_timing

_TIMING_COMMIT = "442b5d251beaa8682fcba1a9bc83af8e82742e4d"
_CANDIDATE = "64f85f78acd2046cafe0212c84b45c1d221affd4"
_URL = "https://github.com/cogkura/cogkura.git"
_LOGICAL = ("counts", "mixed_operations", "semantic_values", "historical_values", "recall_outputs")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(gzip.decompress(path.read_bytes()))


def validate_snapshot(
    report: dict[str, Any], baseline: dict[str, Any], *, backend: str, size: int
) -> None:
    workload = report["workload"]
    env = report["environment"]
    identity = env.get("core_direct_url") or {}
    vcs = identity.get("vcs_info") or {}
    count = 100 if size == 1000 else 101
    if (
        workload.get("backend") != backend
        or workload.get("observations") != size
        or workload.get("revisions") != 5
        or workload.get("recalls") != count
        or workload.get("profiled") is not False
        or workload.get("diagnostic") is not None
        or workload.get("context") is not False
        or workload.get("cognitive_state_audit") is not False
        or len(report["profiles"]) != count
    ):
        raise ValueError("snapshot workload differs from the checked fixture")
    if (
        identity.get("url") != _URL
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != _CANDIDATE
        or env.get("core_runtime_version") != "0.20.2"
        or env.get("core_distribution_version") != "0.20.2"
        or not env.get("core_module")
    ):
        raise ValueError("snapshot was not identified as the official candidate")
    if backend == "memory":
        if not workload.get("memory_state_fork") or workload.get("sqlite_state_clone"):
            raise ValueError("incorrect memory snapshot method")
        if report.get("memory_state_fork_matches") is not True:
            raise ValueError("memory state comparison did not pass")
    elif backend == "sqlite":
        if not workload.get("sqlite_state_clone") or workload.get("memory_state_fork"):
            raise ValueError("incorrect SQLite snapshot method")
        if report.get("sqlite_state_clone_matches") is not True:
            raise ValueError("SQLite state comparison did not pass")
        if (
            not isinstance(report.get("sqlite_state_clone_table_counts"), dict)
            or not report["sqlite_state_clone_table_counts"]
        ):
            raise ValueError("SQLite table coverage was not recorded")
    else:
        raise ValueError("unknown backend")
    if report.get("recorded_fixture_state_after_matches") is not True:
        raise ValueError("selected post-recall fixture state did not match")
    if any(report[key] != baseline[key] for key in _LOGICAL):
        raise ValueError("snapshot results differ from the original timing fixture")


def review(data: Path) -> dict[str, Any]:
    results = []
    for backend in ("memory", "sqlite"):
        for size in (100, 250, 1000):
            timing = [
                data / f"0.20.2-exact-442b5d2-{backend}-{size}-run{n}.json.gz" for n in range(1, 4)
            ]
            timing_reports = [_read(path) for path in timing]
            if any(item["environment"].get("commit") != _TIMING_COMMIT for item in timing_reports):
                raise ValueError("original timing commit differs")
            reviewed_timing = review_timing(timing)
            if (
                not reviewed_timing["all_runs_pass"]
                or not reviewed_timing["post_measurement_state_checked"]
            ):
                raise ValueError("original timing or selected-state checks failed")
            prefix = (
                "fork-memory"
                if backend == "memory"
                else "clone-final-sqlite"
                if size == 100
                else "clone-sqlite"
            )
            snapshots = [
                data / f"0.20.2-experimental-{prefix}-{size}-run{n}.json.gz" for n in range(1, 4)
            ]
            seen: set[int] = set()
            for path in snapshots:
                report = _read(path)
                validate_snapshot(report, timing_reports[0], backend=backend, size=size)
                pid = report["environment"]["pid"]
                if pid in seen:
                    raise ValueError("snapshot process IDs are not distinct")
                seen.add(pid)
            results.append(
                {
                    "backend": backend,
                    "observations": size,
                    "original_timing_passed": True,
                    "original_timing_samples": reviewed_timing["runs"],
                    "independent_state_runs_passed": len(snapshots),
                    "snapshot_pids": sorted(seen),
                    "timing_report_sha256": [
                        hashlib.sha256(path.read_bytes()).hexdigest() for path in timing
                    ],
                    "snapshot_report_sha256": [
                        hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshots
                    ],
                }
            )
    return {
        "candidate": _CANDIDATE,
        "original_timing_commit": _TIMING_COMMIT,
        "waiver": {
            "date": "2026-11-23",
            "scope": "Core 0.20.2",
            "decision": (
                "The user explicitly waived complete cognitive-state verification within "
                "each individual timed run, replacing it with separate timing and "
                "state-integrity gates."
            ),
        },
        "timing_and_integrity_gates_passed": True,
        "original_runs_contained_complete_per_run_state_checks": False,
        "snapshot_timings_used_as_original_protocol": False,
        "groups": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = review(args.data)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Original timing and independent state-integrity gates passed for all six groups.")


if __name__ == "__main__":
    main()
