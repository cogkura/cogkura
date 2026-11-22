"""Review raw fresh-process retrieval reports without discarding failed runs.

Run the unchanged mixed workload separately with profile_retrieval.py, using
--uninstrumented, five revisions, and 101 recalls for 100/250 or 100 for 1000.
Pass the three JSON files from one backend and workload to this script.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path
from typing import Any


def review(paths: list[Path]) -> dict[str, Any]:
    reports = [
        json.loads(
            gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        )
        for path in paths
    ]
    if len(reports) < 3:
        raise ValueError("at least three independent process reports are required")
    identity = reports[0]["workload"]
    environment = reports[0]["environment"]
    if identity["observations"] not in (100, 250, 1000):
        raise ValueError("release protocol covers 100, 250 and 1000 observations")
    expected_count = 100 if identity["observations"] == 1000 else 101
    if identity["revisions"] != 5 or identity["recalls"] != expected_count:
        raise ValueError("incorrect revisions or recall sample count")
    if identity["backend"] not in ("memory", "sqlite", "postgres"):
        raise ValueError("unknown backend")
    if (
        identity["profiled"]
        or identity["diagnostic"] is not None
        or identity["context"]
        or identity.get("cognitive_state_audit", False)
    ):
        raise ValueError("release recall measurements must be uninstrumented and recall-only")
    keys = (
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
    )
    pids: set[int] = set()
    baseline = {
        key: reports[0][key]
        for key in (
            "counts",
            "mixed_operations",
            "semantic_values",
            "historical_values",
            "recall_outputs",
        )
    }
    runs = []
    for path, report in zip(paths, reports, strict=True):
        if report["workload"] != identity or any(
            report["environment"][key] != environment[key] for key in keys
        ):
            raise ValueError(f"inconsistent workload or source/environment identity: {path}")
        pid = report["environment"]["pid"]
        if pid in pids:
            raise ValueError("process IDs must be distinct")
        pids.add(pid)
        if any(report[key] != value for key, value in baseline.items()):
            raise ValueError(f"logical output or state differs: {path}")
        if report.get("recorded_fixture_state_after_matches") is False:
            raise ValueError(f"recorded fixture state changed after measurement: {path}")
        profiles = report["profiles"]
        if len(profiles) != expected_count:
            raise ValueError(f"incorrect number of latency observations: {path}")
        samples = [item["wall_seconds"] for item in profiles]
        warm_median = statistics.median(samples[1:])
        p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
        metric = p95 if identity["observations"] == 1000 else warm_median
        limit = 2.0 if identity["observations"] == 1000 else 1.0
        runs.append(
            {
                "file": str(path),
                "pid": pid,
                "first_seconds": samples[0],
                "warm_median_seconds": warm_median,
                "p95_including_first_seconds": p95,
                "target_seconds": limit if identity["backend"] != "postgres" else None,
                "passes_target": metric < limit if identity["backend"] != "postgres" else None,
                "process_peak_rss": report["process_peak_rss"],
                "host_load_average_before_fixture": report["environment"][
                    "host_load_average_before_fixture"
                ],
            }
        )
    return {
        "workload": identity,
        "runs": runs,
        "logical_results_match": True,
        "post_measurement_state_checked": all(
            report.get("recorded_fixture_state_after_matches") is True for report in reports
        ),
        "all_runs_pass": all(run["passes_target"] for run in runs)
        if identity["backend"] != "postgres"
        else None,
        "note": "Host load is contextual evidence, not proof that a run is invalid.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        result = review(args.reports)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["all_runs_pass"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
