#!/usr/bin/env python3
"""Measure prepare_context latency and write observational benchmark artifacts.

Reproducible release command (also invoked by scripts/benchmark_release.sh):

    uv run python scripts/run_prepare_context_benchmark.py

Writes machine-readable output to results/ (gitignored). Timings are
observational evidence only — correctness is enforced by pytest.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.prepare_context import (  # noqa: E402
    environment_metadata,
    print_table,
    run_all_benchmarks,
    write_artifacts,
)


async def _main() -> None:
    results = await run_all_benchmarks()
    output_dir = ROOT / "results"
    write_artifacts(results, output_dir)
    print_table(results, environment_metadata())


if __name__ == "__main__":
    asyncio.run(_main())
