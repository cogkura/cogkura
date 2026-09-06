#!/usr/bin/env bash
# Reproducible release quality gates plus observational prepare_context benchmarks.
#
# Correctness: ruff, format, mypy, and pytest (including workload-count assertions).
# Timings: uv run python scripts/run_prepare_context_benchmark.py
#   writes results/benchmark-results.{json,md} (gitignored, machine-dependent).
# Historical checkpoint: docs/findings/0.15.12-performance-baseline.md
set -euo pipefail

cd "$(dirname "$0")/.."

uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python scripts/run_prepare_context_benchmark.py
