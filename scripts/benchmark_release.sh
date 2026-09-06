#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run pytest tests/test_prepare_context_benchmarks.py -q --no-cov
