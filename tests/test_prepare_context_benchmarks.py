"""Deterministic prepare_context workload assertions (0.15.12)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.prepare_context import (  # noqa: E402
    build_fixture,
    measure_workload_counts,
    memory_factory,
)

_FIXTURE_SCALES = ("small", "medium", "large")
_EXPECTED_MEMORY_COUNTS = {"small": 8, "medium": 40, "large": 120}


@pytest.mark.asyncio
@pytest.mark.parametrize("scale", _FIXTURE_SCALES)
async def test_prepare_context_workload_counts_are_stable(scale: str) -> None:
    memory = memory_factory()
    memory_count, relationship_count = await build_fixture(memory, scale=scale)
    stats = await measure_workload_counts(memory)

    assert memory_count == _EXPECTED_MEMORY_COUNTS[scale]
    assert relationship_count > 0
    assert stats.candidate_count > 0
    assert stats.returned_count > 0
    assert stats.chunk_count > 0
    assert stats.token_count > 0
    assert stats.chunk_count <= stats.candidate_count
