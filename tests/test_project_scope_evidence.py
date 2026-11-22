"""Scoped project questions retain the fixture's independently named evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from probe_project_rank_depth import measure as measure_rank_depth  # noqa: E402
from probe_project_subject_scope import measure  # noqa: E402


@pytest.mark.asyncio
async def test_project_scoped_evidence_and_backend_parity() -> None:
    memory = await measure("memory")
    sqlite = await measure("sqlite")
    memory_queries = {item["name"]: item for item in memory["queries"]}
    sqlite_queries = {item["name"]: item for item in sqlite["queries"]}
    expected = {
        "current_architecture": "project-000 storage engine sqlite_local_postgres_hosted",
        "historical_architecture": "project-000 storage engine postgres_primary",
        "incident": "An expired database connection pool caused intermittent recall failures",
        "migration": "deployment pipeline now verifies migration completion",
        "broad": "Revised architecture: SQLite for local agents, PostgreSQL",
    }
    assert memory_queries.keys() == sqlite_queries.keys() == expected.keys()
    for name, evidence in expected.items():
        outputs = memory_queries[name]["ranked_outputs"]
        assert outputs == sqlite_queries[name]["ranked_outputs"]
        assert any(evidence in statement for _, statement, _ in outputs), name


@pytest.mark.asyncio
async def test_project_unscoped_deep_results_include_named_evidence_and_match() -> None:
    memory = await measure_rank_depth("memory")
    sqlite = await measure_rank_depth("sqlite")
    for left, right in zip(memory["queries"], sqlite["queries"], strict=True):
        assert left["name"] == right["name"]
        assert left["ranked_outputs"] == right["ranked_outputs"]
        if "expected_evidence" in left:
            assert left["expected_rank"] is not None
            assert left["expected_rank"] == right["expected_rank"]
            assert left["expected_evidence"] in left["ranked_outputs"][left["expected_rank"] - 1][1]
