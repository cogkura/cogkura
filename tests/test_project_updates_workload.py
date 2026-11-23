"""Project updates preserve their historical results across storage backends."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assess_project_updates import run  # noqa: E402
from profile_project_updates import profile  # noqa: E402


@pytest.mark.asyncio
async def test_project_updates_match_across_backends() -> None:
    left = await run("memory", 2)
    right = await run("sqlite", 2)
    assert left["before"] == {
        k: v for k, v in right["before"].items() if not k.startswith("sqlite_")
    }
    for a, b in zip(left["stages"], right["stages"], strict=True):
        assert (a["day"], a["value"], a["sizes"]) == (
            b["day"],
            b["value"],
            {k: v for k, v in b["sizes"].items() if not k.startswith("sqlite_")},
        )
    for a, b in zip(left["queries"], right["queries"], strict=True):
        assert {k: v for k, v in a.items() if k != "wall_seconds"} == {
            k: v for k, v in b.items() if k != "wall_seconds"
        }


@pytest.mark.asyncio
async def test_project_update_profile_keeps_expected_results() -> None:
    result = await profile("memory", 2)
    assert len(result["stages"]) == 3
    assert all(stage["recall_pipeline"]["calls"] == 1 for stage in result["stages"])
    assert [item["semantic_values"] for item in result["workload"]["queries"]] == [
        ["workers_4"],
        ["workers_8"],
        [],
    ]
