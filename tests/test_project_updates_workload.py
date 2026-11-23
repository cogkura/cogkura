"""Project updates preserve their historical results across storage backends."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assess_project_updates import run  # noqa: E402


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
