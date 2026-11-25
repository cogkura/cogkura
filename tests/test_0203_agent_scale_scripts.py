"""Exercise the Agent-scale fixture and measurement drivers at small scale."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from measure_agent_scale import measure  # noqa: E402
from prepare_agent_scale_fixture import build  # noqa: E402


@pytest.mark.asyncio
async def test_agent_scale_fixture_and_fixed_state_driver(tmp_path: Path) -> None:
    database = tmp_path / "fixture.db"
    await build(database, projects=2, observations=12)
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT subject_id, count(*) FROM observations GROUP BY subject_id ORDER BY subject_id"
        ).fetchall() == [("project-000", 12), ("project-001", 12)]
    report = await measure(database, samples=2)
    assert report["subject_counts"] == {"project-000": 12, "project-001": 12}
    assert len(report["samples_seconds"]) == 2
    assert all(item["found"] for item in report["evidence"].values())
    assert report["logical_database_bytes_before"] == report["logical_database_bytes_after"]
    with pytest.raises(FileExistsError):
        await build(database, projects=2, observations=12)
