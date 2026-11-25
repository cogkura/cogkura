"""Exercise the Agent-scale fixture and measurement drivers at small scale."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from audit_unaffected_subject_history import audit  # noqa: E402
from measure_agent_scale import measure  # noqa: E402
from prepare_agent_scale_fixture import build  # noqa: E402
from probe_unrelated_relationship_scale import probe as probe_relationships  # noqa: E402
from probe_unrelated_revision_scale import probe as probe_revisions  # noqa: E402
from probe_unrelated_semantic_scale import probe  # noqa: E402
from probe_unrelated_support_scale import probe as probe_support  # noqa: E402


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
    revision_database = tmp_path / "revision.db"
    shutil.copyfile(database, revision_database)
    relationship_database = tmp_path / "relationship.db"
    shutil.copyfile(database, relationship_database)
    support_database = tmp_path / "support.db"
    shutil.copyfile(database, support_database)
    baseline_database = tmp_path / "baseline.db"
    shutil.copyfile(database, baseline_database)
    variant = await probe(database, facts=3, samples=2, subject_index=1)
    assert variant["added_semantic_memories"] == 3
    assert variant["subject_counts"] == {"project-000": 12, "project-001": 12}
    assert all(item["found"] for item in variant["evidence"].values())
    revised = await probe_revisions(revision_database, corrections=1, samples=2, subject_index=1)
    assert revised["corrections_applied"] == 1
    assert revised["withdrawals_applied"] == 1
    assert revised["subject_counts"] == {"project-000": 12, "project-001": 12}
    assert all(item["found"] for item in revised["evidence"].values())
    related = await probe_relationships(relationship_database, edges=3, samples=2, subject_index=1)
    assert related["added_relationships"] == 3
    assert related["subject_counts"] == {"project-000": 12, "project-001": 12}
    assert all(item["found"] for item in related["evidence"].values())
    supported = await probe_support(support_database, supports=2, samples=2, subject_index=1)
    assert supported["support_count"] == 2
    assert supported["observation_evidence_count"] == 2
    assert supported["subject_counts"] == {"project-000": 12, "project-001": 12}
    assert all(item["found"] for item in supported["evidence"].values())
    for altered in (database, revision_database, relationship_database, support_database):
        report = audit(baseline_database, altered)
        for table in report.values():
            assert table["baseline_rows_preserved"]
            assert table["unmatched_new_rows"] == 0
            assert set(table["new_rows_by_current_subject"]) <= {"project-001"}
