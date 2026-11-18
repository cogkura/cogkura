"""Historical correction metadata remains recoverable after SQLite reopen."""

from __future__ import annotations

import runpy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    MemoryKind,
    SemanticCorrectionConfig,
    SemanticRevisionViewState,
    SemanticSlotQuery,
)
from cogkura.algorithms import corrections


@pytest.mark.asyncio
async def test_legacy_closed_correction_survives_sqlite_reopen(tmp_path: Path) -> None:
    path = tmp_path / "history.db"
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_semantic_history_workload.py"
    run_workload = runpy.run_path(str(script))["run_workload"]
    original = corrections._closed_snapshot

    def legacy_closed_snapshot(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        if kwargs["closed_by"]["kind"] == "withdrawal":
            snapshot.pop("knowledge_at", None)
            snapshot.pop("correction", None)
        return snapshot

    owner = await EmbeddedMemory.open(path, **options)
    try:
        with patch.object(corrections, "_closed_snapshot", legacy_closed_snapshot):
            result = await run_workload(owner.memory, 1, 2, withdraw=True)
        assert result["passed"]
        revisions = await owner.memory.list_semantic_revisions(tenant_id="history-workload")
        original_key = next(
            r.revision_key
            for r in revisions
            if r.snapshot.get("object_value") == "environment-0000"
        )
        closed = next(
            r for r in revisions if r.snapshot.get("closed_by", {}).get("kind") == "withdrawal"
        )
        assert "correction" not in closed.snapshot
        assert "knowledge_at" not in closed.snapshot
    finally:
        await owner.close()

    reopened = await EmbeddedMemory.open(path, **options)
    try:
        before = datetime(2026, 1, 2, tzinfo=UTC) - timedelta(microseconds=1)
        results = await reopened.memory.recall(
            "service-0000 deployment environment",
            tenant_id="history-workload",
            subject_id="service-0000",
            as_of=before,
        )
        assert [r.memory.object_value for r in results if r.memory_kind is MemoryKind.SEMANTIC] == [
            "environment-0000"
        ]
        assert all("environment-0001" not in r.memory.statement for r in results)
        slot = await reopened.memory.inspect_semantic_slot(
            SemanticSlotQuery("service-0000", "deployment_environment"),
            tenant_id="history-workload",
            subject_id="service-0000",
            as_of=before,
        )
        assert [
            r.revision_key
            for r in slot.revisions
            if r.view_state is SemanticRevisionViewState.VISIBLE_CURRENT
        ] == [original_key]
        revisions = await reopened.memory.list_semantic_revisions(tenant_id="history-workload")
        closed = next(
            r for r in revisions if r.snapshot.get("closed_by", {}).get("kind") == "withdrawal"
        )
        assert "correction" not in closed.snapshot
    finally:
        await reopened.close()
