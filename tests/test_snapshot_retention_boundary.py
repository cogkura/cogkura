"""History reflects content retained at ingestion, without retroactive pruning."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cogkura import ComplementaryLearningSemanticConsolidator, EmbeddedMemory, ObservationInput
from cogkura.observations.retention import ObservationRetentionMode


@pytest.mark.asyncio
async def test_hash_only_new_content_does_not_rewrite_old_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "retention-history.db"
    tenant = "retention"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old_content = "old-retained-marker-3e812a"
    new_content = "new-hash-only-marker-8e91af"
    metadata_marker = "metadata-not-minimized-c512"
    metadata = {
        "conversation_id": "same-session",
        "semantic_facts": [
            {
                "predicate": "mode",
                "object_value": "public",
                "cardinality": "one",
                "authority": {"basis": "application_asserted"},
            }
        ],
    }
    consolidator = ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True
    )
    async with await EmbeddedMemory.open(path, semantic_consolidator=consolidator) as memory:
        await memory.observe(
            ObservationInput(
                tenant_id=tenant,
                subject_id="service",
                source_namespace="retention",
                source_record_id="record",
                source_version="1",
                content=old_content,
                observed_at=now,
                metadata=metadata,
            )
        )
        await memory.process(tenant_id=tenant, as_of=now)
    async with await EmbeddedMemory.open(
        path, semantic_consolidator=consolidator, retention_mode=ObservationRetentionMode.HASH_ONLY
    ) as memory:
        await memory.observe(
            ObservationInput(
                tenant_id=tenant,
                subject_id="service",
                source_namespace="retention",
                source_record_id="record",
                source_version="2",
                content=new_content,
                observed_at=now + timedelta(days=1),
                metadata={**metadata, "marker": metadata_marker},
            )
        )
        await memory.process(tenant_id=tenant, as_of=now + timedelta(days=1))
    with sqlite3.connect(path) as connection:
        old_payloads = [
            row[0]
            for row in connection.execute(
                "SELECT snapshot FROM episode_history WHERE tenant_id=?", (tenant,)
            )
            if row[0] is not None
        ]
        all_payloads = [
            row[0]
            for table in ("episode_history", "semantic_history", "semantic_revision_history")
            for row in connection.execute(
                f"SELECT snapshot FROM {table} WHERE tenant_id=?", (tenant,)
            )
            if row[0] is not None
        ]
        current_and_history = [
            row[0]
            for table in ("observations", "observation_revisions")
            for row in connection.execute(f"SELECT content FROM {table}")
            if row[0] is not None
        ]
        stored_metadata = connection.execute(
            "SELECT metadata FROM observations WHERE tenant_id=?", (tenant,)
        ).fetchone()[0]
    assert any(old_content in payload for payload in old_payloads)
    assert all(new_content not in payload for payload in (*all_payloads, *current_and_history))
    assert metadata_marker in stored_metadata
    async with await EmbeddedMemory.open(path) as memory:
        await memory.clear(tenant_id=tenant)
    with sqlite3.connect(path) as connection:
        for table in ("episode_history", "semantic_history", "semantic_revision_history"):
            assert (
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE tenant_id=?", (tenant,)
                ).fetchone()[0]
                == 0
            )
