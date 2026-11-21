"""The shared snapshot format preserves the complete derived memory state."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from cogkura import ComplementaryLearningSemanticConsolidator, Memory, ObservationInput
from cogkura.exceptions import StorageError
from cogkura.storage.historical_snapshots import (
    decode_memory_snapshot,
    encode_memory_snapshot,
    encode_revision_snapshot_or_none,
    select_revision_snapshot,
)


async def test_round_trip_derived_memories() -> None:
    at = datetime(2026, 1, 1, tzinfo=UTC)
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id="tenant",
            subject_id="service",
            source_namespace="test",
            source_record_id="one",
            content="The database is PostgreSQL.",
            observed_at=at,
            metadata={
                "entity_ids": ["database"],
                "semantic_facts": [
                    {
                        "predicate": "database",
                        "object_value": "postgresql",
                        "cardinality": "one",
                        "authority": {"basis": "application_asserted"},
                    }
                ],
            },
        )
    )
    await memory.encode_episodes(tenant_id="tenant", as_of=at)
    await memory.consolidate_semantics(tenant_id="tenant", as_of=at)
    episodes = await memory.list_episodes(tenant_id="tenant")
    semantics = await memory.list_semantic_memories(tenant_id="tenant")
    revisions = await memory.list_semantic_revisions(tenant_id="tenant")
    assert len(episodes) == len(semantics) == 1
    assert episodes[0].evidence and semantics[0].derivations
    for original in (*episodes, *semantics):
        assert decode_memory_snapshot(encode_memory_snapshot(original)) == original
    assert len(revisions) == 1
    payload = encode_revision_snapshot_or_none(revisions[0])
    assert payload is not None
    assert select_revision_snapshot([(at, payload)], as_of=at) == revisions[0]
    assert select_revision_snapshot([(at, "invalid")], as_of=at) is None

    nested = {"values": ["before"]}
    captured = encode_memory_snapshot(replace(episodes[0], metadata=nested))
    nested["values"].append("after")
    assert decode_memory_snapshot(captured).metadata == {"values": ["before"]}
    with pytest.raises(TypeError, match="Unsupported snapshot metadata"):
        encode_memory_snapshot(replace(episodes[0], metadata={"value": (1, 2)}))


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "{}",
        '{"version": 2, "kind": "episode", "memory": {}}',
        '{"version": true, "kind": "episode", "memory": {}}',
    ],
)
def test_invalid_or_future_snapshot_fails_closed(payload: str) -> None:
    with pytest.raises(StorageError, match="snapshot"):
        decode_memory_snapshot(payload)
