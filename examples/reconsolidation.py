"""Reconsolidation example: temporal supersession and historical recall."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 6, 1, tzinfo=UTC)
_T2 = datetime(2027, 1, 1, tzinfo=UTC)


async def main() -> None:
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        )
    )
    tenant_id = "company_123"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="customer_42",
            source_namespace="chat.messages",
            source_record_id="message_1",
            event_type="message",
            content="Preferred vendor was Acme through mid-year.",
            observed_at=_T0,
            metadata={
                "conversation_id": "conv-1",
                "semantic_facts": [
                    {
                        "predicate": "preferred_vendor",
                        "object_value": "Acme",
                        "object_entity_id": "acme",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "valid_from": _T0.isoformat(),
                        "valid_until": _T1.isoformat(),
                    }
                ],
            },
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="customer_42",
            source_namespace="chat.messages",
            source_record_id="message_2",
            event_type="message",
            content="Preferred vendor switched to Beta.",
            observed_at=_T1,
            metadata={
                "conversation_id": "conv-2",
                "semantic_facts": [
                    {
                        "predicate": "preferred_vendor",
                        "object_value": "Beta",
                        "object_entity_id": "beta",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "valid_from": _T1.isoformat(),
                        "valid_until": _T2.isoformat(),
                    }
                ],
            },
        )
    )

    await memory.encode_episodes(tenant_id=tenant_id)
    result = await memory.consolidate_semantics(tenant_id=tenant_id)
    print("Consolidation:", result)

    current = await memory.list_semantic_memories(tenant_id=tenant_id)
    historical = await memory.list_semantic_memories(tenant_id=tenant_id, valid_at=_T0)
    revisions = await memory.list_semantic_revisions(tenant_id=tenant_id)

    print("\nCurrent projection:")
    for item in current:
        print("-", item.object_value, item.status.value, item.valid_from, item.valid_until)

    print("\nValid at", _T0.isoformat(), "(includes superseded revisions):")
    for item in historical:
        print("-", item.object_value, item.status.value, item.valid_from, item.valid_until)

    print("\nRevision history:")
    for revision in revisions:
        print(
            "-",
            revision.memory_key[:12],
            "r",
            revision.revision_number,
            revision.status.value,
            revision.valid_from,
            revision.valid_until,
        )


if __name__ == "__main__":
    asyncio.run(main())
