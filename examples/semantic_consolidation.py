"""Semantic consolidation end-to-end example."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput

_SEMANTIC_FACT = {
    "predicate": "preferred_database",
    "object_value": "postgresql",
    "object_entity_id": "postgresql",
    "cardinality": "one",
    "polarity": "affirm",
    "qualifiers": {"environment": "production"},
}


async def main() -> None:
    memory = Memory()
    tenant_id = "local"
    subject_id = "customer_42"

    for index, conversation_id in enumerate(("conv-1", "conv-2"), start=1):
        await memory.observe(
            ObservationInput(
                tenant_id=tenant_id,
                subject_id=subject_id,
                source_namespace="direct",
                source_record_id=f"message_{index}",
                content="Database preference discussion.",
                observed_at=datetime.now(UTC),
                metadata={
                    "conversation_id": conversation_id,
                    "entity_ids": [subject_id],
                    "semantic_facts": [_SEMANTIC_FACT],
                },
            )
        )

    encoding = await memory.encode_episodes(tenant_id=tenant_id, subject_id=subject_id)
    consolidation = await memory.consolidate_semantics(
        tenant_id=tenant_id,
        subject_id=subject_id,
    )
    semantic_memories = await memory.list_semantic_memories(
        tenant_id=tenant_id,
        subject_id=subject_id,
    )

    print(f"episodes created={encoding.created}")
    print(
        f"semantic promoted={consolidation.promoted} "
        f"created={consolidation.created} failures={consolidation.extracted_failures}"
    )
    for item in semantic_memories:
        print(f"{item.predicate}={item.object_value} support={item.support_count}")


if __name__ == "__main__":
    asyncio.run(main())
