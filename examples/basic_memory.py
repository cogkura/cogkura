"""Basic usage example for Cognema."""

import asyncio
from datetime import UTC, datetime

from cognema import Memory, ObservationInput


async def main() -> None:
    memory = Memory()
    tenant_id = "local"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="george",
            source_namespace="direct",
            source_record_id="1",
            content="George discussed cognitive memory algorithms",
            observed_at=datetime.now(UTC),
            metadata={
                "conversation_id": "research",
                "source": "conversation",
                "tags": ["research", "memory"],
            },
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="george",
            source_namespace="direct",
            source_record_id="2",
            content="The team agreed to prototype deterministic recall first.",
            observed_at=datetime.now(UTC),
            metadata={"conversation_id": "research"},
        )
    )

    results = await memory.recall(
        "What did George discuss about memory?",
        tenant_id=tenant_id,
    )
    for result in results:
        print(f"{result.score:.2f} :: {result.observation.content}")

    encoding = await memory.encode_episodes(tenant_id=tenant_id)
    episodes = await memory.list_episodes(tenant_id=tenant_id)
    print(f"episodes created={encoding.created} listed={len(episodes)}")
    if episodes:
        print(episodes[0].statement)

    consolidation = await memory.consolidate_semantics(tenant_id=tenant_id)
    semantic_memories = await memory.list_semantic_memories(tenant_id=tenant_id)
    print(f"semantic promoted={consolidation.promoted} listed={len(semantic_memories)}")

    memory.sleep()


if __name__ == "__main__":
    asyncio.run(main())
