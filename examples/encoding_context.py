"""Encoding-context example for Cogkura 0.16.0."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationContext, ObservationInput


async def main() -> None:
    memory = Memory()
    tenant_id = "acme"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="developer-1",
            source_namespace="github",
            source_record_id="redis-decision",
            source_type="discussion",
            content=(
                "We decided not to use Redis because another stateful dependency "
                "would increase operational complexity."
            ),
            observed_at=datetime.now(UTC),
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("redis", "payments-api"),
            },
            context=ObservationContext(
                conversation_id="arch-42",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="payments-api",
                temporal_context=("queue-redesign",),
            ),
        )
    )

    await memory.encode_episodes(tenant_id=tenant_id)
    episodes = await memory.list_episodes(tenant_id=tenant_id)
    if not episodes:
        print("No episodes encoded.")
        return

    signature = episodes[0].encoding_context
    print("Encoding context:")
    print(f"  conversation: {signature.conversation_ids}")
    print(f"  goal: {signature.goals}")
    print(f"  activity: {signature.activities}")
    print(f"  domain: {signature.domains}")
    print(f"  entities: {signature.entity_ids}")

    results = await memory.recall(
        "Why did we decide not to use Redis?",
        tenant_id=tenant_id,
    )
    for result in results:
        print(f"{result.score:.2f} :: {result.memory.statement}")


if __name__ == "__main__":
    asyncio.run(main())
