"""Declarative activation example."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput


async def main() -> None:
    memory = Memory()
    tenant_id = "local"

    for index, conversation_id in enumerate(("conv-1", "conv-2"), start=1):
        await memory.observe(
            ObservationInput(
                tenant_id=tenant_id,
                subject_id="george",
                source_namespace="direct",
                source_record_id=str(index),
                content="PostgreSQL incident resolved with deterministic recall.",
                observed_at=datetime.now(UTC),
                metadata={"conversation_id": conversation_id},
            )
        )

    await memory.encode_episodes(tenant_id=tenant_id)
    first = await memory.recall("PostgreSQL incident", tenant_id=tenant_id, limit=3)
    print(f"first recall: {len(first)} results")
    if first:
        print(f"  activation={first[0].activation:.3f} score={first[0].score:.3f}")

    await memory.record_access(first[:1], tenant_id=tenant_id, request_id="demo-run-1")
    second = await memory.recall("PostgreSQL incident", tenant_id=tenant_id, limit=3)
    if first and second:
        print(f"after reinforcement activation={second[0].activation:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
