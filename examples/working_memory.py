"""Working-memory selection example."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput


async def main() -> None:
    memory = Memory()
    tenant_id = "local"
    goal = "Choose storage appropriate for production while minimising operational complexity."

    observations = [
        "PostgreSQL is already operated by the team.",
        "Redis would introduce an additional operational dependency.",
        "PostgreSQL supports the required transactional workload.",
        "The company Christmas party was held in Manchester.",
    ]
    for index, content in enumerate(observations, start=1):
        await memory.observe(
            ObservationInput(
                tenant_id=tenant_id,
                subject_id="george",
                source_namespace="direct",
                source_record_id=str(index),
                content=content,
                observed_at=datetime.now(UTC),
            )
        )

    await memory.encode_episodes(tenant_id=tenant_id)
    await memory.consolidate_semantics(tenant_id=tenant_id)

    workspace = await memory.select_working_memory(
        "What should we use for production storage?",
        tenant_id=tenant_id,
        subject_id="george",
        goal=goal,
    )

    print("Working memory:")
    for item in workspace.items:
        print(
            item.rank,
            item.memory.statement,
            item.components.activation,
            item.components.goal_relevance,
            item.components.inhibition,
            item.components.final_score,
            item.estimated_tokens,
        )

    workspace = await memory.select_working_memory(
        "What are the trade-offs?",
        tenant_id=tenant_id,
        subject_id="george",
        goal=goal,
        previous=workspace,
    )

    print("Working memory after follow-up:")
    for item in workspace.items:
        print(
            item.rank,
            item.memory.statement,
            item.components.carryover,
            item.components.final_score,
        )


if __name__ == "__main__":
    asyncio.run(main())
