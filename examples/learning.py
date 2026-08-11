"""Learning example: outcome feedback, ACT-R reinforcement, and corrections."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput
from cogkura.models import (
    LearningFeedback,
    LearningOutcome,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
    RetrievalCue,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 6, 1, tzinfo=UTC)


async def main() -> None:
    memory = Memory()
    tenant_id = "company_123"
    subject_id = "customer_42"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id=subject_id,
            source_namespace="chat.messages",
            source_record_id="message_1",
            event_type="message",
            content="The team prefers PostgreSQL for production storage.",
            observed_at=_T0,
            metadata={
                "conversation_id": "conv-1",
                "semantic_facts": [
                    {
                        "predicate": "preferred_database",
                        "object_value": "postgresql",
                        "object_entity_id": "postgresql",
                        "cardinality": "one",
                        "polarity": "affirm",
                    }
                ],
            },
        )
    )
    await memory.encode_episodes(tenant_id=tenant_id)
    await memory.consolidate_semantics(tenant_id=tenant_id)

    semantics = await memory.list_semantic_memories(tenant_id=tenant_id)
    semantic = semantics[0]
    goal = RetrievalCue(text="Choose production storage with low operational complexity.")

    result = await memory.learn(
        LearningFeedback(
            tenant_id=tenant_id,
            feedback_id="task_001",
            subject_id=subject_id,
            goal=goal,
            occurred_at=_T1,
            items=(
                MemoryFeedback(
                    identity=MemoryIdentity(
                        memory_kind=MemoryKind.SEMANTIC,
                        memory_key=semantic.memory_key,
                    ),
                    outcome=LearningOutcome.HELPFUL,
                ),
            ),
        )
    )
    print("Learning:", result)

    states = await memory.list_learning_state(
        tenant_id=tenant_id,
        goal=goal,
    )
    print("Learning state rows:", len(states))

    incorrect = await memory.learn(
        LearningFeedback(
            tenant_id=tenant_id,
            feedback_id="task_002",
            subject_id=subject_id,
            occurred_at=_T1,
            items=(
                MemoryFeedback(
                    identity=MemoryIdentity(
                        memory_kind=MemoryKind.SEMANTIC,
                        memory_key=semantic.memory_key,
                    ),
                    outcome=LearningOutcome.INCORRECT,
                ),
            ),
        )
    )
    print("Incorrect feedback:", incorrect)
    print("Corrections still flow through observe -> encode -> consolidate.")


if __name__ == "__main__":
    asyncio.run(main())
