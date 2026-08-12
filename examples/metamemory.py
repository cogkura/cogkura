"""Metamemory example: assess retrieved memory without mutating cognitive state."""

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
    tenant_id = "local"
    subject_id = "team"

    observations = [
        "PostgreSQL is already operated by the team.",
        "PostgreSQL supports the required transactions.",
        "Redis introduces an additional operational dependency.",
        "PostgreSQL was selected for production.",
    ]
    for index, content in enumerate(observations, start=1):
        await memory.observe(
            ObservationInput(
                tenant_id=tenant_id,
                subject_id=subject_id,
                source_namespace="chat.messages",
                source_record_id=f"message_{index}",
                event_type="message",
                content=content,
                observed_at=_T0,
                metadata={"conversation_id": "architecture_123"},
            )
        )

    await memory.encode_episodes(tenant_id=tenant_id)
    await memory.consolidate_semantics(tenant_id=tenant_id)

    semantics = await memory.list_semantic_memories(tenant_id=tenant_id)
    goal = RetrievalCue(text="Recall the production database decision.")

    for semantic in semantics:
        await memory.learn(
            LearningFeedback(
                tenant_id=tenant_id,
                feedback_id=f"helpful-{semantic.memory_key}",
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

    assessment = await memory.assess_memory(
        "What database did we select for production?",
        tenant_id=tenant_id,
        subject_id=subject_id,
        goal=goal,
        as_of=_T1,
    )

    print(f"Cue coverage:            {assessment.signals.cue_coverage:.2f}")
    print(f"Top retrieval strength:  {assessment.signals.top_retrieval_strength:.2f}")
    print(f"Mean retrieval strength: {assessment.signals.mean_retrieval_strength:.2f}")
    if assessment.signals.evidence_confidence is not None:
        print(f"Evidence confidence:     {assessment.signals.evidence_confidence:.2f}")
    print(f"Semantic conflict:       {assessment.signals.semantic_conflict:.2f}")
    print(f"Provenance diversity:    {assessment.signals.provenance_diversity:.2f}")
    if assessment.signals.forgetting_pressure is not None:
        print(f"Forgetting pressure:     {assessment.signals.forgetting_pressure:.2f}")
    if assessment.signals.learned_utility is not None:
        print(f"Learned utility:         {assessment.signals.learned_utility:.2f}")
    print("Flags:", ", ".join(flag.value for flag in assessment.flags) or "none")


if __name__ == "__main__":
    asyncio.run(main())
