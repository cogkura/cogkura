"""Competition diagnostics and transient interference example for Cogkura 0.17.x."""

import asyncio
from datetime import UTC, datetime, timedelta

from cogkura import CompetitionConfig, Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator


async def _seed_deployment_memories(
    memory: Memory,
    *,
    tenant_id: str,
    subject_id: str,
    observed_at: datetime,
) -> None:
    for source_record_id, object_value, delta_days in (
        ("jenkins-era", "jenkins", 120),
        ("gha-era", "github_actions", 10),
    ):
        await memory.observe(
            ObservationInput(
                tenant_id=tenant_id,
                subject_id=subject_id,
                source_namespace="facts",
                source_record_id=source_record_id,
                source_type="fact",
                content=f"payments-api deployment_system {object_value}",
                observed_at=observed_at - timedelta(days=delta_days),
                metadata={
                    "entity_ids": ["payments-api"],
                    "semantic_facts": [
                        {
                            "predicate": "deployment_system",
                            "object_value": object_value,
                            "subject_entity_id": "payments-api",
                            "cardinality": "many",
                            "polarity": "affirm",
                            "qualifiers": {},
                        }
                    ],
                },
            )
        )
    await memory.process(tenant_id=tenant_id, as_of=observed_at)


async def main() -> None:
    tenant_id = "acme"
    subject_id = "operator-1"
    observed_at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    query = "payments-api deployment_system"

    diagnostic_memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
    )
    await _seed_deployment_memories(
        diagnostic_memory,
        tenant_id=tenant_id,
        subject_id=subject_id,
        observed_at=observed_at,
    )
    diagnostic_only = await diagnostic_memory.inspect_recall(
        query,
        tenant_id=tenant_id,
        limit=10,
        as_of=observed_at,
    )
    print("=== Diagnostics only (apply_interference=False) ===")
    print(f"competition counters: {diagnostic_only.competition}")
    for candidate in diagnostic_only.returned:
        if candidate.competition is None:
            continue
        print(
            f"  {candidate.memory.statement[:48]}… "
            f"activation={candidate.activation:.3f} "
            f"competitors={candidate.competition.competitor_count}"
        )

    interference_memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
    )
    await _seed_deployment_memories(
        interference_memory,
        tenant_id=tenant_id,
        subject_id=subject_id,
        observed_at=observed_at,
    )
    recalled = await interference_memory.recall(
        query,
        tenant_id=tenant_id,
        limit=10,
        as_of=observed_at,
    )
    print("\n=== Transient interference enabled (apply_interference=True) ===")
    for result in recalled:
        diagnostics = result.diagnostics
        penalty = result.components.interference
        before = diagnostics.activation_before_interference if diagnostics else None
        print(
            f"  {result.memory.statement[:48]}… "
            f"A_pre={before:.3f} A_final={result.activation:.3f} I={penalty:.3f}"
            if before is not None
            else f"  {result.memory.statement[:48]}… activation={result.activation:.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
