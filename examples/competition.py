"""Competition diagnostics example for Cogkura 0.17.0."""

import asyncio
from datetime import UTC, datetime, timedelta

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator


async def main() -> None:
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )
    tenant_id = "acme"
    subject_id = "operator-1"
    observed_at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)

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

    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=tenant_id,
        limit=10,
        as_of=observed_at,
    )
    print(f"competition counters: {inspection.competition}")
    for candidate in inspection.returned:
        if candidate.competition is None or candidate.competition.competitor_count == 0:
            continue
        print(f"\nCandidate: {candidate.memory.statement}")
        for evidence in candidate.competition.competitors:
            print(
                f"  competitor={evidence.competitor_identity.memory_key[:12]}… "
                f"strength={evidence.strength:.2f} direction={evidence.direction.value} "
                f"same_slot={evidence.same_semantic_slot}"
            )


if __name__ == "__main__":
    asyncio.run(main())
