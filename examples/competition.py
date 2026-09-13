"""Competition diagnostics, eligibility, and transient interference (0.17.x)."""

import asyncio
from datetime import UTC, datetime, timedelta

from cogkura import CompetitionConfig, Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator


async def _seed_store(
    memory: Memory,
    *,
    tenant_id: str,
    subject_id: str,
    observed_at: datetime,
) -> None:
    observations = (
        ("jenkins", "deployment_system", "jenkins", "payments-api", 120),
        ("gha", "deployment_system", "github_actions", "payments-api", 10),
        ("mysql", "production_database", "mysql", "platform", 200),
        ("pg", "production_database", "postgresql", "platform", 20),
    )
    for source_record_id, predicate, object_value, subject_entity_id, delta_days in observations:
        await memory.observe(
            ObservationInput(
                tenant_id=tenant_id,
                subject_id=subject_id,
                source_namespace="facts",
                source_record_id=source_record_id,
                source_type="fact",
                content=f"{subject_entity_id} {predicate} {object_value}",
                observed_at=observed_at - timedelta(days=delta_days),
                metadata={
                    "entity_ids": [subject_entity_id],
                    "semantic_facts": [
                        {
                            "predicate": predicate,
                            "object_value": object_value,
                            "subject_entity_id": subject_entity_id,
                            "cardinality": "many",
                            "polarity": "affirm",
                            "qualifiers": {},
                        }
                    ],
                },
            )
        )
    await memory.process(tenant_id=tenant_id, as_of=observed_at)


def _print_eligibility(label: str, inspection: object) -> None:
    print(f"\n=== {label} ===")
    run = inspection.competition
    if run is not None:
        print(
            f"pairs accepted={run.accepted_competition_pairs} "
            f"eligible={run.behaviorally_eligible_pairs} "
            f"rejected={run.behaviorally_rejected_pairs}"
        )
        if run.rejected_by_reason:
            print(f"rejected_by_reason: {dict(run.rejected_by_reason)}")
    for candidate in inspection.returned:
        if candidate.competition is None:
            continue
        eligible = candidate.competition.behaviorally_eligible_competitor_count
        rejected = candidate.competition.behaviorally_rejected_competitor_count
        penalty = (
            candidate.competition.interference.total_penalty
            if candidate.competition.interference is not None
            else 0.0
        )
        print(
            f"  {candidate.memory.statement[:52]}… "
            f"diag={candidate.competition.competitor_count} "
            f"eligible={eligible} rejected={rejected} I={penalty:.3f}"
        )
        for evidence in candidate.competition.competitors:
            if evidence.behavioral_eligibility is None:
                continue
            print(
                f"    vs {evidence.competitor_identity.memory_key}: "
                f"{evidence.behavioral_eligibility.reason.value} "
                f"(eligible={evidence.behavioral_eligibility.eligible})"
            )


async def main() -> None:
    tenant_id = "acme"
    subject_id = "operator-1"
    observed_at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)

    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
        competition_config=CompetitionConfig(enabled=True, apply_interference=True),
    )
    await _seed_store(memory, tenant_id=tenant_id, subject_id=subject_id, observed_at=observed_at)

    deployment = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=tenant_id,
        limit=10,
        as_of=observed_at,
    )
    _print_eligibility("Deployment query (eligible rivals may interfere)", deployment)

    database = await memory.inspect_recall(
        "platform production_database",
        tenant_id=tenant_id,
        limit=10,
        as_of=observed_at,
    )
    _print_eligibility("Database query (deployment bleed blocked behaviourally)", database)


if __name__ == "__main__":
    asyncio.run(main())
