"""Explicit source assessment; the application owns any abstention decision."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput, SemanticDiagnosticsConfig, SemanticSlotQuery


async def main() -> None:
    clock = datetime(2026, 11, 14, tzinfo=UTC)
    memory = Memory(semantic_diagnostics_config=SemanticDiagnosticsConfig(enabled=True))
    for n in range(2):
        await memory.observe(
            ObservationInput(
                tenant_id="example",
                subject_id="service",
                source_namespace="documents",
                source_record_id=f"policy-{n}",
                content="Deployment policy is pipeline.",
                observed_at=clock,
                metadata={
                    "conversation_id": f"policy-{n}",
                    "source_ref": f"docs/policy-{n}.md",
                    "semantic_facts": [
                        {
                            "predicate": "deployment_policy",
                            "object_value": "pipeline",
                            "cardinality": "one",
                        }
                    ],
                },
            )
        )
    result = await memory.process(tenant_id="example", as_of=clock)
    context = await memory.prepare_context("deployment policy", tenant_id="example", as_of=clock)
    sources = await memory.assess_source_memory(context)
    slot = await memory.inspect_semantic_slot(
        SemanticSlotQuery("service", "deployment_policy"), tenant_id="example", as_of=clock
    )
    print(context.render())
    print("Revision coverage:", sources.quality.revision_coverage_complete)
    print("Semantic relation coverage:", sources.quality.relation_coverage_complete)
    print("Slot state:", slot.state)
    print("Source independence:", slot.conflict.independence)
    print("Decision history:", slot.decision_history)
    print("Pass decisions:", result.semantics.decisions)


if __name__ == "__main__":
    asyncio.run(main())
