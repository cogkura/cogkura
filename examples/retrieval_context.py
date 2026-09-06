"""Retrieval-context example for Cogkura 0.16.1."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationContext, ObservationInput, RetrievalContext


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
    query = "Why did we decide not to use Redis?"
    retrieval_context = RetrievalContext(
        conversation_id="arch-42",
        thread_id="queue-selection",
        goal="reduce-operational-complexity",
        activity="architecture-decision",
        domain="payments-api",
        temporal_context=("queue-redesign",),
    )

    baseline = await memory.prepare_context(query, tenant_id=tenant_id)
    contextual = await memory.prepare_context(
        query,
        tenant_id=tenant_id,
        retrieval_context=retrieval_context,
    )
    print("Rendered working memory unchanged:", baseline.render() == contextual.render())

    inspection = await memory.inspect_recall(
        query,
        tenant_id=tenant_id,
        retrieval_context=retrieval_context,
    )
    print(f"Retrieval context supplied: {inspection.retrieval_context is not None}")
    for candidate in inspection.returned:
        if candidate.diagnostics and candidate.diagnostics.context_match:
            match = candidate.diagnostics.context_match
            print(f"Context match score={match.score} coverage={match.cue_coverage}")


if __name__ == "__main__":
    asyncio.run(main())
