"""Retrieval-context example for Cogkura 0.16.4."""

import asyncio
from datetime import UTC, datetime, timedelta

from cogkura import Memory, ObservationContext, ObservationInput, RetrievalContext


async def main() -> None:
    memory = Memory()
    tenant_id = "acme"
    subject_id = "developer-1"
    observed_at = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)

    observations = [
        ObservationInput(
            tenant_id=tenant_id,
            subject_id=subject_id,
            source_namespace="github",
            source_record_id="redis-decision",
            source_type="discussion",
            content=(
                "We decided not to use Redis because another stateful dependency "
                "would increase operational complexity."
            ),
            observed_at=observed_at,
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
        ),
        ObservationInput(
            tenant_id=tenant_id,
            subject_id=subject_id,
            source_namespace="github",
            source_record_id="auth-redis",
            source_type="discussion",
            content="Auth service uses Redis for session caching.",
            observed_at=observed_at + timedelta(hours=4),
            metadata={
                "conversation_id": "arch-43",
                "entity_ids": ("redis", "auth-api"),
            },
            context=ObservationContext(
                conversation_id="arch-43",
                thread_id="auth-cache",
                goal="improve-session-latency",
                activity="architecture-decision",
                domain="auth-api",
            ),
        ),
    ]
    for observation in observations:
        await memory.observe(observation)
    await memory.process(tenant_id=tenant_id, subject_id=subject_id)

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
    print("No-context render unchanged:", baseline.render() == contextual.render())

    baseline_recall = await memory.recall(query, tenant_id=tenant_id)
    contextual_recall = await memory.recall(
        query,
        tenant_id=tenant_id,
        retrieval_context=retrieval_context,
    )
    baseline_keys = [result.memory.memory_key for result in baseline_recall]
    contextual_keys = [result.memory.memory_key for result in contextual_recall]
    print("Recall order may change with matching context:", baseline_keys != contextual_keys)
    print("Baseline order:", baseline_keys)
    print("Contextual order:", contextual_keys)

    inspection = await memory.inspect_recall(
        query,
        tenant_id=tenant_id,
        retrieval_context=retrieval_context,
    )
    print(f"Retrieval context supplied: {inspection.retrieval_context is not None}")
    if inspection.context is not None:
        print(
            f"Inspect context state={inspection.context.state.value} "
            f"margin={inspection.context.context_margin} "
            f"reasons={[reason.value for reason in inspection.context.reasons]}"
        )
    for candidate in inspection.returned:
        if candidate.diagnostics and candidate.diagnostics.context_reinstatement:
            rest = candidate.diagnostics.context_reinstatement
            print(
                f"Episodic reinstatement applied={rest.applied} "
                f"contribution={rest.activation_contribution:.3f} "
                f"reason={rest.reason.value}"
            )
        if candidate.diagnostics and candidate.diagnostics.support_context:
            support = candidate.diagnostics.support_context
            print(
                f"Semantic support-context applied={support.applied} "
                f"strength={support.strength:.3f} "
                f"contribution={support.activation_contribution:.3f} "
                f"supports={support.support_count}"
            )


if __name__ == "__main__":
    asyncio.run(main())
