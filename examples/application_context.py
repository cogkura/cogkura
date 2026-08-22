"""Application integration example without an external LLM."""

import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput


async def main() -> None:
    memory = Memory()
    tenant_id = "shop"
    subject_id = "customer_42"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id=subject_id,
            source_namespace="orders",
            source_record_id="order_123",
            event_type="purchase",
            content="Customer purchased running shoes in UK size 11.",
            observed_at=datetime.now(UTC),
            metadata={
                "semantic_facts": [
                    {
                        "predicate": "shoe_size",
                        "object_value": "UK 11",
                        "cardinality": "one",
                        "polarity": "affirm",
                    },
                    {
                        "predicate": "shoe_weight_preference",
                        "object_value": "lightweight",
                        "cardinality": "one",
                        "polarity": "affirm",
                    },
                ],
            },
        )
    )

    await memory.process(tenant_id=tenant_id, subject_id=subject_id)

    context = await memory.prepare_context(
        "I'd like another pair, but something lighter.",
        tenant_id=tenant_id,
        subject_id=subject_id,
        goal="Help the customer choose suitable running shoes.",
        prompt_budget_tokens=1500,
    )

    print(context.render())
    print(f"Estimated tokens: {context.estimated_tokens}")
    print(f"Assessment flags: {list(context.assessment.flags)}")

    await memory.record_context_use(context)


if __name__ == "__main__":
    asyncio.run(main())
