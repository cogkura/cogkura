# Application integration

Cogkura sits between application data and the LLM or agent runtime you already use. It owns memory behaviour. Your application owns data extraction, model invocation, reasoning, tools, and outcome interpretation.

## Boundary

```text
Application data
      ↓
ObservationInput
      ↓
observe / ingest
      ↓
process
      ↓
cognitive memory
      ↓
prepare_context
      ↓
MemoryContext
      ↓
application LLM / agent
      ↓
record_context_use / learn
```

Cogkura prepares memory context. It does not call the LLM.

## Write path

Store normalised observations, then explicitly form recallable memory:

```python
await memory.observe(ObservationInput(...))
await memory.process(tenant_id="shop", subject_id="customer_42")
```

`observe()` and `ingest()` store observations only. `process()` runs episodic encoding and semantic consolidation with one shared `as_of` timestamp.

Processing cadence does not rehearse memory. Calling `process()` repeatedly without new observations leaves cognitive activation references unchanged. Historical observations keep their source chronology: processing them today does not make the represented evidence recent.

Plain-language queries can retrieve relevant current semantic facts via lexical slot matching, evidence-linked relevance from supporting episodes, bounded entity association, bounded soft admission, and authoritative current admission even when base-level activation has decayed. Structured `RetrievalCue` fields remain the precise path when you have predicate or entity metadata.

`valid_at` selects which semantic facts are valid at a snapshot time; it does not by itself mean the query asks for historical state. Use explicit historical wording (`previously`, `before`, `what was`, …) when you intend past-state retrieval.

When recall returns fewer memories than expected, use `inspect_recall()` to see which candidates were below threshold, below the soft floor, insufficiently relevant, filtered by semantic status, filtered, or displaced by the result limit.

### Deactivation semantics

`encode_episodes()` inside `process()` marks episodes inactive when they no longer have backing observations. Episodes are not deleted.

- Tenant-wide `process(tenant_id=...)` deactivates stale active episodes across the tenant.
- Subject-scoped `process(tenant_id=..., subject_id=...)` only deactivates stale episodes for that subject; episodes for other subjects stay active.
- `result.episodes.deactivated` reports how many episodes were marked inactive during the call.

Empty scoped processing (observations cleared or none for that subject) can deactivate previously active episodes for that scope without affecting other subjects.

## Read path

Prepare bounded, assessed context in one read operation:

```python
context = await memory.prepare_context(
    "I'd like another pair, but something lighter.",
    tenant_id="shop",
    subject_id="customer_42",
    goal="Help the customer choose suitable running shoes.",
    prompt_budget_tokens=1500,
)
```

`MemoryContext` exposes structured fields:

- `context.items` — selected working-memory items
- `context.assessment` — metamemory assessment for the same retrieval
- `context.estimated_tokens` — token estimate for selected statements
- `context.render()` — deterministic plain-text rendering of selected memories

Use structured fields for custom prompts, evaluation, and observability. Use `render()` when a simple text block is enough.

### Source provenance (0.18.0)

Source attribution is an explicit, read-only second step, so ordinary context preparation does not add source-store reads:

```python
await memory.observe(
    ObservationInput(
        tenant_id="repository",
        source_namespace="repository",
        source_record_id="adr-0001",
        source_version="1",
        content="Migrations run during deployment.",
        observed_at=now,
        metadata={"source_ref": "docs/adr/0001.md"},
    )
)
await memory.observe(
    ObservationInput(
        tenant_id="repository",
        source_namespace="repository",
        source_record_id="runbook-deploy",
        source_version="7",
        content="Deployments run migrations before traffic.",
        observed_at=now,
        metadata={"source_ref": "docs/runbooks/deploy.md"},
    )
)
await memory.process(tenant_id="repository")

context = await memory.prepare_context("How are migrations deployed?", tenant_id="repository")
provenance = await memory.resolve_provenance(context)

for item in provenance.items:
    print(item.memory_identity, item.distinct_source_count, item.is_complete)
    for source in item.sources:
        print(source.source_ref, source.observation_revision, source.resolution_status)
```

`source_ref` is optional, untrusted application metadata. A missing locator does not remove resolved evidence. `is_latest_known_revision=False` means a newer source revision exists; it does not mean the memory is false or unauthoritative. Provenance does not change rendered context, recall, ranking, consolidation, or learning. See [`design-source-provenance-0.18.0.md`](design-source-provenance-0.18.0.md).

### Source authority (0.18.1)

Recurrence remains the normal way a fact becomes semantic memory. To admit one confirmed fact from a single observation, the application enables the consolidator and marks that fact:

```python
from cogkura import Memory
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator

memory = Memory(
    semantic_consolidator=ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True,
    ),
)

await memory.observe(
    ObservationInput(
        tenant_id="repository",
        subject_id="payments-api",
        source_namespace="repository",
        source_record_id="adr-0001",
        source_version="1",
        content="Database migrations run through the deployment pipeline.",
        observed_at=now,
        metadata={
            "source_ref": "docs/adr/0001.md",
            "semantic_facts": [
                {
                    "predicate": "migration_execution_strategy",
                    "object_value": "deployment_pipeline",
                    "cardinality": "one",
                    "authority": {"basis": "application_asserted"},
                }
            ],
        },
    )
)
```

The application is asserting authority. Cogkura does not infer it from `confirmed`, `source_ref`, or confidence. The declaration has to be stored on the observation revision the episode cites, which means the ingest must change source version, content, deletion, or encoding context. A metadata-only repeat is `UNCHANGED`.

One authoritative episode produces a semantic memory with `support_count=1`. A competing current value in the same cardinality-one slot is deferred, not replaced. Removing the declaration later does not retract the memory. Inspect the route with `semantic_promotion_from_metadata(memory.metadata)` or `resolve_provenance(context).items[].promotion`. `render()` stays free of authority text. See [`design-source-authority-0.18.1.md`](design-source-authority-0.18.1.md).

## Metamemory

`prepare_context()` always includes `MemoryAssessment`. Check flags such as `MISSING_KNOWLEDGE` before invoking your model. Cogkura reports memory state; it does not decide business responses.

## External model invocation

Pseudo-code only:

```python
response = await application_llm(
    system=SYSTEM_PROMPT,
    memory=context.render(),
    message=user_message,
)
```

Provider choice, prompt templates, and tool loops stay in the application.

## Recording use

Selection is not use. After the model consumes context:

```python
await memory.record_context_use(context, request_id=response.request_id)
```

Only memories selected into working memory are recorded. Retrieved candidates that were not selected are not reinforced.

## Learning feedback

Outcome feedback remains separate:

```python
await memory.learn(feedback)
```

`prepare_context()` does not learn or reinforce automatically.

Core recall remains deterministic. Optional future LLM extraction or cue enrichment belongs outside `Memory.recall()`; applications may add adapters at ingestion or query time without changing the memory contract.

## Low-level APIs

Research and advanced integrations can still call `encode_episodes()`, `consolidate_semantics()`, `recall()`, `inspect_recall()`, `select_working_memory()`, `assess_memory()`, and `record_access()` directly. Application integration APIs orchestrate those mechanisms without replacing them.
