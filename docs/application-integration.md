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
