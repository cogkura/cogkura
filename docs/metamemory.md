# Metamemory and memory monitoring

Target release: `0.10`
Status: Shipped

## Summary

Cogkura `0.10.0` adds deterministic **metamemory**: read-only assessment of retrieved memory state.

`Memory.assess_memory()` answers:

> Given a retrieval request, what does Cogkura know about the state and quality of the memories it can currently retrieve?

It does **not** answer whether an eventual LLM answer is correct.

## Monitoring vs control

Metamemory reports signals and diagnostic flags. The host application decides whether to reason, search, abstain, or ask the user.

There is no `should_answer`, `should_search`, or overall confidence score.

## Public API

```python
assessment = await memory.assess_memory(
    "What database did we select for production?",
    tenant_id="company_123",
    goal="Recall the production database decision.",
)
```

Returns `MemoryAssessment` with `signals`, `flags`, bounded `items`, and aggregate counts.

Assessment operates over `recall(limit=candidate_pool_size)` — memories that cross the declarative retrieval threshold.

## Signals

Independent dimensions (never collapsed into one score):

- **Cue coverage** — collective match to the retrieval cue (union token coverage for free text).
- **Retrieval strength** — `RecallResult.score` (top and mean); accessibility, not truth.
- **Evidence confidence** — retrieval-weighted `memory.confidence`.
- **Semantic conflict** — contested or contradictory semantic evidence.
- **Provenance diversity** — distinct `observation_id` traces across retrieved memories.
- **Freshness** — optional, only when `freshness_half_life_seconds` is set.
- **Forgetting pressure** — live estimate from `retention_score_from_base_level`.
- **Learned utility** — context-specific utility from `0.9` learning (neutral `0.5` when enabled with no feedback).

## Warning flags

Diagnostic flags such as `LOW_RETRIEVAL_STRENGTH`, `CONFLICTING_SEMANTIC_MEMORY`, `MISSING_KNOWLEDGE`, and `NO_RETRIEVED_MEMORY` are emitted in a fixed order when thresholds are crossed.

`MISSING_KNOWLEDGE` (`0.13`) fires when recall returns weak or low-coverage results and no retrieved ACTIVE semantic matches the cue slot (predicate, entities, or current-state tokens). A full pool of unrelated weak hits still abstains; empty pools emit `NO_RETRIEVED_MEMORY` only.

## Read-only guarantees

`assess_memory()` does not call `record_access()`, `apply_forgetting()`, `learn()`, or any store writes. Monitoring does not rehearse or mutate memory.

## Limitations

- Assesses only threshold-qualified recalled memories, not all latent storage.
- Provenance diversity counts observation traces, not independent external sources.
- No persistence or PostgreSQL migration for assessments.

See [`examples/metamemory.py`](../../examples/metamemory.py).
