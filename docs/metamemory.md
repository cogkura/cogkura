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

`MISSING_KNOWLEDGE` (`0.13`) fires when recall returns weak or low-coverage results and no retrieved ACTIVE semantic matches the cue slot (predicate, entities, or current-state tokens). A full pool of unrelated weak hits still abstains. An empty pool emits `NO_RETRIEVED_MEMORY`. Interference flags may accompany that flag when competition, transient interference, or persistent inhibition explains the empty retrieval. Those flags describe retrieval state. They do not mean a memory is false, an answer is wrong, or a memory should be deleted.

`COMPETING_MEMORIES` (`0.17.5`) means credible query-scope competition exists: at least one scope-eligible competition relationship. Raw diagnostic competition that fails the behavioural scope gate does not set it. It does not mean the memories contradict each other, and it does not set `CONFLICTING_SEMANTIC_MEMORY` or `MISSING_KNOWLEDGE`.

`HIGH_INTERFERENCE` (`0.17.5`) means current proactive or retroactive transient interference is high enough to meet `MetamemoryConfig.high_interference_pressure_threshold` (default `0.50`), or it pushed a candidate below the retrieval threshold. Competition alone does not set it.

`RETRIEVAL_INHIBITION_ACTIVE` (`0.17.5`) means persistent retrieval-induced inhibition is actively reducing accessibility in the current retrieval. Stored traces that are not applied, and matched traces that have recovered below the effective cutoff, do not set it. There is no `HIGH_INHIBITION` severity flag in this release.

## Interference observability (`0.17.5`)

`inspect_recall().interference` and `assess_memory().interference` carry the same `RetrievalInterferenceObservability` summary. It is aggregated from the pre-threshold discrimination set of one retrieval evaluation, so a candidate dropped below threshold remains visible in the summary even when it is absent from final recall. `None` on a pressure means that mechanism was not evaluated. `0.0` means it was evaluated and had no effect.

`DeterministicMemoryMonitor` only classifies that precomputed summary. It does not read inhibition, activation, or semantic stores. `recall()` results are unchanged.

## Read-only guarantees

`assess_memory()` does not call `record_access()`, `apply_forgetting()`, `learn()`, or any store writes. Monitoring does not rehearse or mutate memory.

## Limitations

- Assesses only threshold-qualified recalled memories, not all latent storage.
- Provenance diversity counts observation traces, not independent external sources.
- No persistence or PostgreSQL migration for assessments.

## Contextual metamemory (`0.16.4+`, hardened `0.16.5`)

`MemoryAssessment.context` carries additive `RetrievalContextDiagnostics` derived from the same recall pool used for flags and signals. Retrieval-level states:

| State | Meaning |
|-------|---------|
| `CONTEXT_NOT_PROVIDED` | No structured retrieval context |
| `CONTEXT_UNAVAILABLE` | Cue exists but no comparable encoding context on plausible candidates |
| `CONTEXT_CONFLICT` | Comparable evidence exists but zero positive contextual matches (`NO_CONTEXTUAL_MATCH`) |
| `CONTEXT_UNDERSPECIFIED` | Positive matches remain ambiguous (margin/tie) |
| `CONTEXT_SUFFICIENT` | Context meaningfully discriminates |

**Canonical surface:** `inspect_recall().context` over the full inspect discrimination set. `assess_memory().context` uses the narrower threshold-qualified pool and may differ when borderline candidates are excluded from recall.

Semantic mixed-support partial conflict surfaces as `PARTIAL_CONTEXT_CONFLICT` when both matching and conflicting supports exist on the same semantic candidate. Episodic partial conflict remains visible on `ContextMatch` dimensions.

Contextual assessment is knowledge about retrieval evidence, not memory content. It must not be confused with slot/entity `_metamemory_match_context` or fed into `MemoryContext.render()`.

See [`examples/metamemory.py`](../../examples/metamemory.py).
