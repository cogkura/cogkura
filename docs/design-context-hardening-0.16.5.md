# Cogkura Core 0.16.5 — Context Hardening and Architecture Freeze

**Status:** Shipped design note.

## Summary

`0.16.5` hardens contextual metamemory and semantic support diagnostics and freezes the first-generation 0.16 contextual-memory architecture. It does **not** add another contextual retrieval mechanism.

Principle: **fix semantics, strengthen contracts, freeze behaviour.**

## Goals

1. Introduce `CONTEXT_CONFLICT` when comparable evidence has zero positive matches.
2. Correct zero-match classification (must not yield `CONTEXT_SUFFICIENT`).
3. Expose semantic mixed-support partial conflict via `PARTIAL_CONTEXT_CONFLICT`.
4. Distinguish not-evaluated vs unavailable support context counts/reasons.
5. Reserve `MemoryContextSignature.concept_ids` on the default encoder.
6. Reconcile weight documentation with implemented defaults (`0.50` / `0.25`).
7. Add consolidated architecture-freeze tests.

## Non-goals

No new context dimensions, fuzzy matching, embeddings, negative penalties, learned weighting, context-based candidate generation, new semantic formulae, context-sensitive forgetting, or working-memory rule changes.

## Metamemory decision tree (frozen)

```text
no retrieval context → CONTEXT_NOT_PROVIDED
comparable_candidate_count == 0 → CONTEXT_UNAVAILABLE
matching_candidate_count == 0 → CONTEXT_CONFLICT (+ NO_CONTEXTUAL_MATCH)
ambiguous positive matches → CONTEXT_UNDERSPECIFIED
else → CONTEXT_SUFFICIENT
```

`CONTEXT_CONFLICT` is observational only; mismatch remains zero bonus (no penalty).

## Semantic support diagnostics

| Condition | comparable | unavailable | reason |
|-----------|------------|-------------|--------|
| No retrieval context | 0 | 0 | `NO_RETRIEVAL_CONTEXT` |
| Cue present, no comparable supports | 0 | N | `NO_COMPARABLE_CONTEXT` |
| Cache/evidence missing internally | 0 | 0 | `NOT_EVALUATED` |

Mixed matching + conflicting supports emit `PARTIAL_CONTEXT_CONFLICT` at retrieval level without changing `Rₛ = ΣRⱼ/K`.

## concept_ids

`concept_ids` remains on `MemoryContextSignature` but is **not** populated by the default episodic encoder. Reserved for future structured concept context; no text inference.

## Architecture freeze

After `0.16.5`, changes to encoding context, retrieval context matching, reinstatement, semantic support propagation, contextual activation integration, observability, and metamemory state semantics require explicit architectural design.

See [`architecture.md`](architecture.md), [`findings/0.16.5-context-hardening.md`](findings/0.16.5-context-hardening.md), and [`tests/test_contextual_memory_architecture_freeze.py`](../tests/test_contextual_memory_architecture_freeze.py).
