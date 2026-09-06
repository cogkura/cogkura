# Retrieval context and context matching — 0.16.1

**Target release:** `0.16.1`  
**Status:** Shipped

## Summary

Cogkura `0.16.1` introduces **retrieval context** and deterministic **context matching** between request-time cues and episodic encoding context. Matching is calculated, inspectable, serializable, and benchmarkable, but does **not** influence activation, admission, ranking, or working-memory selection. Behavioural integration is planned for `0.16.2`.

## Vocabulary

```text
ObservationContext → MemoryContextSignature → RetrievalContext → ContextMatch
```

- **ObservationContext** — supplied at ingest (`0.16.0`).
- **MemoryContextSignature** — stored on episodic memories (`0.16.0`).
- **RetrievalContext** — structured cues on recall APIs (`0.16.1`).
- **ContextMatch** — ephemeral comparison result (`0.16.1`).

## Public API

- `RetrievalContext` in `cogkura.observations.encoding_context`.
- Optional `RetrievalCue.retrieval_context` and `retrieval_context=` on `recall`, `inspect_recall`, `select_working_memory`, `prepare_context`, and `assess_memory`.
- `DeterministicContextMatcher`, `ContextMatch`, `ContextDimensionMatch`, `ContextMatchState`.
- `RetrievalDiagnostics.context_match` attached **after** declarative ranking.

## Matching contract

Primary scored dimensions (equal weight): `conversation`, `thread`, `session`, `goal`, `activity`, `domain`, `location`, `temporal_context`.

- Exact normalised string comparison only.
- Missing cue dimensions do not participate.
- Missing encoding values are `UNAVAILABLE`, not `MISMATCH`.
- Extra encoded values do not penalise a matching cue.
- Attributes are diagnostic-only and excluded from the primary score.
- Semantic candidates are not matched (`context_match=None`).

## Non-goals (0.16.1)

- No activation term (`λCᵢ`) or ranking changes.
- No storage migration or persisted matches.
- No query-text parsing for context.
- No fuzzy, embedding, or LLM matching.

## Next

- **0.16.2** — context reinstatement influences accessibility during recall.
