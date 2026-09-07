# Context observability and metamemory (0.16.4)

## Status

Shipped in Cogkura `0.16.4`.

## Summary

`0.16.0`–`0.16.3` introduced encoding context, retrieval-context matching, episodic reinstatement, and semantic support-context propagation. **`0.16.4` explains that evidence** without changing activation, admission, ranking, working-memory selection, or `record_context_use`.

Contextual metamemory is **observational**: it classifies retrieval-level contextual evidence on `inspect_recall` and fills additive `MemoryAssessment.context`. It does not persist traces, add penalties, rewrite queries, or feed `MemoryContext.render()`.

## Public surfaces

| Surface | Role |
|---------|------|
| `RetrievalDiagnostics.crossed_activation_threshold_due_to_context` | Per-candidate threshold attribution |
| `RecallInspectionCandidate.rank_before_context` / `rank_after_context` / `context_rank_delta` | Inspect-only pre/post ranks over the discrimination set |
| `RecallInspectionResult.context` | Canonical retrieval-level `RetrievalContextDiagnostics` |
| `MemoryAssessment.context` | Same classifier over the narrower recall pool from `assess_memory` |
| `MetamemoryConfig.context_underspecified_margin` | Engineering calibration for underspecification ties (default `0.0`) |

## Classification

`DeterministicRetrievalContextPolicy` states:

1. Empty/missing `RetrievalContext` → `context_not_provided`
2. No comparable encoding context on discrimination-set candidates → `context_unavailable`
3. ≥2 comparable candidates with top−second context strength ≤ margin → `context_underspecified`
4. Else → `context_sufficient`

Candidate-level mixed match/mismatch stays on existing `ContextMatch` dimensions. Retrieval-level `CONTEXT_CONFLICT` is intentionally omitted.

## Invariants

- `rank()` scoring unchanged; full attribution runs in `inspect()` post-pass only.
- Discrimination set includes `RETURNED`, `BELOW_THRESHOLD`, `LIMITED`; excludes relevance-filtered and other filtered dispositions.
- Pre-context rank uses `activation_before_context`, not post-context `rank_activation`.
- No synthetic confidence float; reasons are structured `ContextObservabilityReason` codes.

See [`findings/0.16.4-context-observability.md`](findings/0.16.4-context-observability.md) for calibration notes.
