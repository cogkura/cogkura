# Interference observability and metamemory (0.17.5)

Shipped in Cogkura `0.17.5`.

> Retrieval behaviour is unchanged. `inspect_recall` and `assess_memory` can now say whether credible competition, current transient interference, or persistent retrieval-induced inhibition is affecting this retrieval.

## Summary

`0.17.0`–`0.17.4` made competition, behavioural eligibility, transient interference, and persistent inhibition individually inspectable on candidates. `0.17.5` adds a query-level summary of their effect.

`RetrievalInterferenceObservability` is attached to `RecallInspectionResult.interference` and `MemoryAssessment.interference`. Both come from the same summarizer over the candidates that reached interference and inhibition evaluation, before threshold filtering, collapse, and result limiting. A memory pushed below threshold remains visible in the summary after it disappears from `recall()`.

`recall()` still returns `list[RecallResult]`. Working-memory selection still uses that thresholded list. No store, migration, or second retrieval was added. When `apply_to_recall` is false, observability does not read `InhibitionStore`.

## State

| State | Meaning |
|-------|---------|
| `not_evaluated` | Competition, transient interference, and persistent application were all off |
| `clear` | At least one mechanism ran, with no scope-eligible competition and no active penalty |
| `competing` | Scope-eligible competition exists, and neither penalty is active |
| `transient_interference` | Transient interference contributes a negative penalty, and persistent inhibition does not |
| `persistent_inhibition` | Persistent inhibition contributes a negative penalty, and transient interference does not |
| `combined` | Both penalties are active |

Unevaluated pressures are `None`. Evaluated pressures with no effect are `0.0`.

The strongest pair is the highest scope-eligible relationship, tie-broken by candidate identity then competitor identity. Its activation margin is the absolute difference of post-inhibition, pre-transient rank activation.

## Flags

These describe retrieval state. They do not judge truth, and they do not imply semantic contradiction or missing knowledge.

| Flag | Set when |
|------|----------|
| `COMPETING_MEMORIES` | `scope_eligible_relationship_count > 0` |
| `HIGH_INTERFERENCE` | Transient interference was evaluated and pressure meets `high_interference_pressure_threshold` (default `0.50`), or transient interference suppressed a candidate below threshold |
| `RETRIEVAL_INHIBITION_ACTIVE` | At least one candidate has a negative persistent penalty and at least one effective trace |

An empty recall can carry `NO_RETRIEVED_MEMORY` together with `HIGH_INTERFERENCE` or `RETRIEVAL_INHIBITION_ACTIVE` when that mechanism caused the suppression. Recovered matched traces stay observable as inactive matched inhibition and do not set `RETRIEVAL_INHIBITION_ACTIVE`.

## Out of scope

Competition matching, eligibility, interference calibration, inhibition decay, trace compaction, automatic clarification, and working-memory interference logic.
