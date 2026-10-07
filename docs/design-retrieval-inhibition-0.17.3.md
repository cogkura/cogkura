# Retrieval-induced inhibitory traces (0.17.3)

Shipped in Cogkura `0.17.3`.

> Inhibitory traces are recorded but are not consulted during retrieval.

`recall`, `inspect_recall`, `prepare_context`, `select_working_memory`, and `assess_memory` stay read-only. Only `record_access` and `record_context_use` may create traces, and only when `InhibitionConfig.enabled` is true.

## Scope eligibility vs transient interference

`BehavioralCompetitionEligibility.scope_eligible` is the 0.17.2 precision gate (lineage, strength, cue-fit, and structural tier).

`eligible` remains the transient-interference gate: `scope_eligible` and direction `PROACTIVE` or `RETROACTIVE`. A credible co-temporal pair can be `scope_eligible=True` with `eligible=False` and `reason=NON_BEHAVIORAL_DIRECTION`. Transient interference is unchanged.

## Write path

```text
retrieval
  -> competition
  -> scope eligibility
  -> RecallResult.diagnostics.inhibition_candidates
  -> record_access / record_context_use
  -> unselected scope-eligible competitor above minimum pressure
  -> InhibitoryTrace
```

Snapshots are frozen from the original competition evaluation. `record_access` does not recall again. Accessibility on the snapshot is the pre-interference presentation score. Raw query text is not stored. `scope_key` is a SHA-256 of the canonical scope.

`InhibitionConfig.enabled` requires `CompetitionConfig.enabled`. It does not require `apply_interference`.

Activation references and inhibitory traces are separate store calls. A repeated `request_id` for the same selected identity, inhibited identity, and scope is a no-op in both stores, so a retry converges. There is no cross-store transaction in this release.

`0.17.3.1` separates the two write sets inside `record_access()`:

```text
consumed memories
    results after min_score / access_minimum_score
    used to decide selective-retrieval inhibition

reinforced memories
    consumed memories that also pass positive-reference burst limiting
    used for activation references and forgetting reactivation
```

`min_score` determines whether a result is considered used. Burst limiting does not. An empty reinforcement set does not skip inhibition recording, so a retry can still persist the trace after the positive reference is already present.

## Out of scope

Applying traces to activation, decay, aggregation, compaction, expiry, metamemory flags, and traces created by presentation or working-memory rejection.
