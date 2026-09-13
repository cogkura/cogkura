# Behavioural competition eligibility and retrieval-scope isolation (0.17.2)

Shipped in Cogkura `0.17.2`.

## Summary

`0.17.2` separates **diagnostic competition** from **behaviourally eligible competition**. A detected competitor does not automatically have permission to suppress activation.

The `0.17.1` matcher remains unchanged and broad. A new deterministic policy gates which accepted pairs may contribute transient interference.

Default remains diagnostic-only: `CompetitionConfig.apply_interference=False`.

## Pipeline

```text
matcher.compare → CompetitionEvidence
  → BehavioralCompetitionPolicy.evaluate (directed, every accepted pair)
  → attach BehavioralCompetitionEligibility on evidence
  → if apply_interference: noisy-OR over eligible pairs only
```

`BehavioralQueryScope` is built once per retrieval from the cue (`subject_id`, `entity_ids`, `predicate`, canonical cue features).

## Eligibility tiers

1. **Same semantic slot** — `SAME_SEMANTIC_SLOT`
2. **Same fact subject and predicate** — `SAME_SUBJECT_PREDICATE`
3. **Query-scope anchor and shared query features** — `QUERY_ANCHORED_COMPETITION`

Reject reasons include weak strength/cue-fit (`minimum_behavioral_strength`, `minimum_behavioral_cue_fit`), `NO_QUERY_SCOPE_ANCHOR`, `NO_SHARED_QUERY_FEATURE`, `ENTITY_OVERLAP_ONLY`, `FEATURE_OVERLAP_ONLY`, `BROAD_SUBJECT_ONLY`, `SAME_LINEAGE`, and `NON_BEHAVIORAL_DIRECTION` (`CO_TEMPORAL`).

## Config gates

| `enabled` | `apply_interference` | Effect |
|-----------|----------------------|--------|
| `False` | * | No matching, eligibility, or interference |
| `True` | `False` | Diagnostics + eligibility reasons; recall scores unchanged |
| `True` | `True` | Diagnostics; interference from eligible pairs only |

`0.17.2` intentionally tightens the experimental `apply_interference=True` path from `0.17.1` — a correctness refinement, not a compatibility bug.

## Injection

```python
Memory(..., behavioral_competition_policy=DeterministicBehavioralCompetitionPolicy())
ACTRDeclarativeActivator(..., behavioral_competition_policy=...)
```

Default policy: `DeterministicBehavioralCompetitionPolicy()`. No per-call kwargs on `recall()` / `inspect_recall()`.

## Diagnostics

- `CompetitionEvidence.behavioral_eligibility`
- `CompetitionDiagnostics.behaviorally_eligible_competitor_count` / `behaviorally_rejected_competitor_count` (diagnostic `competitor_count` unchanged)
- `CompetitionRunDiagnostics.behaviorally_eligible_pairs` / `behaviorally_rejected_pairs` / `rejected_by_reason`
- Ineligible pairs do not appear in `TransientInterferenceDiagnostics.contributions`

## Out of scope

Persistent inhibition, co-temporal penalties, metamemory flags, working-memory algorithm changes, second interference formula, embeddings/LLMs.
