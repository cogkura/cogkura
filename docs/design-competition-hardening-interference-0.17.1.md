# Competition hardening and transient interference (0.17.1)

Shipped in Cogkura `0.17.1`.

## Summary

`0.17.1` hardens `0.17.0` cue-competition matching and introduces the first **behavioural** competition mechanism: bounded, retrieval-local **transient interference** when explicitly enabled.

Default remains diagnostic-only: `CompetitionConfig.apply_interference=False`.

## Hardening

- **Subject compatibility** uses structured query anchors (`cue.entity_ids`), exact `subject_entity_id`, or cue subject against fact subjects — not broad tenant `subject_id` alone when entities conflict.
- **Semantic effective time** prefers `valid_from`, else latest visible SUPPORT episode `started_at` from retrieval-scoped `episode_by_id`, else `last_supported_at`.
- Same-lineage exclusion and superseded semantics rules are unchanged.

## Shared evaluation

`evaluate_competition()` operates on `RecallResult` and is invoked from both `rank()` and `inspect()` after admission annotation and before threshold/sort. One pass; competitor accessibility is frozen from pre-interference activation.

## Transient interference

When `enabled=True` and `apply_interference=True`:

```text
S(j) = sigmoid(A_pre(j) - τ)
p(j→i) = strength(i,j) × S(j)
P_dir(i) = 1 - ∏(1 - p)   # noisy-OR over bounded competitors
I_dir(i) = - weight_dir × P_dir
I(i) = I_proactive + I_retroactive ≤ 0
A_final = A_pre + I
```

- `CO_TEMPORAL` evidence contributes zero pressure.
- Semantic admission from pre-interference activation is preserved; interference may drop threshold-only candidates and reorder.
- `RetrievalDiagnostics` records pre-interference activation and threshold-crossing flags.
- `RecallInspectionCandidate` records interference rank deltas.

## Config gates

| `enabled` | `apply_interference` | Effect |
|-----------|----------------------|--------|
| `False` | * | No matching, diagnostics, or interference |
| `True` | `False` | Hardened diagnostics; recall unchanged aside from matcher precision |
| `True` | `True` | Diagnostics + transient interference on `rank()` and `inspect()` |

## Out of scope

Persistent inhibition, co-temporal penalties, metamemory flags, working-memory-specific interference, embeddings/LLMs.
