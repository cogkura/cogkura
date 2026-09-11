# Competition representation and diagnostics (0.17.0)

## Status

Shipped in Cogkura `0.17.0`.

## Summary

`0.17.0` introduces **observational cue-competition diagnostics** on `inspect_recall`. Competition identifies which retrieved memories plausibly satisfy the same cue, explains why, and classifies competitors as proactive, retroactive, or co-temporal.

Competition is **diagnostic only** in `0.17.0`. It does not change activation, admission, ranking, working-memory selection, forgetting, reconsolidation, or persistent state.

This is distinct from `0.15.6` **relevance-tier ranking competition** among distinct semantics. That ranking behaviour is unchanged.

## Public surfaces

| Surface | Role |
|---------|------|
| `RecallInspectionCandidate.competition` | Per-candidate `CompetitionDiagnostics` |
| `RecallInspectionResult.competition` | Retrieval-level `CompetitionRunDiagnostics` counters |
| `CompetitionConfig` on `Memory(...)` | Enable/disable and engineering thresholds |
| `CompetitionEvidence` | Pairwise strength, direction, cue fits, and structural evidence |

When `CompetitionConfig.enabled=False`, competition diagnostics are omitted (`None`).

## Matching model

Competition strength is deterministic and explainable:

```text
competition_strength = relationship_strength × joint_cue_fit
joint_cue_fit = min(candidate effective cue fit, competitor effective cue fit)
```

Relationship tiers (precision over recall):

1. Same semantic `slot_key` → strongest
2. Compatible fact subject + same `predicate` → strong
3. Compatible subject + entity/feature overlap → moderate
4. Entity overlap without subject compatibility → rejected

Effective cue fit reuses existing retrieval diagnostics (`max(semantic_relevance, text_cue_fit)`) and multiplies by existing context correspondence when retrieval context is populated.

Effective memory time precedence:

- Episode: `started_at` else `created_at`
- Semantic: `valid_from` else `last_supported_at` else `created_at`

## Invariants

- Competition is retrieval-local; no persisted competition relationships.
- Candidate universe is the existing inspect discrimination set after `valid_at`, supersession, and collapse rules.
- Support-lineage pairs (semantic memory and its SUPPORT episodes) do not compete.
- `rank()` is unchanged; analysis runs in `inspect()` post-pass only.

## Out of scope (`0.17.0`)

Interference penalties, inhibitory traces, metamemory interference flags, and working-memory-specific interference belong to later `0.17.x` releases.
