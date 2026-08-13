# String cues, candidate generation, and current-state design reference

**Target release:** Cogkura `0.12`  
**Status:** Shipped  
**Primary implementation:** `src/cogkura/algorithms/activation.py`, `src/cogkura/algorithms/spreading.py`, `src/cogkura/memory.py`

## Summary

Cogkura `0.12` improves string-query recall when callers pass plain text instead of structured `RetrievalCue.entity_ids`. The release adds deterministic entity seeding for spreading, semantic slot admission before the retrieval threshold cut, superseded-slot penalties on supporting episodes, numeric-aware duplicate collapse, and explicit presented-vs-used access recording.

Seeding does **not** mutate caller cues. When `cue.entity_ids` is non-empty, behaviour matches `0.11`.

## Public API changes

| Method | Change |
|--------|--------|
| `recall(query, ...)` | String cues may seed spreading sources from candidate entity overlap; semantic slot admission may retain ACTIVE slot facts below threshold |
| `record_access(..., min_score=None)` | Skip rows below `min_score` or `ActivationConfig.access_minimum_score`; optional burst limiting |

`recall()` remains presentation-only — it does not append activation references. `record_access()` is use.

## String-cue entity seeding

When `enable_text_entity_seeding` is `true` (default) and the cue has text but empty `entity_ids`:

1. Collect candidate `entity_ids` whose tokenised form intersects cue tokens, or whose raw id appears as a cue token.
2. Pass the sorted seed to spreading only (via `spread_sources`); the caller's `RetrievalCue` is unchanged.
3. Apply a reduced entity partial-match weight (`seeded_entity_partial_match_weight`, default `0.75`) instead of the explicit-entity weight (`1.5`).

Hyphenated ids such as `charge-ledger` tokenise on non-alphanumerics; no alias table is required in `0.12`.

## Semantic slot admission

After scoring all candidates and before the threshold cut, `enable_semantic_slot_admission` (default `true`) may admit:

- `ACTIVE` semantics for matching slots when the cue requests current state, sets `predicate`, or has explicit/seeded entity overlap;
- their `SUPPORT` episodes from `StoredSemanticMemory.derivations`.

Admitted identities are ordered ahead of threshold survivors, then near-duplicate collapse and `limit` apply. Admitted rows may remain even when `activation < retrieval_threshold`.

Superseded semantics are loaded for derivation indexing only when `valid_at is None`; they are not presented unless historical retrieval already would include them.

## Current-state episode penalties

`ActivationComponents.current_state` reports the term separately from `partial_match`.

For current-state string cues, episodes that `SUPPORT` an `ACTIVE` slot receive the existing bonus; episodes that `SUPPORT` a `SUPERSEDED` slot receive a `-current_state_weight` penalty. Both apply when an episode supports multiple relations.

## Numeric duplicate collapse

When `collapse_normalize_numeric_tokens` is `true` (default), purely numeric tokens are stripped before Jaccard collapse only. IDF text similarity and fingerprint equality are unchanged.

## Access recording

- `access_minimum_score` defaults to `None` (no floor) for backward compatibility.
- `record_access(..., min_score=)` overrides the config default per call.
- `access_burst_limit` / `access_burst_window_seconds` optionally cap writes per identity within a sliding window; defaults leave `0.11` write behaviour.

## Configuration defaults (new in 0.12)

```text
enable_text_entity_seeding:          true
seeded_entity_partial_match_weight:  0.75
enable_semantic_slot_admission:      true
collapse_normalize_numeric_tokens:   true
access_minimum_score:                None
access_burst_limit:                  None
access_burst_window_seconds:         3600.0
```

## Notes

- No PostgreSQL migration is required.
- Structured `RetrievalCue` fields remain first-class; string cues are the compatibility path.
