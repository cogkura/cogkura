# Retrieval eligibility, global ranking, and temporal relevance design reference

**Target release:** Cogkura `0.14.0`  
**Status:** Shipped  
**Primary implementation:** `src/cogkura/algorithms/activation.py`, `src/cogkura/algorithms/relevance.py`, `src/cogkura/algorithms/working_memory.py`

## Summary

Cogkura `0.14.0` separates **eligibility** from **ranking** for semantic slot admission, restores soft entity-based admission as a threshold bypass only, decouples current-state lifecycle policy from admission, and adds precision-aware deterministic text matching for declarative partial match and working-memory goal relevance.

## Global ranking (Slice A)

Admitted identities no longer prepend the ranked list. All eligible candidates — threshold survivors and soft-admitted rows — compete in one global sort by `(-activation, memory_kind, memory_key)`.

Order: score → eligibility → current-state policy exclusions → global sort → duplicate/same-slot collapse → limit.

## Soft entity admission (Slice B)

`enable_entity_slot_admission` (default `true`) lets explicit `RetrievalCue.entity_ids` and text-seeded entity ids open slot admission when `slot_admission_requires_current_state_or_predicate` is `true`. Incident tags still seed spreading only.

Admission is threshold bypass only: no prepend, score bonus, or current-state policy activation.

## Current-state policy (Slice C)

`_current_state_policy_active` drives SUPERSEDED-only SUPPORT hard exclusion and lifecycle activation bias. Active when `valid_at is None` and the cue requests current state, sets `predicate`, or sets `object_value`. Entity admission alone does not activate policy.

## Temporal lifecycle bias (Slice D)

`_current_state_activation()` returns `0.0` when current-state policy is inactive. Historical `valid_at` recall receives no present-day ACTIVE/SUPERSEDED ranking bias.

## Precision-aware text matching (Slice E)

`enable_text_precision_matching` (default `true`) uses weighted F1 (query recall × candidate precision) inside ACT-R partial matching. When `false`, query-coverage behaviour from `0.13` is preserved.

## Working-memory text relevance (Slice F)

`calculate_cue_relevance` uses the same F1 principle for goal text overlap. Structured fields, inhibition, same-slot collapse, and metamemory union coverage are unchanged.

## Configuration defaults (new in 0.14)

```text
enable_entity_slot_admission:        true
enable_text_precision_matching:      true
```

Existing admission, conjunction, exclusion, and IDF fields are unchanged.

## Notes

- No PostgreSQL migration.
- No new `recall()` arguments.
- Metamemory thresholds unchanged; `MISSING_KNOWLEDGE` semantics preserved.
- `__version__` in `cogkura` matches `pyproject.toml` (`0.14.0`).
