# Retrieval corrections and ranking separation design reference

**Target release:** Cogkura `0.14.1`  
**Status:** Shipped  
**Primary implementation:** `src/cogkura/algorithms/activation.py`

## Summary

Cogkura `0.14.1` keeps the `0.14` eligibility architecture (soft admission, global ranking, policy vs admission) and corrects four interactions: lexical current-state detection, historically aware admission, query-coverage accessibility vs F1 ranking, and slot-scoped current-state bonuses.

## Lexical current-state intent (Slice A)

`_cue_requests_current_state` inspects query text only. Structured `predicate` or `object_value` no longer suppress tokens such as `current`, `currently`, `now`, `live`, `today`.

`current_state_policy_active` remains separate: live retrieval (`valid_at is None`) AND (lexical current-state cue OR predicate OR object).

## Historical admission (Slice B)

`_semantic_slot_admission_identities` receives `valid_at`. Live admission still requires `ACTIVE`. Historical admission uses the already-visible revision and does not reject present `SUPERSEDED` status. SUPPORT admission follows the admitted revision.

## Coverage accessibility vs F1 ranking (Slice C)

Query coverage (`_text_query_coverage`) drives ACT-R partial matching, `retrieval_threshold`, `RecallResult.activation`, and `RecallResult.score`.

When `enable_text_precision_matching` is `true` (default), weighted F1 (`_text_cue_fit`) is used only for an internal `rank_activation` sort among eligible candidates. When `false`, ranking uses coverage as well.

F1 never independently removes a candidate from eligibility. Admission remains threshold bypass only.

## Slot-scoped current-state activation (Slice D)

Matching semantic slots are identified independently of admission (predicate, entity, or distinctive current-state tokens). Current-state bonuses and SUPPORT terms apply only to matched slots and their SUPPORT episodes. Unrelated `ACTIVE` semantics receive no slot-specific bonus. SUPERSEDED-only SUPPORT exclusion for live current-state retrieval is unchanged.

## Configuration

No new public fields. `enable_text_precision_matching` now means: coverage for accessibility, F1 for ordering.

## Notes

- No PostgreSQL migration.
- No new `recall()` arguments.
- Working memory, metamemory thresholds, forgetting, and `conjunction_weight` are unchanged.
- `__version__` in `cogkura` matches `pyproject.toml` (`0.14.1`).
