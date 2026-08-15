# Temporal mode, structured slot fit, and metamemory answerability

**Target release:** Cogkura `0.14.2`  
**Status:** Shipped  
**Primary implementation:** `src/cogkura/algorithms/activation.py`, `src/cogkura/algorithms/metamemory.py`

## Summary

Cogkura `0.14.2` keeps the `0.14.1` accessibility architecture (coverage for the threshold, F1 for ordering, admission as threshold bypass) and adds three narrower concepts: an internal temporal retrieval mode, ordering-only structured semantic-slot fit, and metamemory answerability.

Accessibility still answers whether a memory can come to mind. Structured and temporal fit answer whether it resolves the query. Answerability answers whether Cogkura possesses a resolved fact.

## Temporal retrieval mode (Slice A)

`TemporalRetrievalMode` is internal (`neutral`, `current`, `historical`). `_temporal_retrieval_mode` is computed once per `rank()`:

- `valid_at is not None` → `HISTORICAL`, including when the text contains `current` / `live` / `now`
- else lexical current-state cue or structured live-slot policy (`predicate` / `object_value`) → `CURRENT`
- else → `NEUTRAL` (entity-only queries stay neutral)

`_current_state_policy_active` is `temporal_mode is CURRENT`. Present ACTIVE/SUPERSEDED status does not bias HISTORICAL ranking.

HISTORICAL mode can activate semantic-slot soft admission. Admission still requires `_semantic_slot_matches_cue`. Text-derived entity seeds can admit a historically visible matching slot. Unrelated historical semantics are not broadly admitted. Present `SUPERSEDED` still does not block historical admission.

## Structured semantic-slot fit (Slice B)

`_semantic_slot_fit` is an ordering-only signal over dimensions actually present in the cue: entity/subject, predicate, object, and temporal compatibility when the mode is not `NEUTRAL`.

- CURRENT: `ACTIVE` is temporally compatible
- HISTORICAL: a revision already in the `valid_at` pool is temporally compatible
- NEUTRAL without predicate/object: `slot_fit=None` so associative entity queries are not penalised

SUPPORT episodes inherit fit only from the semantic revision they actually support.

Rank score:

```text
rank_activation = accessibility - coverage_PM + F1_PM + mismatch_penalty * (slot_fit - 1.0)
```

(`slot_fit is None` → no adjustment.) Perfect structured fit keeps accessibility/rank strength; mismatch is a bounded demotion. No `slot_fit_weight`. `RecallResult.activation`, threshold eligibility, latency, and `record_access` scores are unchanged.

## Metamemory answerability (Slice C)

Internal `MemoryAnswerability`: `resolved`, `unresolved`, `not_applicable`.

Answerability applies to slot-like queries (`predicate`, `object_value`, CURRENT, or HISTORICAL). Exploratory NEUTRAL recall is `NOT_APPLICABLE` and keeps the existing coverage/strength `MISSING_KNOWLEDGE` fallback.

- `RESOLVED`: a temporally valid semantic assertion matches the cue slot and asserts an object/value. CURRENT requires present `ACTIVE`. HISTORICAL uses `valid_at` visibility.
- `UNRESOLVED`: slot-like query with related memories but no resolving semantic assertion → `MISSING_KNOWLEDGE` even when coverage and retrieval strength are high
- Matching uses the same `_stored_semantic_matches_cue` primitive as recall

Monitoring flags and metamemory thresholds are unchanged.

## Configuration

No new public fields. No `historical_weight`, `slot_fit_weight`, or answerability threshold.

## Notes

- No PostgreSQL migration.
- No new `recall()` / `assess_memory()` arguments.
- Working memory, forgetting, learning, spreading, and `conjunction_weight` are unchanged.
- `__version__` in `cogkura` matches `pyproject.toml` (`0.14.2`).
