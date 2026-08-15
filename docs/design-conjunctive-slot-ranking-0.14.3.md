# Conjunctive slot matching and positive structured ranking

**Target release:** Cogkura `0.14.3`  
**Status:** Shipped  
**Primary implementation:** `src/cogkura/algorithms/activation.py`

## Summary

Cogkura `0.14.3` keeps the `0.14.2` accessibility architecture and corrects two structured-retrieval defects:

1. explicit entity, predicate, and object constraints are conjunctive in shared semantic matching;
2. perfect structured slot fit provides positive bounded ranking evidence relative to candidates with no applicable slot fit.

## Conjunctive structured matching (Slice A)

`_semantic_fields_match_cue` evaluates all explicit structured fields as a contract:

- `cue.predicate` must match when supplied;
- `cue.object_value` must match when supplied;
- `cue.entity_ids` must overlap candidate entities when supplied.

A predicate match no longer bypasses entity or object constraints. Distinctive text cannot rescue an explicit structured mismatch.

When no explicit entity, predicate, or object is supplied, seeded entities and distinctive current-state tokens continue to locate slots as in prior releases.

Admission, current-state slot matching, structured slot fit, and metamemory resolution share `_stored_semantic_matches_cue` / `_semantic_fields_match_cue`.

## Positive structured ranking (Slice B)

Structured slot fit uses discrete values:

- `None` — not applicable (neutral associative entity-only queries, or temporal mode without predicate/object and without a usable entity locator);
- `1.0` — required structured constraints match and temporal state is compatible;
- `0.0` — a required constraint fails or temporal state is incompatible;
- `0.5` — reserved for genuinely indeterminate temporal compatibility (not used in typical paths).

Rank adjustment reuses the existing mismatch scale, centred at neutral:

```text
structured_adjustment = mismatch_penalty * (slot_fit - 0.5)
```

Perfect fit (`1.0`) yields `+0.5 × mismatch_penalty`. No applicable fit (`None`) yields `0`. Mismatch (`0.0`) yields `-0.5 × mismatch_penalty`.

SUPPORT episodes inherit fit only from semantic slots they actually support. Diagnostics may include `slot_fit_source=support` on inherited episodes.

`RecallResult.activation`, threshold eligibility, latency, and `record_access` semantics are unchanged.

## Configuration

No new public fields. No `slot_fit_weight`.

## Notes

- No PostgreSQL migration.
- No new `recall()` / `assess_memory()` arguments.
- Working memory, forgetting, learning, spreading, and `conjunction_weight` are unchanged.
- Metamemory thresholds and answerability states are unchanged.
- `__version__` in `cogkura` matches `pyproject.toml` (`0.14.3`).
