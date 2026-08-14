# Gated slot admission, association, and metamemory design reference

**Target release:** Cogkura `0.13`  
**Status:** Shipped  
**Primary implementation:** `src/cogkura/algorithms/activation.py`, `src/cogkura/algorithms/metamemory.py`, `src/cogkura/algorithms/working_memory.py`, `src/cogkura/memory.py`

## Summary

Cogkura `0.13` tightens when semantic slot admission runs, improves episodic association for multi-entity cues, hard-excludes superseded-only SUPPORT episodes on current-state recall, adds a metamemory `MISSING_KNOWLEDGE` flag, and collapses duplicate slot SUPPORT in rank and working memory.

Entity ids and incident tag seeding remain **spreading and partial-match seeds only** — they do not open slot admission unless `force_slot_admission` is set or `slot_admission_requires_current_state_or_predicate` is disabled.

## Gated slot admission (Slice A)

Admission requires `enable_semantic_slot_admission` and one of:

- `force_slot_admission` (`false` by default — test/debug override)
- `slot_admission_requires_current_state_or_predicate=false`
- current-state cue tokens (`currently`, `current`, `now`, `live`, `today`, …)
- structured `RetrievalCue.predicate`

Once admission is active, matching uses the `0.12` rules: predicate match, entity overlap, or distinctive token overlap with ACTIVE slot statements/objects. SUPPORT episodes for admitted ACTIVE slots are prepended with the semantic row.

## Multi-entity conjunction (Slice B)

When `enable_multi_entity_conjunction` is `true` (default) and the effective cue entity set has at least two ids, candidates sharing two or more of those ids receive `conjunction_weight` (default `0.5`) added to activation. Spreading already scores single-hop neighbours; no second conjunction term is added for one entity plus a spread neighbour.

## Distinctive episodic match (Slice C)

- `distinctive_token_idf_scale` (default `1.5`) scales candidate-set IDF weights for cue tokens in `incident_cue_tokens`.
- `enable_incident_tag_seeding` (default `true`) unions episode metadata tag tokens (`metadata["tags"]` or `metadata["episode"]["tags"]`) into seeding sources like entity ids. Tags do not admit slots.

## Hard superseded exclusion (Slice D)

When `exclude_superseded_support_on_current_state` is `true`, slot admission is active, and `valid_at is None`, episode candidates whose support index is SUPERSEDED-only are dropped before collapse/`limit`. Episodes supporting both ACTIVE and SUPERSEDED slots are kept. Historical `valid_at` listing behaviour is unchanged.

## Metamemory presented ≠ known (Slice E)

`MemoryAssessmentFlag.MISSING_KNOWLEDGE` is set when **all** of:

1. No retrieved ACTIVE semantic matches the cue predicate, entity overlap, or current-state slot tokens.
2. `cue_coverage < missing_knowledge_coverage_threshold` **or** `top_retrieval_strength < missing_knowledge_strength_threshold`.

A full weak recall pool still flags. Empty pools continue to emit `NO_RETRIEVED_MEMORY` only.

## Working-memory precision (Slice F)

- `WorkingMemoryConfig.collapse_same_slot_support` (default `true`) drops extra SUPPORT rows sharing a semantic `slot_key`, keeping ACTIVE semantics and one SUPPORT episode.
- Goal text containing `stale` (including `ignore stale`) applies `stale_goal_penalty` to metadata tags containing `stale` and to SUPERSEDED semantic rows.

Working memory still does not call `record_access()`.

## Rank-time same-slot collapse (Slice G)

`rank()` extends near-duplicate collapse with same-slot SUPPORT collapse using `ActivationConfig.collapse_same_slot_support`. ACTIVE semantic rows are kept alongside one SUPPORT episode for the slot; duplicate SUPPORT siblings collapse.

## Configuration defaults (new in 0.13)

```text
slot_admission_requires_current_state_or_predicate: true
force_slot_admission:                               false
enable_multi_entity_conjunction:                    true
conjunction_weight:                                 0.5
distinctive_token_idf_scale:                        1.5
enable_incident_tag_seeding:                        true
exclude_superseded_support_on_current_state:        true
collapse_same_slot_support:                         true
missing_knowledge_coverage_threshold:               0.35
missing_knowledge_strength_threshold:               0.45
stale_goal_penalty:                                 0.35
```

## Notes

- No PostgreSQL migration.
- No new `recall()` arguments; `valid_at` is passed through to `rank()` internally.
- `__version__` in `cogkura` matches `pyproject.toml` (`0.13.0`).
