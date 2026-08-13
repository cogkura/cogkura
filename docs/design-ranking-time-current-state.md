# Ranking, time, and current-state design reference

**Target release:** Cogkura `0.11`  
**Status:** Shipped  
**Primary implementation:** `src/cogkura/memory.py`, `src/cogkura/algorithms/activation.py`, `src/cogkura/algorithms/forgetting.py`

## Summary

Cogkura `0.11` improves declarative recall on longitudinal fixtures by fixing simulated time, episode visibility at `valid_at`, cue weighting, near-duplicate collapse, current-state semantic bias, and importance-aware forgetting.

The public `Memory` API remains the contract. Benchmark consumers should not post-filter recall results.

## Public API changes

| Method | Change |
|--------|--------|
| `encode_episodes(..., as_of=None)` | Optional simulated write time for `created_at` / `updated_at` |
| `consolidate_semantics(..., as_of=None)` | Reconciler and store writes use `as_of` instead of wall clock |
| `recall(..., valid_at=...)` | Episodes included only when `started_at <= valid_at` |
| `apply_forgetting(..., as_of=...)` | Episodes with `started_at > as_of` are excluded from evaluation |

Default `as_of=None` keeps live behaviour (`datetime.now(UTC)`).

## Simulated time

When `as_of` is set on encode or consolidate:

- In-memory and Postgres stores stamp `created_at` / `updated_at` from `as_of`.
- Semantic reconciliation `valid_from` uses the reconciler `as_of`.
- `recall(..., as_of=T)` does not raise when memory `created_at` equals `T`.

## Episode visibility at `valid_at`

Semantic memories already honour revision windows via `valid_at`. Episodes now use:

```text
include episode iff episode.started_at <= valid_at
```

`ended_at` is episode duration, not a visibility cutoff.

## Ranking improvements

### Candidate-set IDF

Text partial match downweights tokens that appear in many candidates in the current pool:

```text
idf(t) = log((N + 1) / (df(t) + 1)) + 1
```

Controlled by `ActivationConfig.enable_candidate_idf` (default `true`).

### Near-duplicate collapse

After scoring, greedy collapse on statement token Jaccard (default threshold `0.75`) and exact `content_fingerprint` equality. Controlled by `ActivationConfig.enable_duplicate_collapse` (default `true`).

### Current-state bias

Semantics with `status=ACTIVE`, later `last_supported_at`, and non-superseded slot values receive a deterministic activation bonus. String cues containing configured tokens (`currently`, `current`, `now`, `live`, `today`) receive extra bias via `ActivationConfig.current_state_weight`.

Structured `RetrievalCue` fields (`predicate`, `object_value`, `entity_ids`) remain first-class.

## Forgetting changes

`EbbinghausForgettingEvaluator` still derives lifecycle from ACT-R base level only.

New `ForgettingConfig` options (defaults on):

- `enable_importance_scaling` — low-importance memories fade faster via retention scaling.
- `importance_retention_floor` — minimum retention multiplier (default `0.15`).
- `protect_semantic_support` — episodes that `SUPPORT` a non-superseded semantic memory cannot reach `FORGOTTEN` (capped at `FADING`).

Spreading activation and partial matching are still excluded from forgetting evaluation.

## Acceptance (library tests)

- Observe → encode/consolidate with `as_of=T0` → recall with `as_of=T0` succeeds.
- Episode with `started_at` after `valid_at` is absent from recall.
- Distinctive query tokens outrank repeated project-name filler when IDF is enabled.
- Near-duplicate paraphrases cannot fill all top ranks when a distinct episode exists.
- Protected supporting episodes remain at `FADING`, not `FORGOTTEN`.

Bench verification against named Atlas queries uses unchanged gold IDs after a `0.11` release.

## Out of scope

- Embedding-based retrieval.
- Post-hoc bench filtering of Cogkura hits.
- Dataset filler rewrites in CogKuraBench.
