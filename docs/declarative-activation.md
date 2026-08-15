# Declarative activation (ACT-R) design reference

**Target release:** Cogkura `0.4`  
**Status:** Shipped (base-level + partial matching)

## Summary

Cogkura `0.4` replaces observation token-overlap `recall()` with ACT-R-inspired **declarative activation** over durable episodic and semantic memories.

Activation combines:

- **Base-level** frequency/recency from access references
- **Partial matching** deterministic cue similarity (mismatch penalties)
- **Spreading activation** — shipped in `0.5`; see [`spreading-activation.md`](spreading-activation.md)

\[
A_i = B_i + S_i + P_i + \epsilon_i
\]

In `0.4` without spreading, \(S_i = 0\) and \(\epsilon_i = 0\) by default.

## Public API

| Method | Purpose |
|--------|---------|
| `recall(query, tenant_id=..., ...)` | Rank episodic + semantic memories by activation |
| `record_access(results, tenant_id=...)` | Explicitly reinforce selected memories |

- `recall()` is pure — it does not mutate access history. `record_access()` records **use**; pass `min_score=` or set `access_minimum_score` to skip weak presented rows.

### Breaking change

`RecallResult` now references `StoredEpisode | StoredSemanticMemory`, not `StoredObservation`. Raw observations are evidence, not declarative recall candidates.

## Configuration defaults

```text
decay:                  0.5
time_unit_seconds:      3600.0   # one hour per activation unit
retrieval_threshold:   -3.0
enable_spreading_activation: true
enable_partial_matching:     true
enable_candidate_idf:        true
enable_duplicate_collapse:   true
duplicate_jaccard_threshold: 0.75
current_state_weight:        0.5
```

Spreading defaults and behaviour: [`spreading-activation.md`](spreading-activation.md).

## 0.11 ranking improvements

See [`design-ranking-time-current-state.md`](design-ranking-time-current-state.md).

- Candidate-set IDF downweights corpus-constant tokens in text partial match (`enable_candidate_idf`, default `true`).
- Near-duplicate collapse after scoring (`enable_duplicate_collapse`, `duplicate_jaccard_threshold=0.75`).
- Current-state activation bias for active semantics and configured cue tokens (`current_state_weight`, `current_state_cue_tokens`).
- `recall(..., valid_at=...)` excludes episodes with `started_at > valid_at`.

## 0.12 string cues and current-state

See [`design-string-cues-current-state.md`](design-string-cues-current-state.md).

- String cues seed spreading sources from candidate entity overlap (`enable_text_entity_seeding`, default `true`).
- Seeded entity partial match uses `seeded_entity_partial_match_weight` (default `0.75`).
- Semantic slot admission retains matching `ACTIVE` slot facts and `SUPPORT` episodes before the threshold cut.
- Episodes supporting superseded slots receive a current-state penalty; term logged on `ActivationComponents.current_state`.
- Near-duplicate collapse strips purely numeric tokens before Jaccard (`collapse_normalize_numeric_tokens`).
- `record_access(..., min_score=...)` and optional burst limits filter reinforcement writes.

## 0.13 gated admission and association

See [`design-gated-slot-admission.md`](design-gated-slot-admission.md).

- Slot admission requires current-state cue tokens, `predicate`, or `force_slot_admission`; entity ids seed spreading only.
- Multi-entity conjunction bonus when two or more cue entities overlap a candidate.
- Incident cue IDF scaling and optional tag seeding from episode metadata.
- SUPERSEDED-only SUPPORT episodes are excluded on current-state recall when `valid_at is None`.
- Same-slot SUPPORT collapse at rank time and in working memory.

## 0.14 retrieval eligibility and global ranking

See [`design-retrieval-eligibility-ranking.md`](design-retrieval-eligibility-ranking.md).

- Admitted candidates compete in one global activation sort; admission bypasses threshold only.
- `enable_entity_slot_admission` restores soft entity-based slot admission without rank priority.
- Current-state policy (live cue, predicate, or object) drives lifecycle bias and superseded SUPPORT exclusion.
- `enable_text_precision_matching` uses weighted F1 for declarative partial match and WM goal text relevance.

## 0.14.1 retrieval corrections

See [`design-retrieval-corrections-0.14.1.md`](design-retrieval-corrections-0.14.1.md).

- Lexical current-state tokens are detected even when `predicate` or `object_value` is set.
- Historical admission does not require present-day `ACTIVE` status.
- Query coverage controls accessibility / threshold; F1 controls eligible ranking only.
- Current-state bonuses are scoped to matched semantic slots and their SUPPORT episodes.

## 0.14.2 temporal mode, slot fit, and answerability

See [`design-temporal-slot-answerability-0.14.2.md`](design-temporal-slot-answerability-0.14.2.md).

- Internal `NEUTRAL` / `CURRENT` / `HISTORICAL` modes; `valid_at` always selects HISTORICAL.
- Historical mode can admit matching text/entity-derived semantic slots.
- Structured slot fit orders eligible candidates only; SUPPORT episodes inherit the slot they support.
- Metamemory answerability can emit `MISSING_KNOWLEDGE` when related retrieval is strong but the requested fact is unresolved.

## Storage

Migration `004_declarative_activation.sql` adds `cogkura.memory_activation_references`.

Postgres apps must pass `PostgresActivationStore` alongside other Postgres stores.

`Memory.clear()` order: activation → semantic → episodic → observations.

## Roadmap

- `0.5` Spreading activation (see [`spreading-activation.md`](spreading-activation.md))
- `0.6` Forgetting / memory dynamics
- `0.7` Working memory / goal-aware attention
