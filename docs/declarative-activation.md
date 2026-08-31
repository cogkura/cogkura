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
| `inspect_recall(query, tenant_id=..., ...)` | Inspect accepted and rejected candidates with activation diagnostics |
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
enable_lexical_slot_matching: true
lexical_slot_min_overlap:    1
semantic_soft_admission_floor: -4.0
max_soft_admitted_semantics: 8
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

- Internal `NEUTRAL` / `CURRENT` / `HISTORICAL` modes; `valid_at` selects semantic validity time (0.15.4+). Historical mode requires explicit historical cue intent.
- Historical mode can admit matching text/entity-derived semantic slots.
- Structured slot fit orders eligible candidates only; SUPPORT episodes inherit the slot they support.
- Metamemory answerability can emit `MISSING_KNOWLEDGE` when related retrieval is strong but the requested fact is unresolved.

## 0.14.3 conjunctive matching and positive structured ranking

See [`design-conjunctive-slot-ranking-0.14.3.md`](design-conjunctive-slot-ranking-0.14.3.md).

- Entity, predicate, and object constraints are conjunctive; object is not bypassed by predicate-only branches.
- Structured rank adjustment is centred at `slot_fit=None`; perfect fit is positive bounded evidence.
- SUPPORT episodes inherit fit only from the semantic slot they support.

## 0.14.4 retrieval diagnostics and support provenance

See [`design-retrieval-diagnostics-support-provenance-0.14.4.md`](design-retrieval-diagnostics-support-provenance-0.14.4.md).

- `RecallResult.diagnostics` captures rank activation, accessibility/ranking partial terms, eligibility, and provenance without changing scoring semantics.
- Accessibility activation and presentation score remain the public retrieval contract; rank-only terms remain diagnostic.
- SUPPORT diagnostics expose derivation-backed semantic revision provenance and selected inherited-fit source.

## 0.15.1 cognitive evidence chronology

- Base-level activation uses derived cognitive traces from episode `ended_at` and semantic support evidence, not storage `created_at`.
- `process()` does not rehearse or reinforce memory when called without new observations.
- `Memory.inspect_recall()` explains accepted and rejected candidates with terminal dispositions and trace detail.
- Persisted access (`RETRIEVED`, `REHEARSED`) and learning reinforcement traces are unchanged.

## 0.15.2 long-horizon semantic accessibility

- Plain-language cues can match semantic slots via lexical predicate/statement overlap.
- Relevant `ACTIVE` semantics below threshold may be soft-admitted within `semantic_soft_admission_floor` and `max_soft_admitted_semantics`.
- Structured `predicate` / `entity_ids` admission behaviour is preserved; the floor applies to text-only lexical admission.
- Current valid semantic facts can remain accessible when relevant without artificial rehearsal.

## 0.15.3 semantic authority and associative reachability

- **Authority** (`SemanticMemoryStatus.ACTIVE` vs `SUPERSEDED`) is separate from **accessibility** (activation) and **relevance** (cue fit).
- Cardinality-one competitors without explicit validity windows reconcile by supporting evidence chronology during consolidation.
- Rank-time evidence-linked relevance uses statements from supporting episodes already in the candidate set (bounded by `max_evidence_link_derivations`).
- `SEMANTIC_CURRENT_ADMISSION` admits `ACTIVE`, valid, relevant semantics when combined direct/evidence relevance meets `semantic_current_min_relevance`, without lowering `semantic_soft_admission_floor`.
- `inspect_recall()` exposes `direct_cue_fit`, `evidence_linked_fit`, `associative_fit`, `semantic_relevance`, and the admission eligibility path.

## 0.15.8 structured entity relationships

- Applications supply directed entity→entity edges via `ObservationInput.metadata["relationships"]` (same pattern as `semantic_facts`); malformed entries fail `observe()` with `ValidationError`.
- `EntityRelationshipStore` persists graph structure on observe (independent of `process()` and observation `UNCHANGED`); Postgres migration `008_entity_relationships.sql`.
- `Memory.list_entity_relationships(tenant_id=..., entity_id=...)` for inspection; `clear()` removes relationships before semantic memories.
- Query concept seeding: cue `entity_ids` (excluding `subject_id`) plus graph endpoints whose `canonical_content_features(entity_id)` are a subset of query features.
- Bounded forward/reverse traversal (`max_relationship_hops`, `max_relationship_neighbours`) with application relation-type weights reaches episodes/semantics through existing SUPPORT indexes.
- `RelevanceTier.STRUCTURED_RELATION` sits between entity association and evidence association; `semantic_relationship_min_relevance` gates soft admission.
- `AssociationPath` supports `seed_entity_id`, `hop_kind="relationship"`, and `relationship_edges`; `inspect_recall()` exposes `relationship_seed_count` and `relationship_paths_used`.
- CogKura stores and traverses supplied structure; it does not infer retail taxonomies. Optional enrichers (rules, parsers, LLMs) remain out of band.

Example non-retail ingest:

```python
metadata = {
    "relationships": [
        {
            "source_entity_id": "CheckoutService",
            "relation_type": "depends_on",
            "target_entity_id": "PaymentService",
            "provenance": "service-map",
        }
    ]
}
```

Re-run CogKuraBench/Demo externally: expect ~4/5 and ~3/5 until those fixtures supply catalog relationships.

## 0.15.7 contextual association

- Candidate-set association indexes are built per recall from eligible episodes and semantics (no new Postgres tables).
- Query-relevant seeds (`association_seed_min_relevance`, `max_association_seeds`) can bridge to semantics even when below `retrieval_threshold`.
- Entity recovery matches seed episode text tokens to known product entity ids without domain taxonomies.
- Three bounded bridges: direct entity hop (×0.5), entity-indexed episode hop (×0.5×0.5), and contextual evidence (×0.25); strongest path wins.
- Contextual hops require two distinctive shared features unless a single entity-label token qualifies (preserves shell-style evidence hops).
- `inspect_recall()` exposes `association_role` (`seed` / `bridge`), `association_seed_count`, `association_paths_used`, and extended `AssociationPath` fields.
- Bridge-only episodes stay `BELOW_THRESHOLD`; association does not write semantic derivations or confidence.

## 0.15.6 semantic competition and recall specificity

- Recall identity is cardinality-aware: `cardinality=one` groups by `slot_key`; `cardinality=many` groups by claim (`memory_key`) so distinct object values compete rather than collapse.
- Dedupe (same recall identity, near-duplicate statements) is separate from competition (distinct memories ranked by relevance specificity, then activation).
- `inspect_recall()` exposes `direct_value_fit`, `direct_predicate_fit`, `relevance_tier`, and collapse diagnostics (`collapse_key`, `collapse_reason`, `collapsed_into`).

## 0.15.5 deterministic semantic relevance and association

- Canonical retrieval content features (stopword filtering, contractions, morphology, punctuation normalization) drive semantic relevance instead of raw token ratios.
- Direct, evidence-linked, and associative relevance expose matched feature tuples and winning `AssociationPath` diagnostics on `inspect_recall()`.
- Evidence-to-evidence association links cue-matched episodes to related product episodes through qualifying shared context before reaching entity-linked semantics.
- CogKura retrieves what stored memory relationships can justify; optional LLM enrichment remains out of band from `Memory.recall()`.

## 0.15.4 temporal snapshot and associative reach

- **`as_of`** is the retrieval/evaluation clock; **`valid_at`** is the semantic validity clock; **temporal intent** (current vs historical) comes from the query, not from `valid_at` alone.
- `valid_at == as_of` with a normal recommendation query evaluates current semantic state at that snapshot (`CURRENT` mode), enabling `SEMANTIC_CURRENT_ADMISSION`.
- Evidence-linked relevance uses saturating aggregation across multiple supporting episodes.
- One-hop associative reach links cue-matched episode entities to semantics that share product/entity identifiers (including structured `object_value` prefixes).
- Evidence-linked association determines which memories may matter; it does not decide how they should affect the answer.

## Storage

Migration `008_entity_relationships.sql` adds `cogkura.entity_relationships`.

Postgres apps must pass `PostgresActivationStore` and `PostgresEntityRelationshipStore` alongside other Postgres stores.

`Memory.clear()` order: learning → activation → dynamics → relationships → semantic → episodic → observations.

## Roadmap

- `0.5` Spreading activation (see [`spreading-activation.md`](spreading-activation.md))
- `0.6` Forgetting / memory dynamics
- `0.7` Working memory / goal-aware attention
