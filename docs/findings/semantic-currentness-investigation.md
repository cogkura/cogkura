# Semantic currentness investigation (0.15.10)

**Status:** evidence only. No retrieval, forgetting, reinforcement, or working-memory algorithm change.

**Question:** why can an old cardinality-many interest remain `ACTIVE`, `temporal_mode=current`, and `eligibility=semantic_current_admission` when a later interest is stronger and a structured path still makes the old fact query-relevant?

**Package fixture:** [`tests/test_semantic_currentness_investigation.py`](../../tests/test_semantic_currentness_investigation.py) (Demo-free: `activity_interest=activity-a` vs `activity-b`, query clock `2026-08-01T12:00Z`).

## Summary

On shipped 0.15.10 this is **intended Core behaviour**, not a threshold bug.

Cardinality-many values stay independently `ACTIVE` until the application supplies `valid_until`, polarity/contradiction, or reconsolidation to `SUPERSEDED`. Forgetting changes `MemoryRetentionState` on dynamics (here `fading`), not `SemanticMemoryStatus`. `SEMANTIC_CURRENT_ADMISSION` then **bypasses** `retrieval_threshold` (default `-3.0`) for live, non-historical, query-relevant semantics whose activation is **below** that floor. Structured relationships are one relevance gate among several; combined/lexical/associative gates can admit the same fact without a graph.

Working memory does not invent currentness. Defaults `minimum_goal_relevance=0.0` and `minimum_selection_score=0.0`, so an admitted low-ranked semantic is selected while chunk capacity remains.

Do not remove current admission without a replacement for long-horizon current facts (protected by `test_ninety_day_current_size_admitted_without_floor` and `test_old_hiking_interest_current_admitted_with_hiking_in_query`). Do not “fix” this by moving `retrieval_threshold` or `semantic_soft_admission_floor`.

**Decision:** **A** for Core 0.15.10 (no algorithm patch). **D** explains Demo-labelled “stale skiing” if the application still emits a durable `activity_interest=skiing` fact. **C** only if a later product requirement is Core-owned gradual loss of currentness for `cardinality=many` without explicit validity. **Not B.**

## Observed behaviour (measured)

Clock: evidence for A at `_T-150d` (`2026-03-04`), B at `_T-20d` / `_T-15d` (`2026-07-12` / `2026-07-17`), query `as_of=valid_at=_T`. Query text: `Recommend outerwear for the customer.` Graph (optional): `activity-a-product IS_A outerwear`. After `apply_forgetting(..., as_of=_T)`.

| Field | Weak `activity-a` (1 browse) | Strong `activity-b` (purchase + repeat) |
|-------|------------------------------|-----------------------------------------|
| `status` / `cardinality` | `ACTIVE` / `many` | `ACTIVE` / `many` |
| `support_count` | 1 | 2 |
| `confidence` | 0.44 | 0.66 |
| `importance` | 0.40 | 0.40 |
| `valid_from` / `valid_until` | none | none |
| `contradiction_count` | 0 | 0 |
| `learn()` counts | 0 | 0 |
| chronology | `first_supported_at` = `created_at` = episode `ended_at` = `2026-03-04` | first `2026-07-12`, last `2026-07-17` |
| activation / `retrieval_threshold` | **-3.91** / **-3.0** (below floor) | **-2.14** / **-3.0** (above floor) |
| `base_level` / `rank_activation` | -4.09 / -3.45 | -2.32 / -1.69 |
| spreading | 0.0 | 0.0 |
| dynamics | `fading`, retention 0.12 | `active`, retention 0.33 |
| eligibility | `semantic_current_admission` | `entity_slot_admission` |
| `soft_admitted` / retrieved / selected | true / true / true | true / true / true |
| `semantic_relevance` | 0.33 | 0.56 |
| `structured_association_fit` | 0.20 | 0.00 |
| `relevance_tier` | `evidence_association` | `evidence_association` |
| reconstructed gates | combined, lexical_evidence, associative_fit, structured_fit, structured_path | combined, lexical_evidence |

Inspection on the outerwear query: 6 considered, 4 returned, 2 rejected; `relationship_seed_count=1`, `relationship_paths_used=1`. Working memory: `selected_count=2`, `candidate_count=4`, `prompt_budget_tokens=2048` (defaults `max_items=8`).

Current admission applies only when activation is **below** threshold. The stronger interest is retrieved by a different eligibility path because it already clears the ACT-R floor. Admission is not a ranking boost and does not encode “weak vs strong”.

## Core lifecycle (do not change in this investigation)

### Authority vs accessibility vs relevance

Documented in [`docs/declarative-activation.md`](../declarative-activation.md) (0.15.3): `SemanticMemoryStatus` is authority; activation is accessibility; cue fit is relevance. `SEMANTIC_CURRENT_ADMISSION` exists so current authoritative facts remain reachable when relevant even if \(B_i\) is old.

Implementation: `_semantic_meets_current_relevance` and `_authoritative_semantic_current_admission_identities` in `src/cogkura/algorithms/activation.py`. A semantic is current-admitted when **all** of:

1. Cue is not historical.
2. Status is not `SUPERSEDED`; live `ACTIVE` (or valid-at window).
3. At least one relevance gate fires (see below).
4. Combined activation is **strictly below** `retrieval_threshold` (default `-3.0`).
5. Rank among such matches is within `max_soft_admitted_semantics` (default 8).

`semantic_soft_admission_floor` (default `-4.0`) is **not** required for this path (0.15.3).

### Relevance gates

`_semantic_meets_current_relevance` is true if any of:

- `combined >= semantic_current_min_relevance` (0.06)
- lexical overlap on direct or evidence features (`lexical_slot_min_overlap`, default 1)
- bridge entity intersection
- `associative_fit >= contextual_association_min_relevance` (0.06)
- `structured_fit >= semantic_relationship_min_relevance` (0.06)

Existing `RetrievalDiagnostics` already expose the inputs (`semantic_relevance`, matched features, `associative_fit`, `structured_association_fit`, `association_path`). This investigation reconstructs gate names in tests; **no new production diagnostic fields** were added.

### Cardinality-many

Hiking and skiing already coexist as independent `ACTIVE` values (`test_many_cardinality_interests_coexist` in `tests/test_semantic_state_associative_recall.py`). There is no gradual “less current” state for `MANY` short of explicit `valid_until`, polarity, or reconsolidation.

### Forgetting

[`docs/forgetting.md`](../forgetting.md): `ACTIVE → FADING → FORGOTTEN` on `StoredMemoryDynamics`, driven by base-level only. `protect_semantic_support` caps SUPPORT episodes at `FADING`. Forgetting lowers activation; it does **not** flip `SemanticMemoryStatus`. The weak interest stayed `ACTIVE` while dynamics became `fading`.

### Working memory

`WorkingMemoryConfig.minimum_goal_relevance` and `minimum_selection_score` default to `0.0`. With unused `max_items` (8) and unused token budget, an admitted semantic is selected because capacity remains, not because WM marked it current.

### Ingestion contract

`ObservationInput` has no first-class strength field. `semantic_facts` may carry `confidence`, `valid_from`, `valid_until`. Episode `importance` defaults from encoding (0.40 for both interests in this fixture). Consolidation `confidence` reflects recurrence (0.44 vs 0.66), not browse-versus-purchase. `Memory.learn()` / `record_access()` were not called; `list_learning_state` counts are 0.

## Comparison matrix

| Case | What Core did |
|------|----------------|
| Weak old + structured path | Stored, `ACTIVE`, query-relevant, **current-admitted**. Activation -3.91 &lt; -3.0. Gates include structured_fit **and** combined/lexical/associative. |
| Strong / extra support | Same `ACTIVE`/`many`. Higher support, confidence, activation. **Not** current-admitted because activation ≥ threshold; retrieved via `entity_slot_admission`. |
| Weak old, no graph | `structured_association_fit=0`. Still current-admitted via combined + lexical_evidence + associative_fit. Relationship is **sufficient, not necessary**. |
| Unrelated payment query | Still **returned** as current-admitted. Structured fit 0. Evidence feature `customer` (overlap 1) plus combined 0.25. Domain-neutral token overlap is enough. |
| Explicit `valid_until` | Weak interest omitted from `list_semantic_memories(..., valid_at=_T)` and not `RETURNED`. Status can remain `ACTIVE` with a closed validity window (not the same as `SUPERSEDED`). |
| Cardinality-one successor | Later `jacket_size=m` leaves `jacket_size=l` as `SUPERSEDED`. Inspection disposition `FILTERED_SEMANTIC_STATUS`. |
| No invalidation | Both many-valued interests remain `ACTIVE` until contradicted. |
| WM selected vs merely admitted | Weak interest selected with `selected_count=2` &lt; `max_items=8` and min scores 0.0. |
| Durable `database=postgres` vs interest | Both `ACTIVE`. No predicate-specific decay or “interest is stale” rule. |

## Admission bypass inventory

| Bypass | Why it exists | Tests that protect it |
|--------|----------------|------------------------|
| `retrieval_threshold` bypass for current admission | Current facts must not vanish solely because \(B_i\) is old | `test_ninety_day_current_size_admitted_without_floor`; `test_old_hiking_interest_current_admitted_with_hiking_in_query` |
| `semantic_soft_admission_floor` not required | 0.15.3: keep floor for other lexical/soft paths; current admission is a separate authority path | 0.15.3 suite in `tests/test_semantic_state_associative_recall.py` |
| Structured / associative / lexical gates | Query-relevant current facts may lack strong predicate/object overlap (graph hop, evidence tokens) | 0.15.7–0.15.8 association/relationship tests; this fixture’s with-graph and no-graph cases |
| Forgetting ≠ semantic status | Retention is dynamics; authority stays until superseded or validity ends | `test_forgetting_does_not_change_semantic_status` (this file); [`docs/forgetting.md`](../forgetting.md) |
| WM min scores 0.0 | Selector is a capacity/budget cut, not a second currentness policy | `WorkingMemoryConfig` defaults; `test_working_memory_selects_admitted_weak_interest_under_capacity` |

Contrast: `test_hiking_query_does_not_admit_skiing` shows skiing is **not** admitted when the cue does not make it relevant. Demo’s jacket query plus `ski-jacket → jacket` is the relevance story, not a silent “always admit old interests” rule.

## Missing fields (not invented)

`StoredSemanticMemory` has no `rehearsal_count` or `reinforcement_count`. `ObservationInput` has no `strength`. Brief vs purchase is not a Core type. Reinforcement is `record_access()` / `learn()`, not inferred from `event_type`.

Gate reconstruction used existing diagnostics; adding `admission_gate_names` would only duplicate what tests already derive.

## Demo mapping (read-only)

Sibling repo `cogkura-demo` (0.3.12) is **not** in this package. It was inspected; **not edited**.

- Hiking durable fact: purchase `evt-013` `activity_interest=hiking`, `cardinality=many`, `polarity=affirm`. Later hiking purchases often omit `semantic_facts`.
- Skiing “stale” gold: six January 2026 **browse** events; only the first carries `activity_interest=skiing` with the **same fact shape** (no `valid_until`, no `confidence`).
- `event_to_observation()` copies `event.type` and `semantic_facts`; `product_id` becomes `metadata["entity_ids"]`. It does not map browse vs purchase onto importance, confidence, or validity.
- Taxonomy: `glacier-ski-jacket` → `ski-jacket` → `jacket`, so a jacket/outerwear query can seed a structured path to the ski product entity.
- Compare on 0.3.12 still **selects** skiing while labelling it stale (`relationship_paths_used=3`; CogKura labelled coverage **5/5** with taxonomy, **3/5** without). Stale is an application gold label, not a Core status.

If Demo wants skiing out of bounded context, Core never received an expiry or a weaker fact type. Mapping a brief browse as a live `cardinality=many` interest is sufficient for 0.15.10 to treat it as current when relevant.

## Later regression guards (external; not Core CI)

Do not run Bench/Demo as a required step of this investigation. If someone later patches admission or WM filler behaviour, re-check externally against the last Demo 0.3.12 Compare snapshot: CogKura **5/5** / 6 units / ~135 tokens with taxonomy (skiing still selected); **3/5** without taxonomy. Core docs still quote older ~4/5 and ~3/5 until fixtures supplied catalog relationships (`docs/declarative-activation.md` 0.15.8). A patch that drops current-admitted many-valued interests will also hit `test_ninety_day_current_size_admitted_without_floor`.

## Decision

| Option | Meaning | Verdict |
|--------|---------|---------|
| **A** | No Core change on 0.15.10 | **Recommended now.** Behaviour matches the 0.15.3 authority/accessibility/relevance split. |
| **B** | Narrow later patch (e.g. 0.15.11) to drop weak admitted semantics | **Reject.** No safe local threshold or WM filler change without regressing long-horizon current facts, unless a replacement admission policy ships in the same change. |
| **C** | New lifecycle for gradual currentness of `cardinality=many` | **Only if product wants Core-owned decay** without `valid_until`. That is a design note, not a constant tweak. Owner: Core. Next version: none until that note exists. |
| **D** | Demo / application ingestion | **Recommended if the symptom is labelled staleness.** Stop emitting a durable current interest, or set `valid_until` / polarity, or accept that Core will treat the fact as live. Owner: Demo. Do not teach Core “skiing is stale”. |

**Recommended owner:** Demo for the outdoor-retail gold mismatch (**D**). Core stays on **A** unless a written lifecycle requirement appears (**C**). **No 0.15.11 version** from this investigation.
