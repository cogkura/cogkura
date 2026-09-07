# Configuration reference (0.16.2)

Cogkura exposes two primary configuration dataclasses on `Memory(...)`: `ActivationConfig` and `WorkingMemoryConfig`. Defaults are defined in [`src/cogkura/models.py`](../src/cogkura/models.py).

## Retrieval stages

Understanding configuration requires separating stages:

```text
candidate → returned → admitted → recalled → chunked → selected → rendered
```

- **candidate**: all episodic and semantic memories considered for a query.
- **returned**: rows that pass recall inspection dispositions for the query.
- **admitted**: soft paths (`SEMANTIC_CURRENT_ADMISSION`, slot admission) that bypass `retrieval_threshold` without boosting rank.
- **recalled**: ranked `RecallResult` list from `Memory.recall()`.
- **chunked**: `WorkingMemoryChunk` groups formed after recall (when `enable_chunking=True`).
- **selected**: chunks that fit `max_items` and token budget.
- **rendered**: `MemoryContext.render()` text shown to downstream reasoning.

`inspect_recall()` explains admission; `prepare_context()` explains selection. They are separate calls.

## Stable public knobs

| Knob | Default | Role |
|------|---------|------|
| `retrieval_threshold` | `-3.0` | ACT-R floor for declarative recall |
| `context_reinstatement_weight` | `0.50` | Bounded episodic boost from retrieval-context match (`0` disables contribution) |
| `semantic_context_reinstatement_weight` | `0.25` | Bounded semantic boost from SUPPORT episode encoding contexts (`0` disables contribution) |
| `max_items` | `8` | Maximum **chunks** when chunking is enabled |
| `max_prompt_tokens` | `2048` | Working-memory token budget |
| `enable_chunking` | `true` | Chunk-based selection vs item-level |
| `enable_structured_relationships` | `true` | Traverse `metadata["relationships"]` at recall |

Application integration:

- `prepare_context(..., prompt_budget_tokens=...)` overrides WM token budget per call.
- `recall(..., valid_at=..., as_of=...)` sets validity time vs cognitive evaluation time.

## Advanced public knobs

### Activation (`ActivationConfig`)

| Knob | Default | Role |
|------|---------|------|
| `context_reinstatement_weight` | `0.50` | Episodic activation boost from context match (`R=M×V`, `C=λ×R`); `0` keeps diagnostics only |
| `semantic_context_reinstatement_weight` | `0.25` | Semantic activation boost from SUPPORT episode contexts (`Rₛ=ΣRⱼ/K`, `Cₛ=λsem×Rₛ`); `0` keeps diagnostics only |
| `semantic_soft_admission_floor` | `-4.0` | Lexical soft-admission floor (not required for current admission) |
| `max_soft_admitted_semantics` | `8` | Cap on soft-admitted semantics |
| `semantic_current_min_relevance` | `0.06` | Combined relevance gate for current admission |
| `contextual_association_min_relevance` | `0.06` | Associative bridge gate |
| `semantic_relationship_min_relevance` | `0.06` | Structured relationship gate |
| `association_seed_min_relevance` | `0.15` | Seed selection for contextual association |
| `max_association_seeds` | `5` | Association seed cap |
| `max_relationship_hops` | `2` | Relationship traversal depth |
| `relationship_default_weight` | `0.4` | Default edge weight |
| `relationship_type_weights` | map | Per-relation weights (`is_a`, `part_of`, …) |
| `enable_entity_slot_admission` | `true` | Entity-based slot soft admission |
| `force_slot_admission` | `false` | Debug override for slot admission |
| `current_state_weight` | `0.5` | Current-state activation bias |
| `minimum_goal_relevance` | n/a | WM only — see below |

See [`docs/declarative-activation.md`](declarative-activation.md) for the full activation surface.

### Working memory (`WorkingMemoryConfig`)

| Knob | Default | Role |
|------|---------|------|
| `candidate_pool_size` | `50` | Recall pool size before chunking |
| `activation_weight` | `0.45` | Selection score weight |
| `goal_relevance_weight` | `0.35` | Goal text relevance weight |
| `importance_weight` | `0.15` | Stored importance weight |
| `minimum_goal_relevance` | `0.0` | Floor on goal relevance (0.0 = no filler cut) |
| `minimum_selection_score` | `0.0` | Floor on final selection score |
| `inhibition_strength` | `0.30` | Redundancy inhibition |
| `redundancy_threshold` | `0.70` | Similarity threshold for inhibition |
| `stale_goal_penalty` | `0.35` | Penalty when goal tokens mismatch |

## Observation contract (application boundary)

Applications supply:

- `ObservationInput` with `source_namespace` + `source_record_id` (stable external IDs)
- `metadata["semantic_facts"]` for durable claims (`predicate`, `object_value`, `cardinality`, optional `valid_from` / `valid_until`, `confidence`)
- `metadata["entity_ids"]` for episodic entity links
- `metadata["relationships"]` for directed structure (`source_entity_id`, `relation_type`, `target_entity_id`)

Guidance:

- transient behaviour → episode only
- strong, repeated, or explicit evidence → semantic fact
- relationships are **supplied facts**, not inferred ontologies

Future `cogkura-ingest` should use only these public paths. Core does not parse arbitrary external schemas.

## Cardinality and temporal semantics

- `cardinality=one`: values in the same slot compete; newer implicit evidence supersedes unless explicit windows overlap (`CONTESTED`).
- `cardinality=many`: independent `ACTIVE` values until explicitly invalidated.
- `as_of`: cognitive evaluation clock (activation, forgetting).
- `valid_at`: world validity time for semantic revision windows.
- `SUPERSEDED`: authority replaced; historical rows remain retrievable with `valid_at`.

See [`docs/reconsolidation.md`](reconsolidation.md) and [`docs/findings/semantic-currentness-investigation.md`](findings/semantic-currentness-investigation.md).

## SUPPORT vs ASSOCIATION

- **SUPPORT** (`SemanticDerivationRelation.SUPPORTS`): stored provenance that contributes to semantic truth; may attach to `SEMANTIC_WITH_SUPPORT` chunks.
- **ASSOCIATION** (`AssociationPath` at recall): rank-time bridge between candidates; does not change stored semantics.

## Activation vs truth

ACT-R activation controls **accessibility and ranking**, not semantic truth. Forgetting may make a memory hard to retrieve without making a valid semantic false. `SEMANTIC_CURRENT_ADMISSION` keeps current authoritative facts reachable when relevant even if base-level is below `retrieval_threshold`.

## Chunks

Working-memory chunks are **ephemeral**. They group recalled memories, preserve member identities for inspection and `record_context_use`, consume bounded capacity, and serialize deterministic context. They are not persisted as long-term memory.

Since 0.15.11, `SEMANTIC_WITH_SUPPORT` chunks with structured predicate/object serialize the **semantic statement only** by default; support episodes remain attached for provenance. See [`docs/working-memory.md`](working-memory.md) for the frozen 0.15.12 members-vs-rendered contract.

## Performance baselines (observational)

Correctness is enforced by pytest. Local timing evidence is optional and machine-dependent:

```bash
./scripts/benchmark_release.sh
# or timings only:
uv run python scripts/run_prepare_context_benchmark.py
```

The helper writes `results/benchmark-results.{json,md}` (gitignored). A committed historical snapshot lives in [`docs/findings/0.15.12-performance-baseline.md`](findings/0.15.12-performance-baseline.md). Timings are **not** CI acceptance thresholds.
