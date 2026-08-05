# Semantic consolidation design reference

**Target release:** Cognema `0.3`  
**Status:** Shipped  
**Research basis:** Complementary Learning Systems (CLS)  
**Primary implementation:** `src/cognema/algorithms/semantic.py`

## Summary

Cognema `0.2` introduced deterministic episodic memory encoding. Version `0.3` adds **semantic consolidation**: deriving stable, reusable knowledge from multiple episodic memories.

The implementation separates:

- **Extraction** — probabilistic or metadata-driven fact candidates from episodes (`SemanticExtractor`)
- **Consolidation** — deterministic promotion, contradiction handling, and scoring (`SemanticConsolidator`)

`recall()` remains observation-only in `0.3`.

## Public API

| Method | Purpose |
|--------|---------|
| `consolidate_semantics(tenant_id=..., subject_id=None)` | Extract facts, consolidate, upsert semantic memories |
| `list_semantic_memories(tenant_id=..., ...)` | List tenant-scoped semantic memories |

`Memory.clear()` order: semantic → episodic → observations.

## Metadata extraction

Observations may include structured facts under `metadata["semantic_facts"]`:

```json
{
  "predicate": "preferred_database",
  "object_value": "postgresql",
  "object_entity_id": "postgresql",
  "cardinality": "one",
  "polarity": "affirm",
  "qualifiers": {"environment": "production"},
  "confidence": 0.95
}
```

`MetadataSemanticExtractor` skips malformed entries and returns `SemanticExtractionResult(candidates, failed)`. Failures surface as `extracted_failures` on `SemanticConsolidationResult`.

## Consolidation rules

`ComplementaryLearningSemanticConsolidator` (default):

- Canonicalises predicates, objects, qualifiers (NFKC, casefold, stable JSON)
- Groups candidates by slot key and claim key
- Promotes when `minimum_supporting_episodes=2` (configurable)
- `cardinality=one`: competing objects contradict
- `cardinality=many`: distinct objects coexist
- Opposing polarity counts as contradiction
- Status `contested` when contradiction ratio exceeds threshold
- Deterministic `statement` projection and `content_fingerprint` for idempotent upserts

## Storage

Semantic memories reuse `cognema.memories` with `memory_type='semantic'`.

Additional tables (migration `003_semantic_consolidation.sql`):

- `semantic_claims` — structured claim fields and support metadata
- `memory_derivations` — episode → semantic links (`supports` / `contradicts`)

`memory_evidence` and `memory_entities` continue to store flattened observation provenance and entity roles.

Postgres apps must pass `PostgresSemanticMemoryStore` alongside observation and episode stores.

## Out of scope (0.3)

- LLM or embedding extractors
- Fuzzy synonym merging
- Associative world model / spreading activation
- Including semantic memories in `recall()`
- Forgetting curves or auto-consolidate on observe
- Automatic `SUPERSEDED` lifecycle transitions

See [`docs/roadmap.md`](roadmap.md) for `0.4` cognitive retrieval.
