# Architecture

## Design direction

Cogkura is a thin cognitive layer between application data and AI reasoning.

Applications keep their own persistence and model infrastructure. Customer data stays in customer-owned schemas. Cogkura owns observations, revisions, checkpoints, and (later) derived memories.

The public API should remain stable even as internals evolve.

## Layers

### Public API

- `Memory`: facade for observation ingestion, processing, context preparation, and recall
- `observe(ObservationInput)` / `ingest(...)`: write path
- `process(...)`: explicit observation-to-memory formation
- `prepare_context(...)`: application-facing read path returning `MemoryContext`
- `record_context_use(context)`: record consumption of prepared context
- `recall(query, tenant_id=...)`: tenant-scoped low-level read path
- `ObservationInput`, `StoredObservation`, `IngestionResult`, `IngestStatus`
- `MemoryContext`, `MemoryProcessingResult`, `RecallResult`

These models are typed and validated so behavior stays explicit.

### Observation pipeline

1. Source connector reads customer records (read-only)
2. Application mapper converts records to `ObservationInput`
3. Policy evaluates attention and acceptance
4. Retention mode transforms content before storage
5. Observation store persists with revision history
6. Checkpoint store advances only after successful batches

### Storage protocols

[`src/cogkura/storage/base.py`](../src/cogkura/storage/base.py):

- `ObservationStore`: normalized observations + revisions (`ingest`, `get_by_source`, `list`, `clear`)
- `InMemoryObservationStore`: default backend for local use and tests
- `CheckpointStore`: per-tenant connector checkpoints

PostgreSQL implementations live in [`src/cogkura/storage/postgres.py`](../src/cogkura/storage/postgres.py) behind the optional `cogkura[postgres]` extra.

Custom stores can satisfy the same protocols without requiring PostgreSQL.

### Source connectors

[`SourceConnector`](../src/cogkura/sources/base.py) protocol with `PostgresTableSource` as the first implementation.

Connectors use compound `(updated_at, id)` cursors. Hard deletes are not detected; soft-delete columns are the supported path. When `soft_delete_column` is configured with an explicit column list, that column is always included in the SELECT so mappers can set `is_deleted`.

### Cognitive algorithms

[`src/cogkura/algorithms/`](../src/cogkura/algorithms/):

- `episodic.py` — deterministic episode encoding
- `semantic.py` — semantic consolidation
- `activation.py` — ACT-R declarative activation (base-level, partial matching, ranking)
- `spreading.py` — bounded entity–memory spreading activation
- `forgetting.py` — Ebbinghaus-inspired retention lifecycle from base-level only

### Retrieval

`recall()` ranks episodic and semantic memories with ACT-R declarative activation (base-level, spreading, partial matching). `valid_at` filters semantic revision windows and episodes (`started_at <= valid_at`). `encode_episodes()` and `consolidate_semantics()` accept optional `as_of` for simulated replay timestamps. `FORGOTTEN` memories are excluded by default (`include_forgotten=True` to opt in). Structured `RetrievalCue.entity_ids` enable associative retrieval. `record_access()` reinforces recalled memories and reactivates forgotten dynamics. `apply_forgetting()` evaluates lifecycle state and compacts old activation references.

### Embeddings and LLM integrations

Embedding providers and LLM providers are planned integration points, not implemented features in `0.1.0`.

Cogkura should orchestrate memory behavior without forcing specific providers.

## Deployment models

1. Same database, separate `cogkura` schema
2. Separate source and memory databases (canonical Docker example)
3. Custom storage via `ObservationStore` / `CheckpointStore` protocols
4. In-memory observation store (default `Memory()` for local use)

## Package layout

```text
src/cogkura/
  memory.py
  observations/
  sources/
  mappers/
  storage/
  migrations/postgres/
  algorithms/
```

## Current implementation boundary

Implemented now:

- public observation API (`observe`, `ingest`);
- application integration API (`process`, `prepare_context`, `record_context_use`, `MemoryContext`);
- declarative activation API (`recall`, `record_access`, `apply_forgetting`);
- episodic encoding API (`encode_episodes`, `list_episodes`);
- semantic consolidation API (`consolidate_semantics`, `list_semantic_memories`);
- observation pipeline, policies, and retention modes;
- storage protocols with in-memory and PostgreSQL backends;
- `DeterministicEpisodicEncoder`, `EpisodeStore`, `MetadataSemanticExtractor`, `SemanticMemoryStore`, `ACTRDeclarativeActivator`, `DeterministicSpreadingActivator`, `EbbinghausForgettingEvaluator`, `ActivationStore`, `MemoryDynamicsStore`;
- `PostgresTableSource` and connector checkpoints;
- ACT-R declarative recall with spreading activation and forgetting dynamics over episodic + semantic memories;
- Docker example, tests, and documentation.

Planned later:

- additional source connectors and provider interfaces.
