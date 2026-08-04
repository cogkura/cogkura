# Architecture

## Design direction

Cognema is a thin cognitive layer between application data and AI reasoning.

Applications keep their own persistence and model infrastructure. Customer data stays in customer-owned schemas. Cognema owns observations, revisions, checkpoints, and (later) derived memories.

The public API should remain stable even as internals evolve.

## Layers

### Public API

- `Memory`: facade for observation ingestion and recall
- `observe(ObservationInput)` / `ingest(...)`: write path
- `recall(query, tenant_id=...)`: tenant-scoped read path
- `ObservationInput`, `StoredObservation`, `IngestionResult`, `IngestStatus`
- `RecallResult` (scored `StoredObservation` matches)

These models are typed and validated so behavior stays explicit.

### Observation pipeline

1. Source connector reads customer records (read-only)
2. Application mapper converts records to `ObservationInput`
3. Policy evaluates attention and acceptance
4. Retention mode transforms content before storage
5. Observation store persists with revision history
6. Checkpoint store advances only after successful batches

### Storage protocols

[`src/cognema/storage/base.py`](../src/cognema/storage/base.py):

- `ObservationStore`: normalized observations + revisions (`ingest`, `get_by_source`, `list`, `clear`)
- `InMemoryObservationStore`: default backend for local use and tests
- `CheckpointStore`: per-tenant connector checkpoints

PostgreSQL implementations live in [`src/cognema/storage/postgres.py`](../src/cognema/storage/postgres.py) behind the optional `cognema[postgres]` extra.

Custom stores can satisfy the same protocols without requiring PostgreSQL.

### Source connectors

[`SourceConnector`](../src/cognema/sources/base.py) protocol with `PostgresTableSource` as the first implementation.

Connectors use compound `(updated_at, id)` cursors. Hard deletes are not detected; soft-delete columns are the supported path. When `soft_delete_column` is configured with an explicit column list, that column is always included in the SELECT so mappers can set `is_deleted`.

### Cognitive algorithms

[`src/cognema/algorithms/`](../src/cognema/algorithms/) remains a stub. Future modules will implement consolidation, decay, attention, and association mechanics.

### Retrieval

`recall()` uses deterministic token overlap over stored observations and always requires `tenant_id`. That placeholder is dependency-free and transparent so cognitive retrieval can replace it later.

### Embeddings and LLM integrations

Embedding providers and LLM providers are planned integration points, not implemented features in `0.1.0`.

Cognema should orchestrate memory behavior without forcing specific providers.

## Deployment models

1. Same database, separate `cognema` schema
2. Separate source and memory databases (canonical Docker example)
3. Custom storage via `ObservationStore` / `CheckpointStore` protocols
4. In-memory observation store (default `Memory()` for local use)

## Package layout

```text
src/cognema/
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

- public observation API (`observe`, `ingest`, `recall`);
- observation pipeline, policies, and retention modes;
- storage protocols with in-memory and PostgreSQL backends;
- `PostgresTableSource` and connector checkpoints;
- tenant-scoped token-overlap recall;
- Docker example, tests, and documentation.

Planned later:

- episodic and semantic memory subsystems;
- consolidation pipelines;
- spreading activation and associative recall;
- goal-aware filtering and working-memory selection;
- additional source connectors and provider interfaces.
