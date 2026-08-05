# Roadmap

## 0.0.x - Foundation (done)

- package structure and typed API;
- memory event models;
- in-memory storage backend;
- deterministic recall baseline;
- tests and documentation;
- PyPI publishing workflow.

## 0.1 - PostgreSQL observation ingestion

- `ObservationInput` and observation storage protocols;
- PostgreSQL schema, migrations, observation store, checkpoint store;
- `PostgresTableSource` with compound cursor pagination;
- `Memory.observe()`, `Memory.ingest()`, and tenant-scoped `Memory.recall()`;
- revision history for create, update, delete, restore;
- Docker PostgreSQL example with seed and mutation scripts;
- unit tests and optional PostgreSQL integration tests.

## 0.2 - Episodic memory (done)

- deterministic episodic encoding from observations;
- temporal context and conversation/thread grouping;
- source observation evidence and entity links;
- attention-based salience scoring;
- in-memory and PostgreSQL episode stores;
- `Memory.encode_episodes()` and `Memory.list_episodes()`.

## 0.3 - Semantic consolidation (done)

- metadata-driven semantic fact extraction (`semantic_facts` on observations);
- `ComplementaryLearningSemanticConsolidator` with recurrence, contradiction, and promotion rules;
- `SemanticMemoryStore` with in-memory and PostgreSQL backends;
- `Memory.consolidate_semantics()` and `Memory.list_semantic_memories()`;
- migration `003_semantic_consolidation.sql` (`semantic_claims`, `memory_derivations`).

## 0.4 - Cognitive retrieval

- spreading activation and associative scoring;
- forgetting curves and attention filtering;
- goal-aware working-memory selection;
- replace transitional token-overlap recall.

## Later

- additional source connectors (SQLite, APIs, queues);
- graph-oriented storage options;
- embedding-provider interfaces;
- LLM-provider interfaces;
- MCP and agent-framework integrations;
- benchmark suites;
- enterprise-oriented extensions.
