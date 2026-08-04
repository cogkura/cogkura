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

## 0.2 - Episodic memory

- richer event encoding from observations;
- temporal context handling;
- source and entity relationship links;
- importance and salience scoring;
- episodic clustering foundations.

## 0.3 - Semantic consolidation

- episodic clustering from observations;
- concept extraction;
- semantic memory construction;
- evidence links from memories to observations;
- contradiction and duplicate handling.

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
