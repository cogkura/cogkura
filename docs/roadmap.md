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

## 0.4 - Declarative activation (done)

- ACT-R base-level frequency/recency activation;
- deterministic partial matching over structured cues;
- retrieval threshold and latency metadata;
- activation reference storage (`memory_activation_references`);
- explicit `record_access()` reinforcement;
- `Memory.recall()` over episodic + semantic memories (hard cut from observation token overlap).

## 0.5 - Spreading activation (done)

- fan-sensitive contextual spreading activation over entity–memory associations;
- bounded multi-hop propagation with distance decay;
- enable spreading in `ActivationConfig` by default;
- transient associative graph built from recall candidates (no new storage).

## 0.6 - Forgetting / memory dynamics (done)

- Ebbinghaus-inspired `ACTIVE → FADING → FORGOTTEN` lifecycle from ACT-R base-level only;
- `MemoryDynamicsStore`, `apply_forgetting()`, and `include_forgotten` on recall;
- reversible reactivation via `record_access()`;
- weighted activation-reference compaction;
- migration `005_forgetting_dynamics.sql`.

## 0.7 - Working memory / inhibition (done)

- deterministic bounded selection from `RecallResult` candidates;
- goal-aware ranking;
- item and prompt budgets;
- competitive redundancy inhibition;
- transient working-memory decay and carry-over;
- immutable `WorkingMemorySnapshot`;
- no persistent working-memory store.

## 0.9 - Learning and reinforcement (done)

- Outcome feedback (`HELPFUL`, `UNHELPFUL`, `INCORRECT`) via `Memory.learn()`.
- Contextual utility for working memory; HELPFUL ACT-R traces; learned associations in spreading.
- Migration `007_learning_reinforcement.sql`.

## 0.8 - Reconsolidation / memory updating (done)

- revision-aware semantic consolidation (`SemanticRevisionCandidate`);
- deterministic temporal reconciliation (`DeterministicSemanticReconciler`);
- `REINFORCES` / `COEXISTS` / `SUPERSEDES` / `CONFLICTS` relation matrix;
- `Memory.list_semantic_revisions()` and `valid_at` historical retrieval;
- migration `006_semantic_reconsolidation.sql`;
- [`docs/reconsolidation.md`](reconsolidation.md).

## Later

- `0.9`: learning / reinforcement (done);
- `0.10`: metamemory (planned);
- `0.10`: additional cognitive maintenance (planned);
- additional source connectors (SQLite, APIs, queues);
- graph-oriented storage options;
- embedding-provider interfaces;
- LLM-provider interfaces;
- MCP and agent-framework integrations;
- benchmark suites;
- enterprise-oriented extensions.
