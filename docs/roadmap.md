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

## 0.14.1 - Retrieval corrections and ranking separation (done)

- Lexical current-state intent independent of structured cue fields.
- Historical slot admission respects `valid_at` rather than present ACTIVE status.
- Query coverage for accessibility; precision-aware F1 for eligible ranking only.
- Current-state bonuses scoped to matched semantic slots.
- [`docs/design-retrieval-corrections-0.14.1.md`](design-retrieval-corrections-0.14.1.md).

## 0.14 - Retrieval eligibility, global ranking, and temporal relevance (done)

- Global ranking over eligible candidates; admission is threshold bypass only.
- Soft entity slot admission (`enable_entity_slot_admission`).
- Current-state policy decoupled from admission; lifecycle bias only under policy.
- Precision-aware text matching (`enable_text_precision_matching`) for recall and WM relevance.
- [`docs/design-retrieval-eligibility-ranking.md`](design-retrieval-eligibility-ranking.md).

## 0.13 - Gated slot admission and association (done)

- Gate slot admission to current-state lexicon, `predicate`, or `force_slot_admission`.
- Multi-entity conjunction bonus; incident IDF scale and tag seeding.
- Hard-exclude SUPERSEDED-only SUPPORT on current-state recall.
- Metamemory `MISSING_KNOWLEDGE`; WM same-slot collapse and stale-goal penalty.
- Rank-time same-slot SUPPORT collapse; `__version__` fix.
- [`docs/design-gated-slot-admission.md`](design-gated-slot-admission.md).

## 0.12 - String cues and access recording (done)

- String-cue entity seeding for spreading (`enable_text_entity_seeding`).
- Semantic slot admission before threshold cut.
- Superseded-slot current-state penalties on supporting episodes.
- Numeric-token duplicate collapse; `record_access(..., min_score=...)`.
- [`docs/design-string-cues-current-state.md`](design-string-cues-current-state.md).

## Later

- `0.9`: learning / reinforcement (done);
- `0.10`: metamemory / memory monitoring (done);
- `0.11`: ranking, simulated time, and current-state recall (done);
- `0.12+`: additional cognitive maintenance (planned);
- additional source connectors (SQLite, APIs, queues);
- graph-oriented storage options;
- embedding-provider interfaces;
- LLM-provider interfaces;
- MCP and agent-framework integrations;
- benchmark suites;
- enterprise-oriented extensions.
