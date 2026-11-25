# Roadmap

## 0.0.x - Foundation (done)

- package structure and typed API;
- memory event models;
- in-memory storage backend;
- deterministic recall baseline;
- tests and documentation;
- PyPI publishing workflow.

## 0.1 - PostgreSQL observation ingestion (done)

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

## 0.8 - Reconsolidation / memory updating (done)

- revision-aware semantic consolidation (`SemanticRevisionCandidate`);
- deterministic temporal reconciliation (`DeterministicSemanticReconciler`);
- `REINFORCES` / `COEXISTS` / `SUPERSEDES` / `CONFLICTS` relation matrix;
- `Memory.list_semantic_revisions()` and `valid_at` historical retrieval;
- migration `006_semantic_reconsolidation.sql`;
- [`docs/reconsolidation.md`](reconsolidation.md).

## 0.9 - Learning and reinforcement (done)

- Outcome feedback (`HELPFUL`, `UNHELPFUL`, `INCORRECT`) via `Memory.learn()`.
- Contextual utility for working memory; HELPFUL ACT-R traces; learned associations in spreading.
- Migration `007_learning_reinforcement.sql`.

## 0.10 - Metamemory / memory monitoring (done)

- Deterministic read-only metamemory via `Memory.assess_memory()`.
- Cue coverage, retrieval strength, evidence confidence, conflict, and forgetting-pressure signals.
- [`docs/metamemory.md`](metamemory.md).

## 0.11 - Ranking, simulated time, and current-state recall (done)

- Simulated time on encode/consolidate; episode visibility at `valid_at`.
- Candidate-set IDF, near-duplicate collapse, current-state semantic bias.
- Importance-aware forgetting and semantic-support protection.
- [`docs/design-ranking-time-current-state.md`](design-ranking-time-current-state.md).

## 0.12 - String cues and access recording (done)

- String-cue entity seeding for spreading (`enable_text_entity_seeding`).
- Semantic slot admission before threshold cut.
- Superseded-slot current-state penalties on supporting episodes.
- Numeric-token duplicate collapse; `record_access(..., min_score=...)`.
- [`docs/design-string-cues-current-state.md`](design-string-cues-current-state.md).

## 0.13 - Gated slot admission and association (done)

- Gate slot admission to current-state lexicon, `predicate`, or `force_slot_admission`.
- Multi-entity conjunction bonus; incident IDF scale and tag seeding.
- Hard-exclude SUPERSEDED-only SUPPORT on current-state recall.
- Metamemory `MISSING_KNOWLEDGE`; WM same-slot collapse and stale-goal penalty.
- Rank-time same-slot SUPPORT collapse; `__version__` fix.
- [`docs/design-gated-slot-admission.md`](design-gated-slot-admission.md).

## 0.14 - Retrieval eligibility, global ranking, and temporal relevance (done)

- Global ranking over eligible candidates; admission is threshold bypass only.
- Soft entity slot admission (`enable_entity_slot_admission`).
- Current-state policy decoupled from admission; lifecycle bias only under policy.
- Precision-aware text matching (`enable_text_precision_matching`) for recall and WM relevance.
- [`docs/design-retrieval-eligibility-ranking.md`](design-retrieval-eligibility-ranking.md).

## 0.14.1 - Retrieval corrections and ranking separation (done)

- Lexical current-state intent independent of structured cue fields.
- Historical slot admission respects `valid_at` rather than present ACTIVE status.
- Query coverage for accessibility; precision-aware F1 for eligible ranking only.
- Current-state bonuses scoped to matched semantic slots.
- [`docs/design-retrieval-corrections-0.14.1.md`](design-retrieval-corrections-0.14.1.md).

## 0.14.2 - Temporal mode, structured slot fit, and metamemory answerability (done)

- Internal `NEUTRAL` / `CURRENT` / `HISTORICAL` retrieval modes; historical mode can activate matching soft admission.
- Structured semantic-slot fit refines eligible ranking only; SUPPORT episodes inherit fit from the slot they support.
- Metamemory answerability distinguishes unresolved knowledge from weak retrieval.
- [`docs/design-temporal-slot-answerability-0.14.2.md`](design-temporal-slot-answerability-0.14.2.md).

## 0.14.3 - Conjunctive slot matching and positive structured ranking (done)

- Explicit entity, predicate, and object constraints are conjunctive in shared semantic matching.
- Perfect structured slot fit provides positive bounded ranking evidence; mismatch is penalised relative to neutral.
- [`docs/design-conjunctive-slot-ranking-0.14.3.md`](design-conjunctive-slot-ranking-0.14.3.md).

## 0.14.4 - Retrieval diagnostics and SUPPORT provenance (done)

- `RecallResult.diagnostics` exposes accessibility vs rank activation, text coverage, cue fit, temporal mode, structured fit, and eligibility diagnostics.
- SUPPORT ranking diagnostics preserve derivation-backed semantic revision provenance and selected inherited-fit source.
- [`docs/design-retrieval-diagnostics-support-provenance-0.14.4.md`](design-retrieval-diagnostics-support-provenance-0.14.4.md).

## 0.15.0 - Application integration and memory context (done)

- `Memory.process()` orchestrates episodic encoding and semantic consolidation with one evaluation timestamp;
- `Memory.prepare_context()` returns bounded working memory and metamemory assessment in one read;
- `MemoryContext` provider-neutral structured boundary with deterministic `render()`;
- `Memory.record_context_use()` records use of selected context memories;
- shared declarative retrieval inside `prepare_context()` (one rank pass per call);
- [`docs/application-integration.md`](application-integration.md) and [`examples/application_context.py`](../examples/application_context.py).
- [`docs/design-application-integration-memory-context-0.15.0.md`](design-application-integration-memory-context-0.15.0.md).

## 0.15.1–0.15.12 — Dependable recall line (done)

- **0.15.1** evidence-chronology traces; `inspect_recall()`
- **0.15.2** lexical slot matching; bounded soft admission
- **0.15.3** cardinality-one evidence reconciliation; `SEMANTIC_CURRENT_ADMISSION`
- **0.15.4** `valid_at` validity clock; current snapshot mode
- **0.15.5** canonical retrieval features; evidence-to-evidence association
- **0.15.6** cardinality-aware recall identity; relevance-tier competition
- **0.15.7** contextual association (seed/bridge hops)
- **0.15.8** structured entity relationships
- **0.15.9** working-memory chunking; coverage-aware selection
- **0.15.10** `SEMANTIC_WITH_SUPPORT` structural primary correctness
- **0.15.11** temporal contract verification; semantic-only support render; performance baselines; configuration/architecture docs freeze
- **0.15.12** architecture-freeze hardening: frozen support-render contract, observational performance baseline snapshot, mandatory canonical regression fixture
- **0.16.0** encoding-context capture on observations and episodes (retrieval-neutral)
- **0.16.1** retrieval-context matching diagnostics (retrieval-neutral)
- **0.16.2** context reinstatement on episodic activation (accessibility, not relevance)
- **0.16.3** semantic support-context propagation on declarative activation (accessibility, not relevance)
- **0.16.4** context observability and contextual metamemory (inspect explains 0.16.0–0.16.3 evidence; retrieval unchanged)
- **0.16.5** context hardening and architecture freeze (`CONTEXT_CONFLICT`, semantic support diagnostic semantics, `concept_ids` contract, consolidated invariants)

See [`docs/configuration.md`](configuration.md) and [`docs/architecture.md`](architecture.md).

## 0.16.0 — Encoding context (done)

- Public `ObservationContext` on `ObservationInput`; persisted on observations and revisions.
- Public `MemoryContextSignature` on episodic memories (`encoding_context`).
- Deterministic aggregation during episodic encoding; Postgres migration `009_encoding_context.sql`.
- Retrieval-neutral in `0.16.0` (no ranking, activation, admission, or working-memory changes).
- [`docs/design-encoding-context-0.16.0.md`](design-encoding-context-0.16.0.md) and [`examples/encoding_context.py`](../examples/encoding_context.py).

## 0.16.1 — Retrieval context matching (done)

- Public `RetrievalContext` on recall APIs and optional `RetrievalCue.retrieval_context`.
- `DeterministicContextMatcher` compares retrieval cues to episodic `encoding_context`.
- Ephemeral `ContextMatch` on `RetrievalDiagnostics`; no storage migration.
- Retrieval-neutral in `0.16.1` (matching calculated after ranking, not used for activation or selection).
- [`docs/design-retrieval-context-0.16.1.md`](design-retrieval-context-0.16.1.md) and [`examples/retrieval_context.py`](../examples/retrieval_context.py).

## 0.16.2 — Context reinstatement (done)

- `DeterministicContextReinstatementPolicy` converts `ContextMatch` into bounded episodic activation contribution (`Rᵢ = Mᵢ × Vᵢ`, `Cᵢ = λctx × Rᵢ`, `Aᵢ' = Aᵢ + Cᵢ`).
- `ActivationConfig.context_reinstatement_weight` (default `0.50`; `0` disables while preserving diagnostics).
- Episodic only; semantic candidates unchanged; positive-only (mismatch/missing → `0`).
- No-context and `weight=0` reproduce `0.16.1` retrieval behaviour.
- [`docs/design-context-reinstatement-0.16.2.md`](design-context-reinstatement-0.16.2.md) and calibration [`docs/findings/0.16.2-context-reinstatement-calibration.md`](findings/0.16.2-context-reinstatement-calibration.md).

## 0.16.3 — Semantic support-context propagation (done)

- `DeterministicSemanticSupportContextPolicy` aggregates unique `SUPPORTS` episode `ContextMatch` evidence into bounded semantic activation (`Rⱼ = Mⱼ × Vⱼ`, `Rₛ = ΣRⱼ/K`, `Cₛ = λsem × Rₛ`, `Aₛ' = Aₛ + Cₛ`).
- `ActivationConfig.semantic_context_reinstatement_weight` (default `0.25`; `0` disables while preserving diagnostics).
- Episodic `0.16.2` path unchanged; semantic candidates keep `context_match=None`; positive-only (mismatch/unavailable → `0`).
- No-context and `semantic_context_reinstatement_weight=0` reproduce `0.16.2` semantic retrieval behaviour.
- [`docs/design-semantic-support-context-0.16.3.md`](design-semantic-support-context-0.16.3.md) and calibration [`docs/findings/0.16.3-semantic-context-calibration.md`](findings/0.16.3-semantic-context-calibration.md).

## 0.16.4 — Context observability and metamemory (done)

- `inspect_recall` exposes retrieval-level `RetrievalContextDiagnostics` plus per-candidate threshold-crossing and pre/post context ranks.
- `MemoryAssessment.context` is additive; existing flags/signals unchanged. Canonical underspecification is inspect, not assess.
- No-context and populated-context retrieval behaviour matches `0.16.3` for identical inputs.
- [`docs/design-context-observability-0.16.4.md`](design-context-observability-0.16.4.md) and calibration [`docs/findings/0.16.4-context-observability.md`](findings/0.16.4-context-observability.md).

## 0.16.5 — Context hardening and architecture freeze (done)

- `CONTEXT_CONFLICT` metamemory when comparable evidence has zero positive matches; `NO_CONTEXTUAL_MATCH` reason code.
- Semantic support diagnostics distinguish `NO_RETRIEVAL_CONTEXT`, `NO_COMPARABLE_CONTEXT`, and `NOT_EVALUATED`; no-context no longer marks supports unavailable.
- Mixed semantic support conflict observable via `PARTIAL_CONTEXT_CONFLICT` without changing `Rₛ`.
- `MemoryContextSignature.concept_ids` reserved (`()` on default encoder); weight documentation reconciled with defaults `0.50` / `0.25`.
- Architecture-freeze suite [`tests/test_contextual_memory_architecture_freeze.py`](../tests/test_contextual_memory_architecture_freeze.py).
- [`docs/design-context-hardening-0.16.5.md`](design-context-hardening-0.16.5.md) and findings [`docs/findings/0.16.5-context-hardening.md`](findings/0.16.5-context-hardening.md).

## 0.17.0 — Competition representation and diagnostics (done)

- Observational cue-competition diagnostics on `inspect_recall` (`CompetitionDiagnostics`, `CompetitionEvidence`, `CompetitionDirection`).
- Deterministic `CompetitionMatcher` over inspect discrimination-set candidates; bounded top-k competitors per candidate.
- `CompetitionConfig` on `Memory(...)`; no storage migration; `rank()` / `recall()` behaviour unchanged.
- [`docs/design-competition-representation-0.17.0.md`](design-competition-representation-0.17.0.md) and [`examples/competition.py`](../examples/competition.py).

## 0.17.1 — Competition hardening and transient interference (done)

- Hardened matcher: query entity anchors, fact-subject compatibility, SUPPORT-episode chronology.
- Shared `evaluate_competition()` on `RecallResult` in `rank()` and `inspect()` after admission, before threshold/sort.
- Opt-in transient interference (`apply_interference`, noisy-OR proactive/retroactive penalties); default diagnostic-only.
- [`docs/design-competition-hardening-interference-0.17.1.md`](design-competition-hardening-interference-0.17.1.md).

## 0.17.2 — Behavioural competition eligibility (done)

- Deterministic eligibility policy between diagnostic matching and transient interference; matcher unchanged.
- `BehavioralCompetitionEligibility` on every accepted pair when competition is enabled; interference uses eligible pairs only.
- `minimum_behavioral_strength` / `minimum_behavioral_cue_fit` on `CompetitionConfig`.
- [`docs/design-competition-behavioral-eligibility-0.17.2.md`](design-competition-behavioral-eligibility-0.17.2.md).

## 0.17.3 — Retrieval-induced inhibitory traces (done)

- Scope eligibility is separate from temporal transient-interference eligibility. Co-temporal pairs can be scope-eligible without interference pressure.
- Explicit `record_access` / `record_context_use` may persist bounded inhibitory traces. Presentation APIs do not.
- Traces are inspectable and are not read during recall.
- [`docs/design-retrieval-inhibition-0.17.3.md`](design-retrieval-inhibition-0.17.3.md).

## 0.17.3.1 — Inhibition recording hardening (done)

- `record_access()` keeps a consumed set after `min_score` and a reinforcement set after burst limiting.
- Inhibition is induced from the consumed set, including when no new activation reference is written.

## 0.17.4 — Persistent inhibition application (done)

- Opt-in scope-matched half-life recovery. Default `apply_to_recall=False` does not read the store.
- Persistent penalty is separate from transient interference and is applied first.
- [`docs/design-inhibition-application-0.17.4.md`](design-inhibition-application-0.17.4.md).

## 0.17.5 — Interference observability (done)

- Query-level `RetrievalInterferenceObservability` on `inspect_recall` and `MemoryAssessment`, from the pre-threshold discrimination set of one retrieval.
- Metamemory flags `COMPETING_MEMORIES`, `HIGH_INTERFERENCE`, and `RETRIEVAL_INHIBITION_ACTIVE`. Retrieval behaviour is unchanged.
- [`docs/design-interference-observability-0.17.5.md`](design-interference-observability-0.17.5.md).

## 0.17.6 — Transient interference reachability (done)

- Transient interference is retained and validated. `apply_interference` stays default off.
- Episodes inherit a shared active support slot, and structured pairs no longer apply cue fit twice at the behavioural strength gate.
- [`docs/design-transient-interference-reachability-0.17.6.md`](design-transient-interference-reachability-0.17.6.md) and [`docs/findings/0.17.6-transient-interference-reachability.md`](findings/0.17.6-transient-interference-reachability.md).

## 0.17.6.1 — Co-consumption inhibition hardening (done)

- Consumed memories are not inhibited through another representation of the same active semantic lineage.
- Same semantic slot does not count as consumption. Transient interference is unchanged.
- [`docs/design-co-consumption-inhibition-hardening-0.17.6.1.md`](design-co-consumption-inhibition-hardening-0.17.6.1.md).

## 0.17.7 — Interference and inhibition architecture freeze (done)

- Retrieval behaviour matches `0.17.6.1`. `apply_interference` and `apply_to_recall` stay off.
- One invariant suite locks competition, eligibility, transient interference, co-consumption, and persistent inhibition. Behavioural changes after this release need a new design note.
- [`docs/design-interference-inhibition-architecture-freeze-0.17.7.md`](design-interference-inhibition-architecture-freeze-0.17.7.md) and [`docs/findings/0.17.7-interference-inhibition-architecture-freeze.md`](findings/0.17.7-interference-inhibition-architecture-freeze.md).

## 0.18.0 — Source identity and provenance monitoring (done)

- Read-only `Memory.resolve_provenance(context)` with typed source attribution and explicit selected versus chunk-member representation.
- Tenant-scoped exact revision resolution for in-memory and PostgreSQL observation stores; no schema migration.
- Source identity, diversity, revision currency, missing-evidence states, and deterministic evidence bounds without retrieval or consolidation changes.
- [`docs/design-source-provenance-0.18.0.md`](design-source-provenance-0.18.0.md) and [`docs/findings/0.18.0-source-provenance.md`](findings/0.18.0-source-provenance.md).

## 0.18.1 — Source authority admission (done)

- Opt-in single-episode semantic promotion for one fact whose current observation revision declares `authority.basis=application_asserted`.
- Recurrence stays the default. Authority does not change confidence, retrieval, forgetting, or automatic correction.
- [`docs/design-source-authority-0.18.1.md`](design-source-authority-0.18.1.md) and [`docs/findings/0.18.1-source-authority.md`](findings/0.18.1-source-authority.md).

## 0.18.2 — Authoritative corrections and withdrawals (done)

- Opt-in replacement and withdrawal of one named semantic revision, with revision snapshots and knowledge-time hiding.
- Recurrence and 0.18.1 authority admission stay in place when correction is disabled.
- [`docs/design-authoritative-corrections-0.18.2.md`](design-authoritative-corrections-0.18.2.md) and [`docs/findings/0.18.2-authoritative-corrections.md`](findings/0.18.2-authoritative-corrections.md).

## 0.18.3 — Source-aware metamemory (included in 0.18.4)

- Separate source assessment and exact slot inspection, optional bounded storage capability.
- Pass-local typed decision diagnostics; no persistent unresolved ledger.
- Conservative historical attribution and unknown independence; cognitive behavior unchanged.
- Core and independent Bench validation covered canonical capped-source parity on PostgreSQL. The capability was included in the 0.18.4 release rather than published separately.
- [Design](design-source-aware-metamemory-0.18.3.md) and [usage](source-metamemory.md).

## 0.18.4 — Source provenance architecture freeze (2026-11-15)

- Consolidated contract audit and cross-feature in-memory/PostgreSQL regression suite.
- Separately approved read-side correction for stale prose from deferred readmissions.
- Core and independent Bench CI/strict gates pass: 392 unchanged query signatures plus one separately approved exact correction, with original gold/raw mismatch retained. All freeze acceptance criteria are resolved for the candidate.
- No new cognitive capability, mandatory protocol, migration or dependency.
- [Freeze design and invariant matrix](design-source-provenance-architecture-freeze-0.18.4.md), [corrective design](design-withdrawn-readmission-context-correction.md), and [findings](findings/0.18.4-source-provenance-architecture-freeze.md).

## 0.19.0–0.19.4 — Embedded memory development

These milestones describe the SQLite work consolidated into the 0.19.5 release, not separate published versions:

- 0.19.0: SQLite observations, checkpoints, transactions and migrations ([design](design-sqlite-foundation-0.19.0.md)).
- 0.19.1: episodic and semantic memory, history and provenance ([design](design-sqlite-derived-memory-0.19.1.md)).
- 0.19.2: persistent activation, dynamics, learning, inhibition and relationships ([design](design-sqlite-cognitive-state-0.19.2.md)).
- 0.19.3: embedded lifecycle and storage ownership ([design](design-embedded-runtime-0.19.3.md)).
- 0.19.4: reliability and workload hardening ([design](design-embedded-reliability-0.19.4.md), [findings](findings/0.19.4-embedded-reliability.md)).

## 0.19.5 — Embedded memory architecture freeze (2026-11-15)

- Core candidate `f289035351e3c101fd932f4d6d9e9cd56583ace0` adds the SQLite storage bundle and embedded lifecycle developed across 0.19.0–0.19.4, plus a PostgreSQL tenant learning-cleanup correction found during freeze validation.
- Core Python 3.11–3.13 and hosted cross-platform CI passed. Bench's strict frozen comparison passed on Python 3.12–3.13, with the separately approved frozen correction. On Python 3.11 it reports pre-existing additional differences from the original gold; all 22 run signatures match the earlier Core on the same runtime. Three-backend comparisons pass with the separately approved support-order differences. Raw comparisons and limits remain documented.
- Bench three-backend checks cover four representative workloads; Core contract and restart tests provide additional coverage. The documented validation applies to the identified source candidate; the release commit adds documentation and version metadata.
- [Freeze design](design-embedded-architecture-freeze-0.19.5.md), [candidate review](review-0.19.5-f289-candidate.md), and [findings](findings/0.19.5-embedded-architecture-freeze.md).

## 0.19.6 — Embedded runtime stabilisation (2026-11-17)

- Candidate `788444d` passed the Core Python 3.11–3.15 Linux/macOS/Windows matrix, PostgreSQL gates and independent Bench reviews with candidate-specific approvals. Raw benchmark comparisons remain available.
- Uninstrumented workloads document a substantial retrieval-scaling limitation. See the [design](design-embedded-stabilisation-0.19.6.md) and [findings](findings/0.19.6-embedded-stabilisation.md).

## 0.20.0 — Retrieval profiling (investigation; not published)

- Investigate the 0.19.6 retrieval scaling issue using versioned workloads,
  optional diagnostics and independent comparisons. No cognitive change is included.
- The [release preparation findings](findings/0.20.0-release-preparation.md)
  record evidence and open acceptance decisions.

## 0.20.1 — Retrieval optimization (released)

- Measured retrieval optimizations and historical reconstruction improvements.
  The [release findings](findings/0.20.1-release.md) identify the published
  artifacts, validation and remaining limitations.

## 0.20.2 — Retrieval and historical memory hardening (released)

- Repeat the released retrieval measurements, assess context preparation, and
  validate historical integrity, upgrades, snapshot safety and backend parity.
- The [release findings](findings/0.20.2-release.md) record the published
  candidate, evidence and bounded approvals.

## 0.20.3 — Architecture freeze and Agent handoff (in progress)

- Validate subject-scoped processing and Agent-facing public APIs, investigate
  scoped-read scaling, and consolidate the cognitive and operational contracts.
- The [design](design-core-architecture-freeze-0.20.3.md) defines the phased
  work and release gates. Candidate validation and release approval remain open.

## Later

- additional source connectors for application data (for example SQLite, APIs and queues);
- graph-oriented storage options;
- embedding-provider interfaces;
- LLM-provider interfaces;
- Agent integration and memory-context contracts after retrieval performance acceptance;
- additional independent benchmark coverage;
- enterprise-oriented extensions.
