# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.4] - 2026-08-28

### Added

- Current-at-snapshot temporal semantics: `valid_at` selects validity time, not historical query intent.
- Saturating aggregation of per-support evidence relevance for multi-episode semantic claims.
- One-hop associative reach from cue-matched episode entities to related semantic memories.
- `RetrievalDiagnostics.associative_fit` for inspection.
- Package regression tests in `tests/test_temporal_associative_recall.py`.

### Changed

- `TemporalRetrievalMode.HISTORICAL` now requires explicit historical cue intent; `valid_at` with a normal recommendation query uses `CURRENT` snapshot mode.
- Historical cue lexicon tightened to phrases and distinctive tokens (no longer treats `to` / `when` alone as historical).
- Authoritative current admission accepts relevance ratio **or** distinctive token overlap count (not only the soft activation floor).
- Evidence-linked relevance aggregates multiple supporting episodes with a bounded saturating formula.

### Fixed

- `as_of == valid_at` demo/bench snapshots no longer misclassify as historical retrieval.
- Old but authoritative semantics with explicit query overlap (for example hiking interest) remain eligible without clearing the soft floor.
- Colour and product-fit semantics can be reached through aggregated evidence and entity-linked association.

### Preserved

- Global `retrieval_threshold`, `semantic_soft_admission_floor`, and decay defaults unchanged.
- `SUPERSEDED` cannot use authoritative current admission; historical-intent queries still retrieve superseded slots.
- Working-memory selection unchanged.

### Notes

- No migration required.

## [0.15.3] - 2026-08-28

### Added

- Cardinality-one reconciliation by supporting evidence chronology when world validity windows are unspecified.
- Bounded rank-time evidence-linked semantic relevance from supporting episode statements already in the candidate set.
- Authoritative current semantic admission (`RetrievalEligibility.SEMANTIC_CURRENT_ADMISSION`) for `ACTIVE`, valid, relevant facts without lowering the activation floor.
- `ActivationConfig.enable_semantic_evidence_linking`, `max_evidence_link_derivations`, and `semantic_current_min_relevance`.
- `RetrievalDiagnostics.direct_cue_fit`, `evidence_linked_fit`, and `semantic_relevance` for inspection.
- Package regression tests in `tests/test_semantic_state_associative_recall.py`.

### Changed

- Incremental `process()` restores `slot_key` from stored semantic memories when hydrating revision inputs.
- Successor selection for supersession follows evidence/validity chronology rather than always treating the incoming candidate as successor.
- `inspect_recall()` can report `FILTERED_SEMANTIC_STATUS` when superseded semantics block current admission.

### Fixed

- Competing cardinality-one claims without explicit validity windows reconcile to a single `ACTIVE` authority (for example jacket size L then M).
- Plain-language cues can reach semantics whose predicates are absent from the query via evidence-linked relevance.
- Long-lived relevant `ACTIVE` semantics admit through current-authority relevance without bypassing unrelated facts.

### Preserved

- `SemanticMemoryStatus.ACTIVE` remains the current-authority status; no rename to `CURRENT`.
- Global `retrieval_threshold`, `semantic_soft_admission_floor`, and decay defaults unchanged.
- `TemporalRetrievalMode.CURRENT` is not broadened to `valid_at is not None`.
- Tied evidence chronology for unspecified ONE competitors still yields `CONFLICTS`.

### Notes

- No migration required.

## [0.15.2] - 2026-08-27

### Added

- Lexical semantic slot matching for plain-language string cues via predicate tokenisation and statement/object overlap.
- Bounded semantic soft admission with `semantic_soft_admission_floor`, `max_soft_admitted_semantics`, and `lexical_slot_min_overlap` on `ActivationConfig`.
- `RecallInspectionDisposition.FILTERED_BELOW_SOFT_FLOOR` and `FILTERED_INSUFFICIENT_RELEVANCE` for clearer inspection outcomes.
- Package regression tests for long-horizon semantic recall in `tests/test_long_horizon_semantic_recall.py`.

### Changed

- NEUTRAL and HISTORICAL string cues can open semantic slot admission when lexical relevance matches; structured predicate/entity cues are unchanged.
- Soft admission applies an activation floor for lexical text-only cues; structured cues bypass the floor.
- Historical soft admission allows visible superseded revisions at `valid_at`; live recall still requires `ACTIVE` semantics.

### Fixed

- Long-lived `ACTIVE` semantic facts (for example current jacket size) can be soft-admitted for relevant plain-language queries without lowering the global retrieval threshold.
- `inspect_recall()` distinguishes below-threshold, below soft floor, and insufficient lexical relevance rejections.

### Preserved

- Global `retrieval_threshold`, `decay`, and `time_unit_seconds` defaults unchanged.
- 0.15.1 evidence chronology and processing-cadence invariance.
- Historical structured admission and superseded SUPPORT exclusion on live current-state recall.

### Notes

- No migration required.

## [0.15.1] - 2026-08-27

### Added

- `Memory.inspect_recall()` for bounded recall inspection with terminal dispositions, cognitive trace detail, and activation components.
- Derived cognitive activation references (`ENCODED`, `SUPPORTED`) from episode and semantic evidence chronology.
- `CognitiveReferenceTrace`, `CognitiveTraceOrigin`, `RecallInspectionResult`, and `RecallInspectionDisposition` public models.
- `InspectableDeclarativeActivator` protocol for inspection-capable activators.
- `RecallInspectionUnsupportedError` when a custom activator does not support inspection.
- Package regression tests for processing-cadence recall stability in `tests/test_incremental_recall_stability.py`.

### Changed

- Declarative base-level activation no longer synthesizes an implicit trace from storage `created_at`.
- Episode encoding references use `ended_at`; semantic support references use supporting episode evidence times.
- Recall and forgetting share the same derived cognitive trace builder.
- PostgreSQL semantic reconciliation passes `as_of` when creating new semantic memory rows.

### Fixed

- Incremental and deferred `process()` cadence no longer produces materially different recall activation for equivalent source evidence.
- Historical import and backfill no longer receive artificial recency from batch materialisation timestamps.

### Preserved

- `recall()` result shape and semantics; inspection is additive.
- Persisted `RETRIEVED` / `REHEARSED` access references and learning reinforcement traces.
- Retrieval threshold, forgetting, supersession, and working-memory selection behaviour.

### Notes

- No migration required.
- **CogKuraBench 0.3.1** (`customer_decision_context_v1`, cogkura backend): incremental standard replay raw recall improved from 0 to 1; evidence-group coverage remains 0/5; `evidence_coverage_at_budget` remains 0.0. All lifecycle cases now report raw recall 1 (cadence parity); query-only prepare no longer diverges from incremental.
- **CogKura Demo 0.3.2** compare (waterproof-jacket, inspect-only): Full History 5/5, Search 4/5, CogKura **0/5** labelled coverage with 0 context tokens (0.15.0 baseline was 2/5). Further retrieval tuning is out of scope for this patch.

## [0.15.0] - 2026-08-22

### Added

- `Memory.process()` orchestrates episodic encoding and semantic consolidation with one shared `as_of` timestamp.
- `Memory.prepare_context()` returns bounded working memory and metamemory assessment in one read operation.
- `MemoryContext` immutable provider-neutral boundary with structured fields and deterministic `render()`.
- `Memory.record_context_use()` records use of selected context memories via existing access semantics.
- Shared declarative retrieval inside `prepare_context()` (one rank pass per call).
- [`docs/application-integration.md`](docs/application-integration.md) and [`examples/application_context.py`](examples/application_context.py).
- [`docs/design-application-integration-memory-context-0.15.0.md`](docs/design-application-integration-memory-context-0.15.0.md).

### Changed

- `select_working_memory()` and `assess_memory()` reuse internal shared retrieval helpers without behaviour change.
- PyPI `Development Status` classifier updated from `2 - Pre-Alpha` to `4 - Beta`.

### Fixed

- Restored naive `valid_at` rejection at the shared declarative ranking boundary so `select_working_memory()`, `assess_memory()`, and `prepare_context()` match `recall()` compatibility.

### Preserved

- All existing lower-level cognitive APIs and presentation-vs-use semantics.
- Working-memory and metamemory candidate-pool compatibility with standalone methods.

### Notes

- No migration required.
- `prepare_context()` raises when metamemory is disabled, matching `assess_memory()`.
- Package `__version__` aligned with `pyproject.toml` (`0.15.0`).

## [0.14.4] - 2026-08-15

### Added

- Structured retrieval diagnostics on `RecallResult.diagnostics` that separate accessibility activation from final rank activation.
- Eligibility and admission diagnostics that expose threshold versus slot-admission pathways.
- Semantic and SUPPORT provenance diagnostics, including selected supporting semantic revision and observation evidence ids.
- [`docs/design-retrieval-diagnostics-support-provenance-0.14.4.md`](docs/design-retrieval-diagnostics-support-provenance-0.14.4.md).

### Fixed

- SUPPORT slot-fit inheritance now keeps derivation-backed semantic revision provenance through ranking diagnostics.

### Preserved

- Query-coverage accessibility and precision-aware lexical ranking.
- Temporal retrieval modes, soft admission semantics, and global ranking.
- Conjunctive structured matching, positive structured slot fit, and stale SUPPORT exclusion.
- Metamemory, working memory, forgetting, and learning behavior.

### Notes

- No migration required.
- `RecallResult.activation` remains accessibility; ranking-specific signals are diagnostic only.
- Package `__version__` aligned with `pyproject.toml` (`0.14.4`).

## [0.14.3] - 2026-08-15

### Fixed

- Structured retrieval cues now apply entity, predicate, and object constraints conjunctively.
- Object constraints are no longer bypassed by higher-priority matching branches.
- Exact semantic slot matches provide positive bounded ranking evidence relative to candidates without applicable slot evidence.
- SUPPORT episodes inherit structured fit only from semantic slots they actually support.
- [`docs/design-conjunctive-slot-ranking-0.14.3.md`](docs/design-conjunctive-slot-ranking-0.14.3.md).

### Preserved

- Query-coverage accessibility and precision-aware lexical ranking.
- Temporal retrieval modes, soft admission, and global ranking.
- Multi-entity conjunction, stale SUPPORT exclusion, metamemory answerability.
- Working memory, forgetting, and learning.

### Notes

- No migration required.
- `RecallResult.activation` remains accessibility; structured adjustment is diagnostic in `reason` only.
- Package `__version__` aligned with `pyproject.toml` (`0.14.3`).

## [0.14.2] - 2026-08-15

### Added

- Explicit internal temporal retrieval modes (`neutral`, `current`, `historical`).
- Deterministic structured semantic-slot fit for eligible-candidate ordering.
- Deterministic metamemory answerability (`resolved`, `unresolved`, `not_applicable`).
- [`docs/design-temporal-slot-answerability-0.14.2.md`](docs/design-temporal-slot-answerability-0.14.2.md).

### Changed

- Historical mode can activate text/entity-derived soft admission; present lifecycle status does not bias historical ranking.
- Semantic slot fit refines eligible-candidate ordering without changing accessibility or the retrieval threshold.
- `MISSING_KNOWLEDGE` distinguishes unresolved knowledge from weak retrieval.

### Preserved

- Accessibility/ranking separation from `0.14.1`.
- Soft admission, global ranking, association, stale suppression, forgetting, learning, and working-memory selection.
- Metamemory monitoring flags and existing confidence thresholds.

### Notes

- No migration required.
- `RecallResult.activation` remains accessibility; structured slot fit is diagnostic in `reason` only.
- Package `__version__` aligned with `pyproject.toml` (`0.14.2`).

## [0.14.1] - 2026-08-15

### Changed

- Lexical current-state detection inspects query text independently of structured predicate/object fields.
- Historical semantic slot admission uses `valid_at` visibility; present `SUPERSEDED` status does not block historical admission.
- Query-coverage partial matching drives accessibility and the retrieval threshold; precision-aware F1 orders eligible candidates only (`enable_text_precision_matching`).
- Current-state activation bonuses apply to cue-matched semantic slots and their SUPPORT episodes, not every ACTIVE memory.
- [`docs/design-retrieval-corrections-0.14.1.md`](docs/design-retrieval-corrections-0.14.1.md).

### Preserved

- `0.14` soft entity admission and global ranking (admission is threshold bypass, not rank priority).
- `0.13` superseded SUPPORT exclusion for live current-state retrieval.
- Working-memory selection, metamemory thresholds, forgetting, and `conjunction_weight`.

### Notes

- No migration required.
- `RecallResult.activation` remains accessibility; internal ranking is diagnostic in `reason` only.
- Package `__version__` aligned with `pyproject.toml` (`0.14.1`).

## [0.14.0] - 2026-08-14

### Added

- Soft entity-based semantic slot admission (`enable_entity_slot_admission`).
- Precision-aware deterministic text matching for declarative partial match and working-memory goal relevance (`enable_text_precision_matching`).
- [`docs/design-retrieval-eligibility-ranking.md`](docs/design-retrieval-eligibility-ranking.md).

### Changed

- Admitted candidates compete in the global activation ranking; admission no longer grants rank priority.
- Current-state lifecycle bias applies only when current-state policy is active (live current/predicate/object cues).
- Historical `valid_at` retrieval does not apply present-day lifecycle bias.
- SUPERSEDED-only SUPPORT exclusion keys off current-state policy, not slot admission alone.
- Working-memory text relevance uses precision-aware F1 matching.

### Preserved

- `0.13` superseded SUPPORT exclusion for live current-state retrieval.
- `0.13` multi-entity conjunction.
- `0.13` incident tag seeding.
- `0.13` metamemory `MISSING_KNOWLEDGE`.
- Same-slot SUPPORT collapse at rank and in working memory.

### Notes

- No migration required.
- Package `__version__` aligned with `pyproject.toml` (`0.14.0`).

## [0.13.0] - 2026-08-14

### Added

- Gated semantic slot admission (`slot_admission_requires_current_state_or_predicate`, `force_slot_admission`).
- Multi-entity conjunction bonus (`enable_multi_entity_conjunction`, `conjunction_weight`).
- Incident cue IDF scaling and tag seeding (`distinctive_token_idf_scale`, `incident_cue_tokens`, `enable_incident_tag_seeding`).
- Hard exclusion of SUPERSEDED-only SUPPORT episodes on current-state recall (`exclude_superseded_support_on_current_state`).
- Metamemory `MemoryAssessmentFlag.MISSING_KNOWLEDGE` with coverage/strength thresholds.
- Working-memory same-slot SUPPORT collapse and stale-goal penalty (`collapse_same_slot_support`, `stale_goal_penalty`).
- Rank-time same-slot SUPPORT collapse (`ActivationConfig.collapse_same_slot_support`).
- [`docs/design-gated-slot-admission.md`](docs/design-gated-slot-admission.md).

### Changed

- `Memory.recall()` passes `valid_at` and an episode slot index into declarative activation.
- Package `__version__` aligned with `pyproject.toml` (`0.13.0`).

### Notes

- Entity ids and incident tags seed spreading/partial match only; they no longer trigger slot admission by default.
- No migration required.

## [0.12.0] - 2026-08-13

### Added

- String-cue entity seeding for spreading activation (`enable_text_entity_seeding`, `seeded_entity_partial_match_weight`).
- Semantic slot admission before the retrieval threshold cut (`enable_semantic_slot_admission`).
- Superseded-slot current-state penalty for supporting episodes; `ActivationComponents.current_state`.
- Numeric-token normalisation for near-duplicate collapse only (`collapse_normalize_numeric_tokens`).
- `Memory.record_access(..., min_score=...)`, `ActivationConfig.access_minimum_score`, and optional access burst limits.
- [`docs/design-string-cues-current-state.md`](docs/design-string-cues-current-state.md).

### Changed

- `recall()` builds an episode support index from superseded semantics (derivation-only when `valid_at is None`).
- In-memory semantic `list(..., status=SUPERSEDED)` returns superseded rows when explicitly requested.
- Spreading activation accepts optional `spread_sources` for seeded entity propagation.

### Notes

- `recall()` is presentation; `record_access()` is use. No migration required.
- Explicit `RetrievalCue.entity_ids` behaviour is unchanged from `0.11` when seeding is disabled.

## [0.11.0] - 2026-08-13

### Added

- Simulated time on `Memory.encode_episodes(..., as_of=...)` and `Memory.consolidate_semantics(..., as_of=...)`.
- Episode visibility at `valid_at` in `recall()`, `select_working_memory()`, and `assess_memory()` (`started_at <= valid_at`).
- Candidate-set IDF for text partial matching (`ActivationConfig.enable_candidate_idf`).
- Near-duplicate collapse at rank time (`enable_duplicate_collapse`, `duplicate_jaccard_threshold`).
- Current-state activation bias for active semantics and configured cue tokens.
- Importance-scaled forgetting and semantic-support protection (`ForgettingConfig`).
- [`docs/design-ranking-time-current-state.md`](docs/design-ranking-time-current-state.md).

### Changed

- Episode and semantic store upserts honour optional `as_of` for `created_at` / `updated_at` (in-memory and Postgres).
- `apply_forgetting()` excludes episodes with `started_at > as_of` and protects supporting episodes from `FORGOTTEN`.

### Notes

- No PostgreSQL migration is required.
- String cues remain the compatibility path; structured `RetrievalCue` fields are unchanged.

## [0.10.0] - 2026-08-12

### Added

- Deterministic metamemory and memory monitoring.
- `Memory.assess_memory()`, `MemoryAssessment`, `MetamemorySignals`, `MetamemoryItem`, `MemoryAssessmentFlag`, `MetamemoryConfig`.
- `MemoryMonitor` protocol and `DeterministicMemoryMonitor`.
- Cue-coverage monitoring, retrieval-strength monitoring, retrieval-weighted evidence confidence.
- Semantic conflict monitoring, observation-provenance diversity, optional evidence freshness.
- Continuous forgetting-pressure monitoring and context-aware learned-utility monitoring.
- Structured diagnostic warning flags.
- Metamemory example and [`docs/metamemory.md`](docs/metamemory.md).

### Changed

- Working-memory relevance matching is shared with metamemory through `algorithms/relevance.py`.
- Learning count aggregation is reusable by working memory and metamemory.

### Notes

- Metamemory is read-only: no `record_access()`, no forgetting lifecycle writes, no learning feedback.
- No overall answer-confidence score is produced.
- No PostgreSQL migration is required.

## [0.9.0] - 2026-08-11

### Added

- Outcome-driven learning via `Memory.learn()` and `Memory.list_learning_state()`.
- `LearningOutcome`, `LearningFeedback`, `LearningConfig`, and persistence through `LearningStore`.
- HELPFUL learning reinforcement traces merged into ACT-R base-level history (without writing to `memory_activation_references`).
- Contextual utility for working-memory selection (`learned_utility_weight`).
- Persistent learned associations for spreading activation (`learned_association_scale`).
- `DeterministicLearningProcessor`, `InMemoryLearningStore`, `PostgresLearningStore`.
- Migration `007_learning_reinforcement.sql`.
- Learning example and [`docs/learning.md`](docs/learning.md).

### Notes

- `learn()` does not call `record_access()`; ACTIVATION ≠ UTILITY ≠ ASSOCIATION.
- `INCORRECT` does not mutate semantic confidence or revisions; corrections stay on the `0.8` observe → consolidate path.

## [0.8.0] - 2026-08-11

### Added

- Temporal semantic reconsolidation with revision history and deterministic relation classification.
- `DeterministicSemanticReconciler` and `SemanticUpdateRelation` (`REINFORCES`, `COEXISTS`, `SUPERSEDES`, `CONFLICTS`).
- `Memory.list_semantic_revisions()` and `valid_at` on `list_semantic_memories()`, `recall()`, and `select_working_memory()`.
- Structured `valid_from` / `valid_until` parsing on `semantic_facts` metadata.
- `SemanticMemoryStore.apply_reconciliation()` with in-memory and PostgreSQL backends.
- Migration `006_semantic_reconsolidation.sql` (`semantic_claim_revisions`, `semantic_revision_relations`, legacy backfill).
- Reconsolidation example and [`docs/reconsolidation.md`](docs/reconsolidation.md).

### Changed

- **Breaking (pre-1.0):** `SemanticConsolidator.consolidate()` returns `list[SemanticRevisionCandidate]`.
- `Memory.consolidate_semantics()` reconciles revisions atomically; `deactivate_missing` removed from this path.
- PostgreSQL semantic `memories.valid_from/until` now store world validity (often `NULL`), not support timestamps.

### Notes

- Unknown temporal chronology yields `CONFLICTS`, not implicit supersession.
- Reconsolidation does not append ACT-R activation references.

## [0.7.0] - 2026-08-11

### Added

- Baddeley-inspired bounded working-memory selection.
- `Memory.select_working_memory()`.
- `WorkingMemoryConfig`, `WorkingMemorySnapshot`, `WorkingMemoryItem`, and `WorkingMemoryComponents`.
- `DeterministicWorkingMemorySelector`.
- Goal-aware working-memory ranking.
- Competitive inhibition for redundant memories.
- Short-timescale working-memory decay and refresh.
- Item and prompt-token budgets.
- Injectable `TokenEstimator` and dependency-free `ApproximateTokenEstimator`.
- Working-memory example and [`docs/working-memory.md`](docs/working-memory.md).

### Notes

- Working memory is transient and not persisted.
- Working-memory decay does not modify ACT-R base-level activation.
- Selection does not automatically reinforce activation history.
- `Memory.recall()` semantics remain unchanged.

## [0.6.0] - 2026-08-10

### Added

- Ebbinghaus-inspired forgetting lifecycle (`ACTIVE` → `FADING` → `FORGOTTEN`) from ACT-R base-level only.
- `MemoryDynamicsStore` with in-memory and PostgreSQL backends.
- `Memory.apply_forgetting()`, `ForgettingConfig`, and `include_forgotten` on `Memory.recall()`.
- `EbbinghausForgettingEvaluator` and `ActivationReferenceTrace` weighted base-level activation.
- `ActivationStore.compact_references()` and `list_reference_traces()` (replaces `list_reference_times`).
- Migration `005_forgetting_dynamics.sql` (`memory_dynamics`, reference `weight` column).
- [`docs/forgetting.md`](docs/forgetting.md) and forgetting/compaction tests.

### Changed

- **Breaking:** `ActivationStore.list_reference_times()` removed; use `list_reference_traces()`.
- `Memory.record_access()` reactivates forgotten/fading dynamics for evaluated identities.
- `Memory.clear()` clears dynamics after activation references.
- `MemoryReference.weight` defaults to `1` (weighted compaction merges preserve unit-weight semantics).

### Notes

- Cognitive forgetting excludes memories from recall; it does not delete stored episodes or semantic claims.
- `Memory.sleep()` remains a synchronous no-op; call `apply_forgetting()` explicitly.

## [0.5.0] - 2026-08-09

### Added

- Spreading activation (`DeterministicSpreadingActivator`) with bounded multi-hop entity–memory propagation.
- `ActivationConfig` fields: `spreading_decay`, `spreading_max_hops`, `spreading_min_activation`.
- `tests/test_spreading_activation.py` and spreading evaluation fixture.
- [`docs/spreading-activation.md`](docs/spreading-activation.md).

### Changed

- **Breaking:** Project renamed from Cognema to Cogkura.
- **Breaking:** PyPI package and import path are now `cogkura` (`pip install cogkura`, `from cogkura import Memory`).
- **Breaking:** PostgreSQL schema default is now `cogkura` (was `cognema`).
- **Breaking:** Demo/integration env vars are now `COGKURA_POSTGRES_*` (was `COGNEMA_POSTGRES_*`).
- Website and repository URLs updated to `cogkura.com` and `github.com/cogkura/cogkura`.
- Spreading activation is enabled by default (`enable_spreading_activation=True`). Disable to reproduce `0.4` recall behaviour.

## [0.4.0] - 2026-08-07

### Added

- ACT-R declarative activation (`ACTRDeclarativeActivator`) with base-level and partial matching.
- Activation models (`RetrievalCue`, `ActivationConfig`, `ActivationComponents`, `MemoryReference`, etc.).
- `ActivationStore` with in-memory and PostgreSQL backends.
- `Memory.record_access()` for explicit reinforcement.
- Migration `004_declarative_activation.sql` (`memory_activation_references`).
- `examples/declarative_activation.py` and evaluation fixture tests.

### Changed

- **Breaking:** `Memory.recall()` now ranks episodic and semantic memories (not observations).
- **Breaking:** `RecallResult` references `StoredEpisode | StoredSemanticMemory` with activation metadata.
- Package version bumped to `0.4.0`.
- `Memory.clear()` order is now activation → semantic → episodic → observations.
- PostgreSQL apps should pass `PostgresActivationStore` alongside other Postgres stores.

### Removed

- Observation token-overlap placeholder retrieval.

## [0.3.0] - 2026-08-05

### Added

- Semantic domain models (`SemanticMemoryInput`, `StoredSemanticMemory`, `SemanticConsolidationResult`, and related enums).
- `MetadataSemanticExtractor` reading `observation.metadata["semantic_facts"]` with malformed-entry rejection counting.
- `ComplementaryLearningSemanticConsolidator` with canonicalisation, recurrence promotion, contradiction handling, and deterministic statement projection.
- `SemanticMemoryStore` protocol with in-memory and PostgreSQL backends (`semantic_claims`, `memory_derivations`).
- `ObservationStore.get_many()` for tenant-scoped observation loads by Cogkura ID.
- `Memory.consolidate_semantics()` and `Memory.list_semantic_memories()` public APIs.
- Migration `003_semantic_consolidation.sql` and example init SQL sync.
- `examples/semantic_consolidation.py` and evaluation fixture tests.

### Changed

- Package version bumped to `0.3.0`.
- `Memory.clear()` order is now semantic → episodic → observations.
- PostgreSQL apps should pass `PostgresSemanticMemoryStore` alongside observation and episode stores.

## [0.2.0] - 2026-08-04

### Added

- Observation policy output persisted on `StoredObservation` (`attention_score`, `retention_class`, `policy_reasons`).
- Episode domain models (`EpisodeInput`, `StoredEpisode`, `EpisodeEncodingResult`, and related types).
- `DeterministicEpisodicEncoder` with conversation/thread grouping, time-gap segmentation, salience scoring, and evidence links.
- `EpisodeStore` protocol with in-memory and PostgreSQL backends (`cogkura.memories`, `memory_evidence`, `memory_entities`).
- `Memory.encode_episodes()` and `Memory.list_episodes()` public APIs.
- Migration `002_episodic_memory.sql` and multi-file `apply_migrations()` runner.
- Unit tests for episode models, encoder, stores, facade, and migration runner.

### Changed

- Package version bumped to `0.2.0`.
- `Memory.clear()` now clears episodes before observations.
- PostgreSQL example init SQL synced with `002` schema for fresh Docker volumes.

## [0.1.0] - 2026-08-04

### Added

- Observation models and ingestion pipeline (`ObservationInput`, revisions, policies, retention).
- `ObservationStore` and `CheckpointStore` protocols.
- In-memory and PostgreSQL observation stores (Postgres via optional `cogkura[postgres]`).
- `PostgresTableSource` with compound `(updated_at, id)` cursors and soft-delete column support.
- `Memory.observe()`, `Memory.ingest()`, and tenant-scoped `Memory.recall()`.
- Revision history for create, update, delete, and restore.
- Docker PostgreSQL example under `examples/postgres_datasource/`.
- Unit tests and optional `@pytest.mark.postgres` integration tests.

### Changed

- Package version bumped to `0.1.0`.
- Single observation-based API: removed `MemoryEvent`, string `observe`, and event `MemoryStorage`.
- `RecallResult` now references `StoredObservation`.
- Roadmap: observation ingestion is `0.1`; episodic memory moves to `0.2`, with later milestones shifted accordingly.
- Documentation updated for observation architecture and deployment models.

## [0.0.1] - 2026-08-03

### Added

- Initial open-source package structure under `src/cogkura`.
- Public API exports: `Memory`, `MemoryEvent`, and `RecallResult` (superseded in `0.1.0`).
- `MemoryEvent` and `RecallResult` typed models with validation (event path removed in `0.1.0`).
- Storage protocol and `InMemoryStorage` backend (replaced by observation stores in `0.1.0`).
- Deterministic token-overlap recall implementation.
- Test suite for models, storage, and memory behavior.
- Project documentation, contribution policies, and security policy.
- GitHub Actions CI and Trusted Publishing workflow.
