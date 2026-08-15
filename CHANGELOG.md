# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
