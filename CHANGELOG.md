# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-04

### Added

- Observation policy output persisted on `StoredObservation` (`attention_score`, `retention_class`, `policy_reasons`).
- Episode domain models (`EpisodeInput`, `StoredEpisode`, `EpisodeEncodingResult`, and related types).
- `DeterministicEpisodicEncoder` with conversation/thread grouping, time-gap segmentation, salience scoring, and evidence links.
- `EpisodeStore` protocol with in-memory and PostgreSQL backends (`cognema.memories`, `memory_evidence`, `memory_entities`).
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
- In-memory and PostgreSQL observation stores (Postgres via optional `cognema[postgres]`).
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

- Initial open-source package structure under `src/cognema`.
- Public API exports: `Memory`, `MemoryEvent`, and `RecallResult` (superseded in `0.1.0`).
- `MemoryEvent` and `RecallResult` typed models with validation (event path removed in `0.1.0`).
- Storage protocol and `InMemoryStorage` backend (replaced by observation stores in `0.1.0`).
- Deterministic token-overlap recall implementation.
- Test suite for models, storage, and memory behavior.
- Project documentation, contribution policies, and security policy.
- GitHub Actions CI and Trusted Publishing workflow.
