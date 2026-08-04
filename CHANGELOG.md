# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-04

### Added

- Observation models and ingestion pipeline (`ObservationInput`, revisions, policies, retention).
- `ObservationStore` and `CheckpointStore` protocols.
- PostgreSQL schema, migrations, observation store, and checkpoint store via optional `cognema[postgres]`.
- `PostgresTableSource` with compound `(updated_at, id)` cursors and soft-delete column support.
- `Memory.observe_input()` and `Memory.ingest()` for normalized observation ingestion.
- Revision history for create, update, delete, and restore.
- Docker PostgreSQL example under `examples/postgres_datasource/`.
- Unit tests and optional `@pytest.mark.postgres` integration tests.

### Changed

- Package version bumped to `0.1.0`.
- Roadmap: observation ingestion is `0.1`; episodic memory moves to `0.2`, with later milestones shifted accordingly.
- Documentation updated for observation architecture and deployment models.

## [0.0.1] - 2026-08-03

### Added

- Initial open-source package structure under `src/cognema`.
- Public API exports: `Memory`, `MemoryEvent`, and `RecallResult`.
- `MemoryEvent` and `RecallResult` typed models with validation.
- Storage protocol and `InMemoryStorage` backend.
- Deterministic token-overlap recall implementation.
- Test suite for models, storage, and memory behavior.
- Project documentation, contribution policies, and security policy.
- GitHub Actions CI and Trusted Publishing workflow.
