# SQLite cognitive state (0.19.2) — design

Persist the existing activation, forgetting, learning, inhibition and entity-relationship protocols in the same SQLite database used for observations and derived memories. Continue to inject stores into `Memory`; working-memory snapshots remain transient. The existing algorithms, rankings and defaults do not change.

A new SQLite migration owns each state table and its tenant-scoped identities. Requests with an idempotency key use the existing backend's key and retry semantics; unkeyed requests remain distinct. UTC timestamps, weighted activation traces, inclusive and exclusive time boundaries, inhibition scopes, learning feedback fingerprints and relationship identities must round-trip. Unavailable records remain absent, not synthesized.

All operations run on the shared SQLite worker. Batch mutations and activation compaction are transactional. In particular, one learning plan must atomically record its feedback event, item outcomes, counts and associations; a duplicate matching fingerprint is a no-op, while a conflicting fingerprint raises `StorageError`. Writers acquire `BEGIN IMMEDIATE` before checking existing state. The per-operation guarantee does not make `Memory.process()` or a sequence of cognitive API calls atomic.

Verification compares SQLite, in-memory and PostgreSQL behavior for retries, tenant isolation, ordering, cutoffs, compaction, forgetting/reactivation, feedback and relationships. Restart tests must recover the same state and retrieval outcomes with the same clock and configuration, and exercise transaction rollback and migration from a populated earlier SQLite database. Run Core gates on Python 3.11–3.13 with PostgreSQL. Performance and operational limits remain subject to 0.19.4 hardening.

## Implementation findings

SQLite stores implement all five protocols through the shared worker and a separate migration. Tests exercise existing PostgreSQL contract cases against SQLite, restart retrieval, activation compaction, feedback rollback and retries, concurrent duplicate learning, inhibition bounds, tenant isolation and an upgrade from a populated 0.19.1 database. Full database-enabled Core suites passed on Python 3.11–3.13 with 992 tests per runtime and no skips; a subsequently added upgrade case passed separately on each runtime. Ruff, formatting and mypy passed on all three runtimes.

Some reads and activation compaction scan tenant-scoped state, while semantic operations retain the inventory materialization limitation documented for 0.19.1. This is a functional implementation rather than a performance freeze; workload scaling and operational hardening remain for 0.19.4.
