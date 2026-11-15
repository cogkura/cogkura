# SQLite foundation (0.19.0) — design proposal

## Purpose and status

0.19.0 introduces SQLite implementations of the existing observation and checkpoint storage contracts. It does not change cognitive algorithms, rankings, defaults, source authority, or the public memory API. This note is a design for implementation, not a claim that SQLite support has shipped. The proposed 0.19.1–0.19.5 milestones below are planning targets, not committed releases.

The acceptance boundary is deliberately narrow: observations, their exact revisions, and connector checkpoints survive a close and reopen, with the logical behaviour of the existing stores. Other stores supplied to `Memory` retain their existing defaults unless explicitly injected. Consequently, 0.19.0 alone is **not** a durable cognitive memory engine: episodes, semantics, and cognitive state will not survive a restart when their default in-memory stores are used.

## Storage inventory and release boundaries

The protocols in `src/cogkura/storage/base.py` are the source of truth. Each has an in-memory and PostgreSQL implementation today; `SemanticInspectionStore` is an optional capability implemented by the semantic stores.

| Protocol | State and important contract | Proposed SQLite milestone |
| --- | --- | --- |
| `ObservationStore` | Tenant-scoped current observations, exact historical revisions, source identity and retention | 0.19.0 |
| `CheckpointStore` | Tenant and connector scoped checkpoint | 0.19.0 |
| `EpisodeStore` | Episodes, encoding context, observation evidence and deactivation | 0.19.1 |
| `SemanticMemoryStore` | Claims, derivations, revision history and atomic reconciliation | 0.19.1 |
| `SemanticInspectionStore` | Optional bounded slot inventory, including history and relations | 0.19.1 candidate; capability remains optional |
| `ActivationStore` | Access references and weighted compaction | 0.19.2 |
| `MemoryDynamicsStore` | Forgetting lifecycle and reactivation | 0.19.2 |
| `LearningStore` | Feedback, utility, associations and reinforcement; atomic, idempotent plans | 0.19.2 |
| `EntityRelationshipStore` | Directed entity associations | 0.19.2 |
| `InhibitionStore` | Scoped inhibitory traces, including request-id deduplication | 0.19.2 |

Transient working-memory snapshots remain transient. The target checkpoints are functional backend coverage in 0.19.2, a cohesive embedded lifecycle in 0.19.3, reliability hardening in 0.19.4, and independent validation and architecture freeze in 0.19.5. The 0.20.0 integration contract and agent transport, sessions and context reuse are outside this work. Cross-backend import/export and cloud sync are deferred.

## Integration and ownership

Use Python's standard-library `sqlite3`; Core remains dependency-free. Add SQLite stores under the existing `storage/` package and SQLite-specific migration resources under `migrations/`, without importing PostgreSQL's optional SQLAlchemy dependency. A shared, explicitly managed SQLite connection owner is injected into both stores, which are then passed to the existing `Memory(observation_store=..., checkpoint_store=...)` constructor. The implementation must expose a supported way to initialize migrations and close that owner in 0.19.0; its exact low-level names can follow repository conventions. It is not the cohesive `Memory` convenience runtime planned for 0.19.3.

One Cogkura-owned database file contains all SQLite store tables as they are added. Never alter customer source tables. The caller owns the storage owner's lifetime; `Memory` does not silently close an injected owner. Initialization must finish before use, and close must be explicit, idempotent, and drain accepted work before closing the connection and worker. Reject new work once closing begins. Reopening constructs a new owner and reruns migration checks. Document that mixing persistent observations with default in-memory derived stores does not persist derived memories, and that `Memory.clear()` does not clear connector checkpoints (the checkpoint protocol has no clear operation).

## Connection, transactions and concurrency

- Create, use and close each `sqlite3` connection on its dedicated worker thread. Submit an entire store operation, including all SQL and row materialization, as one unit; never run SQLite calls or lock waits on the event-loop thread. A single shared owner serializes its own operations; do not share a connection concurrently between threads. Multiple owners or processes coordinate through SQLite's file locks.
- Configure and verify WAL on opening a file database, enable `foreign_keys` on **every** connection, and use an explicit finite busy timeout (proposed default: 5 seconds). Set and document `synchronous=FULL` for durable commits. WAL allows readers on separate connections while a writer is active; it does not promise parallel writes. Treat failure to enable the required settings as an initialization error, not a silent fallback. WAL requires a suitable local filesystem and may create `-wal` and `-shm` sidecars.
- Use explicit transactions. Each observation ingest obtains a write reservation with `BEGIN IMMEDIATE` **before** reading its current row; the decision, current-row update and revision insert commit together. This serializes competing writers across owners and avoids duplicate revision numbers or lost updates. A checkpoint set is one atomic transaction. Reads see committed data; multi-query operations requiring one consistent view use a read transaction. Roll back on every operation failure before reusing the connection.
- Busy/locked after the timeout becomes a `StorageError` with the cause preserved and enough context to diagnose contention. Other SQLite I/O, schema and transaction errors are also surfaced as `StorageError`; invalid application inputs retain existing validation behaviour. Do not retry a partially completed transaction invisibly. Test contention across independent connections/processes, including success after lock release and bounded failure after timeout.
- Cancellation of an awaiting coroutine cannot stop an already running SQLite call safely. Once submitted, the worker must finish or roll back the complete operation and remain usable; cancellation does not prove that a write failed to commit. Document this uncertain outcome and safe retry using existing ingest/checkpoint semantics. Shutdown must account for such in-flight work and must not abandon an open transaction.

The atomic boundary is a store operation, not `Memory.ingest()` or `Memory.process()`. Ingestion writes observations individually and advances a checkpoint only after a successful batch, including accepted policy rejections; a failed batch can leave earlier observations committed with the old checkpoint. A crash after observation commits but before checkpoint commit leads to replay. Checkpoint failure must not be described as rolling back the batch. Later SQLite reconciliation and learning implementations must preserve their own atomic and idempotent protocol guarantees, without implying whole-process atomicity.

## Schema and migration design

Use a separate ordered SQLite migration sequence and SQLite-owned version ledger, independent of PostgreSQL's migration numbers and `apply_migrations`. Package migration files in both wheel and source distribution and test installation artifacts. The initial migration creates the ledger, observations, revisions and checkpoints with the fields required by the existing models, including observation/revision encoding context, retention policy fields and historical source metadata. It does not pre-create speculative semantic or cognitive tables.

Use text IDs, explicit integer booleans, JSON text and UTC timestamps in a consistently round-trippable representation. Convert through existing models and context canonicalization; do not rely on SQLite JSON extensions. Preserve nullable fields and timezone-aware values. Constrain unique `(tenant_id, source_namespace, source_record_id)`, `(observation_id, revision_number)` and `(tenant_id, connector_id)` keys; enforce revision foreign keys and deletion behaviour. Index tenant/source and tenant/subject access paths. Every revision lookup joins through its tenant-scoped observation. Parameterize values and bound or chunk large ID/revision batches to respect SQLite's variable limits.

On open, acquire a write transaction before checking and applying migrations, so concurrent initializers cannot interleave. Apply each pending migration and record its version in the same transaction; failure rolls back both schema changes and ledger entry. Do not use `executescript` in a way that implicitly commits the managed transaction. Reopening an up-to-date database is a no-op. Reject an unknown newer version, missing/out-of-order ledger entries or an incompatible/non-Cogkura database clearly rather than overwriting it. Define a recognizable database identity and test it, including empty new files and unrelated populated files. A failed migration must leave the previous version usable. Do not modify the PostgreSQL migration system.

SQLite files are not encrypted by default. Document that content, metadata, revisions, checkpoints and WAL sidecars can contain sensitive data, and that host/device encryption and filesystem access controls are the application's responsibility. Do not promise secure erasure or encrypted backups.

## Observation and checkpoint semantics

Use the existing pipeline and `RetainedObservation`; the store persists retained values, never raw content in place of redacted or hash-only values. `REFERENCE_ONLY` remains unsupported by the current retention layer. Preserve full, redacted and hash-only content, hashes, metadata, encoding context and policy fields as supplied.

Identity is the tenant/source namespace/source record triple. Initial ingest creates a stable observation ID and revision 1. Subsequent ingest is unchanged only when source version, retained content hash, deletion state and canonical encoding context match; changes to metadata or other fields alone do not create a revision under the current contract. Otherwise create one successive revision with the existing `UPDATED`, `DELETED` or `RESTORED` transition; retain the original ID. Even an initially deleted input follows existing create behaviour. Keep historical source version, metadata, context and observation time for each revision. Current fields are updated as in existing stores; an unchanged ingest does not refresh them.

`get_by_source`, `list`, `get_many`, `get_revisions_many` and `clear` must scope by tenant. `list` honours subject and deleted filters. Exact historical lookup deduplicates requested identities, omits missing or cross-tenant revisions and never substitutes the latest revision; returned source identity, current revision and deletion state are current observation fields while revision metadata, source version and observed time are historical. Clearing a tenant removes its observations and revisions only, leaving other tenants and checkpoints intact. Preserve checkpoint JSON values and isolate the same connector ID across tenants; checkpoint writes replace the previous value.

## Verification and release gates

Build a shared contract suite using the existing in-memory and PostgreSQL stores and the new SQLite stores. PostgreSQL tests remain environment-gated locally but must run with a real PostgreSQL service in the release gate; SQLite tests must run without the postgres extra. Compare logical values and statuses, normalizing independently generated IDs and timestamps only where appropriate; separately assert identity stability within each backend. Do not silently redefine ordering: PostgreSQL orders observation lists by observed time and ID, while in-memory currently follows insertion order. Compare unordered results where the protocol does not promise order, and record any ordering-sensitive caller or backend discrepancy for explicit resolution.

| Area | Required cases |
| --- | --- |
| Observation lifecycle | Create, repeated unchanged, update, delete, restore, initial deletion; context-only change and metadata-only unchanged; stable ID and contiguous revisions. |
| Reads and isolation | Subject/deletion filters, empty and large batches, duplicate/missing revision requests, exact old revisions after updates and deletion, same source across tenants, cross-tenant IDs, tenant clear. |
| Retention and serialization | Full, redacted, hash-only, policy fields, nested JSON, context, nulls, Unicode and timezone-aware timestamps; no original content in retained records where excluded. |
| Checkpoints and ingestion | Set/get/replacement and tenant isolation; successful batch, rejected records, failed batch with no checkpoint advance, replay after partial batch and checkpoint write failure. |
| Persistence and atomicity | Close/reopen with stable IDs and full history; inject failure between current-row and revision writes and verify rollback; checkpoint failure preserves previous value; foreign keys enabled. |
| Migration and packaging | Fresh database, repeat initialization, concurrent initializers, rollback on failed migration, unknown/incompatible versions and unrelated database; packaged SQL in installed wheel and sdist. |
| Runtime behaviour | Event loop remains responsive during a blocked database operation; concurrent writes and readers across owners, busy timeout/error, cancellation followed by usable connection, close with queued work, reopen. |

Run the repository's Core lint, format, typing and test gates on Python 3.11–3.13, plus the real PostgreSQL parity suite. Capture any pre-existing backend differences explicitly; intended behavioural changes require design and regression assessment rather than being hidden in the SQLite implementation. Performance, abrupt-process recovery, backup/restore and full cognitive backend parity receive their broader hardening in 0.19.4–0.19.5, but basic transactional rollback and restart durability are required now.

0.19.0 is accepted when the new dependency-free stores meet this matrix, observations and checkpoints survive reopening, historical revision and ingestion behaviour match the established contracts, and no cognitive algorithm or default has changed. Documentation and findings must distinguish verified results from the proposed later milestones.
