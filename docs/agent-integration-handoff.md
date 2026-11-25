# CogKura Core Agent integration handoff

**Working draft for 0.20.3.** Do not pin a 0.20.3 release from this document
until an exact candidate has completed its independent review and the release
owner has approved and published it. The current released baseline is Core
0.20.2, tag `v0.20.2`, commit
`fb72911ce58173bf3669181d0e2c25eeefeaa58d`. The final 0.20.3 handoff must record
its own tag and verified artifact identities.

## Runtime and identities

Use `EmbeddedMemory.open(path)` for a local SQLite runtime and close its owner
on shutdown. Own the database and its backups in the Agent's protected local
data directory. SQLite WAL is not supported on network filesystems; encryption,
secure deletion and operating-system isolation are not supplied automatically.
PostgreSQL is optional and belongs to deployments that need shared managed
persistence, not to the local embedded setup.

The application supplies a stable tenant identity and resolves repository or
project metadata into a stable `subject_id`. Core does not infer subject
identity from a project name in a query. Keep identities consistent across
observations, processing, retrieval and restart.

## Observation and processing

Send normalized `ObservationInput` values with stable source identities and
source references. `observe()` stores observations; explicitly call
`Memory.process(tenant_id=..., subject_id=...)` to encode and consolidate the
known changed subject. Maintain Agent-owned dirty-subject tracking and coalesce
updates where appropriate. Tenant-wide processing is an explicit maintenance
operation; repeated broad processing can amplify retained history.

Do not assume that a single-subject process covers shared evidence. In the
validated two-project example, one supporting episode from each project only
consolidates the shared claim when both are processed together. Identify and
schedule any broader dependencies explicitly. `process()` is not one atomic
transaction across the entire cognitive pipeline; callers must handle errors,
write ordering, recovery and retry according to the individual API contracts.

## Retrieval, answerability and use

Pass the resolved subject for project-specific `prepare_context()`, `recall()`
and `inspect_recall()` calls. Tenant-wide requests remain available when the
question is broad. Core preserves its cognitive ranking; the known unscoped
fixture puts project incident and migration evidence at ranks 26 and 27.

Check `context.assessment.flags` before treating retrieved text as evidence.
`MISSING_KNOWLEDGE` remains an answerability warning even when `context.render()`
is non-empty. The Agent must warn, clarify or use an appropriate fallback rather
than invent a missing fact. Use `resolve_provenance()` when source identity is
needed; source locators are application metadata, not automatic truth claims.

Preparation is not consumption. `record_context_use(context)` records every
selected result. If only part of the context was consumed, pass only the used
`RecallResult` objects to `record_access()` instead. Apply explicit feedback
through `learn()` only when the application has an outcome to record. Agent
warning logic, selective-use policy, session reuse and MCP remain Agent work.

## History, operations and performance

Historical reconstruction depends on retained, compatible snapshots. Missing or
legacy history cannot always be recovered, and arbitrary concurrent writes do not
provide a coherent cross-store historical read. Handle
`HistoricalStateUnavailableError` as an unavailable historical state, not as
proof that a fact never existed. Follow the [SQLite backup and recovery
instructions](sqlite-storage.md) and test restore before relying on a backup.

The owner accepted the measured 50-project × 250-observation initial-Agent
fixture and deferred historical storage compaction. The 100-correction probe
increased that fixture by approximately 28.93 MB; its growth pattern and limits
are in [history findings](findings/0.20.3-history-amplification.md). This is a
specific synthetic workload, not a universal growth rate. The reference
`prepare_context()` target is warm p95 below 2.5 seconds; the measured fixture,
configuration and results are in [scaling findings](findings/0.20.3-scoped-read-scaling.md).
Neither timing nor storage results are production guarantees. Establish
application-specific monitoring and retention practices.
