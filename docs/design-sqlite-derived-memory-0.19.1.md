# SQLite derived memory (0.19.1) — design proposal

## Scope

Persist episodes and semantic memories in the existing Cogkura-owned SQLite file through `EpisodeStore` and `SemanticMemoryStore`. Implement the optional `SemanticInspectionStore` capability when its bounded and historical semantics can be preserved. Continue to use the existing algorithms and `Memory` dependency injection; activation, dynamics, learning, inhibition, entity relationships and working-memory snapshots are outside this milestone. This is not the complete persistent cognitive-state backend planned for 0.19.2.

The 0.19.0 observation/checkpoint implementation is the base. The SQLite, PostgreSQL and in-memory observation contract and the full Python 3.11–3.13 Core gates have now been exercised as part of this work.

## Data and transaction model

Add a new ordered SQLite migration; upgrades must retain all 0.19.0 observations, exact revisions and checkpoints. Use the existing worker and connection owner. Persist stable tenant-scoped episode and semantic identities, episode encoding context, entity links and observation evidence including revision and order. Source evidence must continue to resolve through the exact observation revision after reopen; never replace missing history with current values.

Persist semantic claims, all revisions and their complete snapshots, derivation relations to episodes, observation evidence, entities and revision-to-revision relations, including effective times. Keep historical revisions even when current claims change or are withdrawn. Model the data explicitly, not as a latest-fact key-value cache. Tenant scope must apply to every lookup, join and reference, including overlapping keys across tenants. Migration and storage constraints must prevent cross-tenant associations and preserve the same deletion behaviour as the existing protocols.

Each upsert or deactivation is a complete transaction. `apply_reconciliation` takes the write reservation before reading current state and applies the entire plan — current claims, revisions, snapshots, derivations, evidence and relationships — atomically. A failure at any point rolls back the complete plan. Retrying an unchanged plan must not create duplicate identities or relations, and its status counts must match established backends. Inspection, historical reads and batched evidence reads must see a consistent snapshot. Do not imply that all of `Memory.process()` is one transaction.

## Contract verification

Use shared SQLite, in-memory and PostgreSQL cases for upsert, unchanged fingerprints, encoding-context changes, inactive records, subject/tenant filters, limits, evidence and entity ordering, and reconciliation results. Compare stable identities within a backend and logical data across backends; document any existing ordering or constraint differences rather than silently changing a frozen contract.

Include an end-to-end restart test that observes source records, encodes episodes, consolidates claims, closes and reopens the file, and verifies identity, evidence, source provenance and semantic history. Exercise correction, withdrawal, readmission, effective-time boundaries, historical snapshots, bounded inspection and idempotent reconciliation. Inject failures between reconciliation writes and verify no partial state remains. Test overlapping tenant keys, unavailable source revisions, large batches, concurrent readers/writers and upgrade from a populated 0.19.0 database. Run existing source authority, provenance and correction regressions against SQLite where their fixtures permit, plus the Core Python 3.11–3.13 and PostgreSQL gates.

Acceptance requires those tests, preserved cognitive behaviour and documented supported limitations. Do not claim durable cognitive state or a full embedded runtime at this milestone.

## Implementation findings

The SQLite stores, migration, correction/withdrawal restart tests and optional slot inspection are implemented. The source-provenance freeze fixture exercises all three backends. Full Core suites with PostgreSQL 16 pass on Python 3.11, 3.12 and 3.13: 980 tests per runtime with no skips. Ruff, format and mypy pass on each runtime. Wheel and source distributions contain the three SQLite migrations. These are development results, not publication or an independent Bench validation.

Semantic operations other than slot inspection currently load the tenant's semantic inventory into memory before applying existing in-memory storage semantics inside the SQLite transaction. Slot inspection selects bounded inventory and relation prefixes in SQL. The broader approach preserves full revision state but has unbounded per-tenant read and memory costs; 0.19.4 must measure and harden it before the embedded architecture freeze. Activation, learning, forgetting and inhibition are not persisted by this milestone.
