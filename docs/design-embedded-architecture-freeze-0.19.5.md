# Embedded memory architecture freeze (0.19.5)

## Purpose and boundary

This candidate verifies the 0.19 embedded storage architecture before the 0.20 integration contract. It introduces no significant capability or cognitive algorithm, ranking, or default change. A passing Core suite alone does not freeze the architecture: independent CogKuraBench comparisons and the evidence below are required. Release publication and tags remain separate.

The 0.19.5 gate requires ordinary restart persistence and the existing store-operation transaction guarantees. It does not require physical power-cut experiments or specialized SQLite I/O fault injection. Working memory remains transient; `Memory.process()` is not one atomic transaction. SQLite requires a supported local filesystem and a fixed SQLite version, and macOS `fullfsync` has a measured write-cost tradeoff. See [SQLite storage](sqlite-storage.md) and the [0.19.4 findings](findings/0.19.4-embedded-reliability.md).

## Verification matrix

| Contract | Evidence required |
| --- | --- |
| Storage coverage | Inventory all protocols in `storage/base.py`, including optional `SemanticInspectionStore`, and verify SQLite implementations and the embedded bundle. |
| Logical behavior | Exercise the same observations, derived memories, provenance, temporal corrections, and cognitive state across in-memory, SQLite and PostgreSQL; distinguish logical identity from backend-generated identifiers. |
| Restart | Close and reopen the embedded runtime and check observations and revisions, episodes, semantic history, checkpoints, cognitive state and retrieval with the same clock and configuration. Working memory remains transient. |
| Existing invariants | Retain authority, correction, provenance, retention, tenant scope, atomic reconciliation and idempotent learning tests. Record any actual differences rather than adjusting expectations. |
| Runtime and limits | Check supported migrations, contention, cleanup and rollback; preserve existing workload measurements, ranked outputs and operational limitations. |
| Core gates | Ruff, format, mypy and full PostgreSQL-enabled tests on Python 3.11–3.13; cross-platform CI results are recorded separately with any skips. |
| Independent Bench | Run the unchanged frozen comparison, source diagnostics and PostgreSQL parity against an identified Core candidate. Add SQLite and restart comparison using public Core APIs, with the original fixtures and comparisons preserved. |

## Benchmark review

The previous 0.18.4 correction approval applies only to its exact version, commit and before/after fingerprints. It is not automatically transferred to this candidate. Record the raw comparison, candidate source identity, environment and all differences. Any intended deviation, including a previously approved change carried into a new candidate, needs explicit review and approval under the existing Bench rules; do not modify frozen gold, replace the baseline or silently broaden the exception.

Core currently retains 0.18.4 package metadata during development. Reports must identify the actual source commit and loaded module as well as the declared version; metadata alone must not be used to identify this candidate as the released 0.18.4 distribution. A full architecture freeze is pending until all required gates pass and any deviations have been explicitly resolved.
