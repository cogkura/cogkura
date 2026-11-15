# Embedded reliability and performance hardening (0.19.4) — design

This milestone validates and hardens the SQLite embedded deployment without changing cognitive algorithms, rankings or defaults. It retains the store-operation transaction boundaries: observation/revision writes, semantic reconciliation and learning plans are atomic, while the multi-stage `Memory.process()` and an ingestion batch are not one transaction.

The verification matrix includes abrupt process exit before and after commit, failed writes and rollback, WAL recovery, concurrent readers and serialized writers, lock timeout and retry, upgrades from every earlier SQLite schema, SQLite online backup and restore, tenant isolation, provenance and semantic revision correctness, and storage and retrieval workloads. Integrity and foreign-key checks run on recovered and backed-up databases. Backup tests use SQLite's backup API, not a raw file copy of a live WAL database. File ownership, permissions and the absence of built-in encryption are documented as host responsibilities.

A reproducible observational benchmark records workload shape, environment, ingest and retrieval latency, process memory and database/WAL growth. It preserves logical output checks alongside timing and does not declare an unmeasured throughput guarantee. Large tenant-wide semantic inventory materialization and cognitive scans are known scaling risks: measure them, improve them only with behavioral parity, and record remaining limits instead of silently resetting comparisons. Python 3.11–3.13 Core and PostgreSQL gates remain required.

0.19.4 is complete only when supported-workload verification reveals no unresolved data loss, corruption, migration or semantic-consistency failure. Independent Bench validation and architecture freeze are 0.19.5, not implied by this milestone.

## Findings

Crash recovery, SQLite online backup, schema upgrades, contention, tenant scope, security boundaries and the recorded synthetic workloads are described in [the findings](findings/0.19.4-embedded-reliability.md). The semantic inventory loader now batches its child queries while preserving the existing model and logical output. Measured statement counts fell, while the recorded traced recall timing did not demonstrate a latency improvement. SQLite still materializes the semantic inventory; the observed workload is not a general capacity guarantee.
