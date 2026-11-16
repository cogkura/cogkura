# Embedded runtime stabilisation and Python compatibility (0.19.6)

Status: implementation in progress; release acceptance is not yet established.
Baseline: Core v0.19.5, release commit `13ad7336de1e8be4eba95fbd673ccb448c612d03`; validated implementation `f289035351e3c101fd932f4d6d9e9cd56583ace0`. The next milestone is the 0.20 agent integration contract.

## Objective and boundaries

Make embedded operation reproducible and operationally understood, without new cognitive capabilities, changed algorithms, mandatory storage protocols, migrations, or a new Memory API. Preserve authority, corrections, provenance, tenant isolation, SQLite transactions, transient working memory, and the 0.19.5 runtime guarantees. `Memory.process()` is not a single transaction, and cancellation does not establish whether an in-flight write committed.

## Plan and release gates

1. **Baseline and Bench**: commit the independent three-backend harness and prior raw and reviewed findings; validate the public 0.19.5 commit with a clean installed distribution. Record Python, SQLite, environment and exact installed identity. Preserve historical gold and do not transfer the candidate-specific `f289035` approvals to another commit.
2. **Python**: test CPython 3.11–3.15, dependency-free and PostgreSQL extras, with SQLite's version guard enabled. Run Linux, macOS and Windows across all five versions and a separate five-version PostgreSQL integration matrix. Record SQLite source ID, filesystem, journal, foreign-key, synchronous and fullfsync settings. Compatibility claims require observed passing results, not merely a configured matrix.
3. **Parity and discrepancies**: retain the unmodified strict frozen output. Investigate all 3.11 historical differences per field and fingerprint, including repeatability and old-Core controls; no blanket tolerance or gold changes. Compare in-memory, SQLite, reopened SQLite, and PostgreSQL across representative observation, memory, cognitive-state and source scenarios, including a fresh-process SQLite restart. Diagnose support ordering without changing public ordering contracts. Any approval must name the exact candidate, baseline digest and fingerprints.
4. **Performance and reliability**: measure uninstrumented cold and warm latency with separate diagnostic runs, realistic revisions and withdrawals, database/WAL size, RSS and CPU. Target 100/500, 1,000/5,000 and, if feasible, 10,000/50,000 observations/revisions. Report resource and time limits instead of extrapolating. Recheck rollback, migrations, online backup, termination recovery, lock errors and tenant isolation. No physical power-cut experiments are required.
5. **Release**: build wheel and sdist from an identified candidate, install into clean environments, verify module/distribution identity, migrations, dependency-free import and restart, and run Bench against that artifact. Compare the tagged commit with the validated candidate. Record raw evidence and limitations. Tag and publish only with separate user authorisation.

## Acceptance

All required Core and Bench gates must pass on Python 3.11–3.15, including PostgreSQL-enabled tests and the 15 portable OS/runtime combinations. Every frozen or cross-backend difference must be explained and, where appropriate, explicitly approved for the exact candidate; raw comparisons remain available. Fresh-process restart, representative performance measurements, exact artifact verification and documented operational limits are mandatory. No unexplained correctness, data-loss or semantic-consistency failure is acceptable.

## Exclusions

No working-memory persistence, cloud sync, cross-backend import, MCP/IDE/session integration, free-threaded or alternate interpreters, Python 3.16, large-scale retrieval redesign, or new cognitive algorithm. No implied power-loss protection or encrypted SQLite files; network filesystems remain outside the supported WAL model. Proposed optimisations require separate review and frozen-output validation.
