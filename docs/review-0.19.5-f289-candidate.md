# Review: 0.19.5 candidate after PostgreSQL cleanup fix

Candidate `f289035351e3c101fd932f4d6d9e9cd56583ace0` adds only the tenant-scoped PostgreSQL learning cleanup fix and its regression test to the earlier candidate `fa9c993d82fd2dda74a9b18f2b80c3b857a26fc7`. It has been pushed to Core main. On 15 November 2026 the user explicitly approved the frozen correction and the three support-order differences for this exact new candidate, including the PostgreSQL outcomes. The earlier approvals applied only to `fa9c993`.

## Frozen comparison

The [new raw frozen report](findings/data/bench-0.19.5-f289-frozen-raw.json) has exactly the same 393 query fingerprints, use signatures and metric fingerprints as the [earlier raw report](findings/data/bench-0.19.5-raw-validation.json). As before, the sole difference from the original 0.18.2 gold is `authoritative_corrections_v1/cogkura-correction/ac-q-c20`, with the same before/after hashes reviewed for 0.18.4 and `fa9c993`. PostgreSQL source-diagnostic parity passes and no paired diagnostic differences were reported. The report loaded this candidate's source checkout; its displayed installed distribution metadata describes a different package and must not be mistaken for the loaded source.

## Three-backend comparison

The [raw three-backend report](findings/data/bench-0.19.5-f289-threeway-raw.json) exercises four workloads using complete in-memory, SQLite and PostgreSQL bundles, and repeats SQLite after each preparation is followed by a close and reopen. The SQLite and in-memory query fingerprints are identical to those for `fa9c993`. Mini and persistent inhibition match across all backends. Three queries differ only in the order of the same supporting-event IDs, with identical query fingerprints for SQLite, reopened SQLite and PostgreSQL:

- `authoritative_corrections_v1/ac-q-c03`: `ac-c03-a`, `ac-c03-b` versus `ac-c03-b`, `ac-c03-a`.
- `source_aware_metamemory_v1/sm-contested`: `sm-conflict-pipeline-1`, `sm-conflict-pipeline-0` versus `sm-conflict-pipeline-0`, `sm-conflict-pipeline-1`.
- `source_aware_metamemory_v1/sm-evidence-cap`: the same two event IDs and ordering difference as `sm-contested`.

All other compared query fields match after considering only those lists as unordered; use signatures, aggregate metrics and source-assessment observations match. Original ordered fingerprints, full query details and raw failing comparisons are retained. The [earlier review](review-0.19.5-support-evidence-order.md) explains the UUID allocation and ordering mechanism. Its approval covered only SQLite and the earlier candidate; the user separately approved these exact outcomes for this candidate, including PostgreSQL.

These results cover four representative workloads and do not prove parity for every possible workload or fresh-process Bench adapter reconstruction. The [hosted CI matrix](https://github.com/cogkura/cogkura/actions/runs/43768559479), [exact Git-installed strict gate](findings/data/bench-0.19.5-f289-frozen-approved.json) and [reviewed Git-installed three-backend comparison](findings/data/bench-0.19.5-f289-threeway-installed-reviewed.json) subsequently passed with the separately approved exceptions; the original raw comparisons remain available.
