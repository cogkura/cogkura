# Review: SQLite supporting-evidence order in the 0.19.5 candidate

Core source candidate: `fa9c993d82fd2dda74a9b18f2b80c3b857a26fc7`. This review is limited to the listed independent Bench cases; it is not an approval for other differences or future candidates.

The [controlled comparison](findings/data/bench-0.19.5-embedded-controlled.json) retains exact fingerprints for the in-memory and SQLite results, including SQLite reopened after each preparation. Three query signatures differ:

| Workload / query | In-memory supporting events | SQLite supporting events |
| --- | --- | --- |
| `authoritative_corrections_v1/ac-q-c03` | `ac-c03-a`, `ac-c03-b` | `ac-c03-b`, `ac-c03-a` |
| `source_aware_metamemory_v1/sm-contested` | `sm-conflict-pipeline-1`, `sm-conflict-pipeline-0` | `sm-conflict-pipeline-0`, `sm-conflict-pipeline-1` |
| `source_aware_metamemory_v1/sm-evidence-cap` | `sm-conflict-pipeline-1`, `sm-conflict-pipeline-0` | `sm-conflict-pipeline-0`, `sm-conflict-pipeline-1` |

The [correction details](findings/data/bench-0.19.5-correction-detail.json) and [source details](findings/data/bench-0.19.5-source-detail.json) retain complete query signatures. The difference is the ordering of `supporting_event_ids` in the authoritative-correction observation; membership is unchanged. A diagnostic comparison that sorts only those lists finds no other query-signature difference in the two cases, while the original strict comparison continues to fail. Aggregate metrics, use signatures, and source-assessment observations also match in the controlled comparison. Mini and persistent inhibition match exactly.

Core's existing semantic consolidation orders flattened supporting evidence by observation UUID. SQLite additionally allocates UUIDs for revision rows, whereas the in-memory observation store does not. Consequently, even starting with the same seeded UUID generator, later observation IDs and their evidence order differ. For example, the correction detail records `ac-c03-a` as `9655c59e-9fef-4236-adb9-7200128ae301` in memory and `b71cdf5c-faa5-4f2b-88b1-f2c3eb658cc2` in SQLite; `ac-c03-b` is `b71cdf5c-faa5-4f2b-88b1-f2c3eb658cc2` in memory and `692648c7-e871-4f34-8368-387eb9f31909` in SQLite. This explains why evidence ordered by opaque IDs can appear in a different order; no cognitive algorithm was changed. The source-inspection contract independently orders known evidence by source identity, and its compared observations match here.

On 15 November 2026 the user explicitly accepted these three order-only differences for this exact candidate. The reviewed comparison pins both memory and SQLite fingerprints for each query and fails on any other difference. The original gold, raw fingerprints, score calculations and other queries remain unchanged. This does not claim that UUID-dependent ordering can never affect another workload or ranking. All other 0.19.5 gates still apply.
