# Review: source inspection ordering in Core 0.18.3

**Status:** Contract clarification approved by the user on 14 November 2026;
the slot-only fix is implemented. The measurements below record the original
review; subsequent validation is recorded in the
[Core findings](findings/0.18.3-source-aware-metamemory.md).

**Reviewed:** Core `9ab2241ad600ed766b9c6487440e7495fe68254c`, Bench
`0bc6487` and its executed PostgreSQL 14.18 validation. Date: 14 November 2026.

## Conclusion

Retain strict logical source-attribution parity for `inspect_semantic_slot()`.
Give that new API a canonical evidence order based on immutable source identity
and numeric observation revision. Preserve existing prepared-context provenance
ordering and every cognitive/storage write path.

The approved design explicitly fixes context ordering, but does not specify a
slot-evidence ordering key. Its stable-prefix and backend-parity requirements
make the current dependency on backend-generated identities inappropriate for
portable slot reports. Documenting arbitrary backend-local prefixes would weaken
that acceptance criterion. The recommendation is a narrowly scoped read-side fix.

## What the approved design says

| Surface | Existing contract | Review recommendation |
| --- | --- | --- |
| `resolve_provenance(context)` and `assess_source_memory(context)` | Same selected primaries/members and evidence order; sequence number, observation ID, revision. | Preserve exactly. These report an existing context under its compatibility contract. |
| Slot memory/revision populations | `(memory_key, revision_number, revision_key)`; relations by their endpoint keys and relation. | Preserve. These are logical semantic keys. |
| Evidence inside each inspected semantic revision | Not explicitly keyed; shared helper currently inherits context ordering. | Define canonical source identity/revision ordering before the global evidence cap. |
| Pass-local decision reports | Bounded records ordered by semantic/source identities, observing actual outcomes. | Preserve. Their collector already uses source namespace/record identity before its opaque-ID fallback. |

References: [context API contract](design-source-aware-metamemory-0.18.3.md#41-prepared-context-assessment),
[bounds and cost](design-source-aware-metamemory-0.18.3.md#bounds-and-cost), and
[non-interference](design-source-aware-metamemory-0.18.3.md#9-non-interference-guarantees).
The acceptance matrix requires stable prefixes and the same logical
reports/states/order/counts/replay across the built-in backends. Comparisons
exclude different physical UUID values; that does not exclude selecting a
different actual source record because of those UUIDs.

## Root cause, including an earlier incomplete diagnosis

The immediate path is:

1. `semantic._flatten_observation_evidence()` sorts observation UUID/revision
   pairs and assigns `sequence_number` from that order.
2. `provenance._memory_evidence()` orders semantic references by
   `(sequence_number, observation_id, observation_revision)`.
3. `source_assessment.inspect_slot_sources()` slices those references by the
   global `max_evidence`, then resolves only the selected references.

Consequently, **sequence numbers already encode UUID ordering**. The earlier
Bench explanation about equal-sequence UUID tie-breakers was incomplete. Merely
replacing the final UUID tie-breaker with a source name, while retaining sequence
as the primary key, does not fix this case.

Code: [semantic evidence construction](../src/cogkura/algorithms/semantic.py),
[shared provenance helper](../src/cogkura/provenance.py), and
[slot assessment](../src/cogkura/algorithms/source_assessment.py).

## Independent review probe

A temporary public-API probe ingested identical four-source, two-claim data in
fresh in-memory stores. Both claims have two ordinary supporting episodes and
overlapping validity. Only the seeded standard-library UUID allocator differed;
inputs, processing time and configuration were identical. No Core methods were
patched and no repository code was changed for the probe.

| UUID seed | Stored pipeline sequence 0 / 1 | Full pipeline evidence order | `max_evidence=1` source |
| --- | --- | --- | --- |
| 0 | `pipeline-0`, `pipeline-1` | `pipeline-0`, `pipeline-1` | `pipeline-0` |
| 1 | `pipeline-1`, `pipeline-0` | `pipeline-1`, `pipeline-0` | `pipeline-1` |
| 2 | `pipeline-1`, `pipeline-0` | `pipeline-1`, `pipeline-0` | `pipeline-1` |

All three reports are contested, truncated and revision-incomplete under the cap.
Both capped membership and uncapped evidence order depend on physical identity.
This reproduces the mechanism without requiring a database. Bench independently
observed `sm-conflict-startup-0` versus `sm-conflict-startup-1` between in-memory
and PostgreSQL stores. Its strict failed gate remains valid.

## Recommended normative contract

For equal tenant/subject scope, logical semantic populations, source histories,
availability, `as_of`, `valid_at`, and limits:

1. Keep semantic revisions and relations in their existing canonical order.
2. Within each revision, order evidence with an available stable source identity
   by `(source_namespace, source_record_id, observation_revision)`. Use ordinary
   case-sensitive string ordering and numeric observation revision; add no
   source-name normalization. Do not use stored semantic evidence sequence or
   physical UUID as an earlier ordering component.
3. Concatenate the revision populations in that existing revision order and
   apply **one global occurrence cap**. Do not balance the cap across sources or
   opposing claims. Preserve repeated evidence occurrences across revisions and
   all existing count definitions. Increasing the cap exposes a longer prefix.
4. This is display ordering only. It conveys no authority, trust, independence,
   certainty, chronology, recency or evidence-strength ranking. In particular,
   vendor `source_version` strings and contribution scores are not sort weights.
5. Use the exact referenced revision for revision-specific metadata. Immutable
   namespace/record identity from the owning observation may still identify a
   source when its exact revision is unavailable; this must retain the existing
   revision-unavailable status and incomplete coverage. It does not permit a
   latest-revision metadata substitution.
6. If neither lookup can establish a source identity, place those references in
   an explicit identity-unavailable suffix, using the existing opaque identity
   as a repeatable within-store fallback. Keep unavailable/incomplete diagnostics
   and counts. Do not promise a portable source-identity prefix for references
   whose logical source identity cannot be established. Raw UUID differences are
   not themselves a parity failure; known source selections remain comparable.
7. Historical visibility must be respected when constructing the displayed
   population and key. A later revision's metadata must not determine the order
   of earlier visible evidence. Preserve current unknown/indeterminate behavior
   where historical attribution is unsupported; do not broaden historical claims.
8. No persisted sequence, fingerprint, semantic key, status, cognitive decision,
   confidence formula, retrieval order, context provenance order, rendered prose
   or counter changes. No new public model, mandatory protocol or migration.

The source/revision ordering rule was explicitly approved before implementation.
Identity-unavailable references cannot have a universally portable logical order
without additional retained identity data; inventing that data is outside 0.18.3.

## Implementation boundary and cost

Implement the ordering inside the slot inspection read path, after obtaining
identity/visibility information and before truncation. Share existing resolution
primitives; do not change the default order in `_memory_evidence()`, renumber
stored evidence, or retune semantic consolidation. Sorting the already-truncated
payload cannot repair which source was selected.

Current lookup requests contain only evidence selected by the cap. Canonical
selection needs stable identities for the candidate population first. A fix may
therefore batch-read metadata for more references from the already returned slot
snapshots. Existing public batch APIs can supply those identities; no per-source
query or unbounded tenant scan is needed. Exact historical metadata lookup and
future-visibility checks must still follow the temporal contract.

This tradeoff must be measured: fixed query count does not imply fixed row count,
metadata bytes or latency. The approved design already defines `max_evidence`
as an output cap, not a hard database-row/byte budget. Retain bounded
revision/relation populations, deduplicated batches, no network fetches and no
writes. If a lightweight identity index/protocol or a hard read budget proves
necessary, report that separately rather than adding schema scope silently.

## Required regression and release checks after approval

- Recreate identical logical stores under several UUID allocations and both
  backends. Compare ordered source namespace/record/revision identities, not UUIDs.
- Include references whose stored sequence numbers differ between those stores;
  equal-sequence tie cases alone would miss the observed defect.
- Verify uncapped order and caps 1, 2, exact population size and overflow, including
  one early revision consuming the budget and repeated references across revisions.
- Cover missing locator, missing exact revision with stable owner identity,
  identity-unavailable fallback, mixed availability, currentness and historical
  boundaries. Assert no future metadata/identity leak or false completeness.
- Preserve existing context-provenance output byte/field equivalence and confirm
  no changes in stored evidence sequences, counters, semantic history or cognition.
- Measure source lookup batches, returned rows/payload and latency as references
  grow; preserve tenant/subject scoping and read-only behavior.
- Run Core Python 3.11–3.13 lint/format/typing/test gates with real PostgreSQL.
- Re-pin Bench to the validated Core commit, then rerun its existing strict gate,
  frozen per-case comparisons and diagnostics parity without weakening gold.

The original review executed source inspection and the three fresh-store probes.
It did not change Core or Bench. The subsequent approved implementation and
release-gate results are recorded separately in the linked findings.
