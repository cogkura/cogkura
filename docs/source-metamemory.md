# Source-aware metamemory (0.18.3)

Source diagnostics describe retained evidence and stored semantic state. They do
not determine whether an answer is true, rank source trust, establish source
independence, or decide whether an application should abstain. Existing retrieval
metamemory, confidence, and cognitive behavior remain unchanged.

## Prepared context

```python
context = await memory.prepare_context(query, tenant_id="tenant", as_of=clock)
sources = await memory.assess_source_memory(context, max_evidence=256)
```

This resolves the selected primaries and included chunk members once, without
another recall. `sources.provenance` preserves the `resolve_provenance()` contract.
The companion `quality` separates exact revision coverage from optional locator
coverage. `SOURCE_REF_MISSING` is a resolved revision without a locator; the older
provenance completeness fields retain their original, stricter meaning.

Counts describe evidence occurrences, including a reference repeated in a primary
and a member. Distinct sources count resolved `(source_namespace, source_record_id)`
pairs. Store lookups are deduplicated; repeated evidence is not independent support.

Semantic relation coverage counts only semantic evidence requiring a proven
`SEMANTIC_SUPPORT` or `SEMANTIC_CONTRADICTION` relation. A missing or ambiguous
semantic derivation relation is unknown. Direct episodic evidence is **not
applicable** to this measure: it contributes to `relation_not_applicable_count`,
not `relation_known_count`. `relation_coverage_complete` is `None` when no semantic
references exist, `False` when applicable semantic relations are missing or omitted by the cap,
and `True` when all required semantic relations are known. Omitting only
non-applicable episodic references does not reduce semantic relation coverage. Empty
reports have `has_evidence=False`; vacuous coverage is not evidence adequacy.

Currentness counts examined evidence and flags cap omissions as incomplete.
It compares cited evidence with the store view at inspection. It is not
external freshness, factual certainty, or a historical latest-pointer query.
Resolving a context prepared earlier can legitimately find its evidence is older
than the present source revision. Locators are inert data and are never fetched.

## Exact semantic slots

```python
from cogkura import SemanticSlotQuery, SourceInspectionLimits

slot = await memory.inspect_semantic_slot(
    SemanticSlotQuery("service", "deployment_policy", {"environment": "production"}),
    tenant_id="tenant",
    subject_id="service",
    as_of=clock,
    valid_at=factual_time,
    limits=SourceInspectionLimits(max_revisions=128, max_relations=256, max_evidence=256),
)
```

Use exact structured identifiers. Slot identity uses existing normalization and
hashing, including qualifiers, and excludes value/polarity/cardinality. Query
qualifiers accept finite scalar values; nested qualifier structures are not
supported by this inspection query. `subject_id` is the storage filter and is
separate from `subject_entity_id`. All reads require a tenant.

The states are:

| State | Meaning |
| --- | --- |
| `KNOWN` | Applicable stored active assertion(s), regardless of recall accessibility. |
| `CONTESTED` | Applicable stored contention/conflict. |
| `WITHDRAWN` | Explicit retraction and no applicable active assertion. |
| `HISTORICAL_ONLY` | Retained earlier/expired assertions without a current answer. |
| `UNKNOWN` | Complete inspected semantic inventory has no applicable assertion or visible history explaining absence. |
| `INDETERMINATE` | Bounds, unsupported storage, inconsistent occupants, or incomplete temporal/history data prevent classification. |

An empty recall, missing locator, source deletion, or forgetting is not withdrawal.
A negative assertion can be known; cardinality-many coexistence and temporal
succession do not by themselves constitute conflict. Missing evidence alone does
not erase a proven stored status. Unknown is relative to retained semantic memory,
not all observations or external reality. Unencoded and unpromoted declarations
are not discovered by scanning source prose.

`conflict.cross_source_disagreement` requires existing incompatibility and opposing
resolved source identities; it is `None` when attribution cannot decide it. Source
independence remains `UNKNOWN`. Retrieval competition is not factual contradiction.
`decision_history` is always `UNAVAILABLE`; there is no durable decision ledger.

## Temporal limits

Intervals are half-open. The supported 0.18.2 correction/withdrawal knowledge and
effective-time boundaries are retained: future asserting observations do not
expose successor/retraction identities or links; known future-effective edits can
be reported as pending. Before the effective boundary the predecessor remains
applicable. Superseded history is not an ongoing disagreement with its successor.

An explicitly supplied `as_of` masks present-store source currency comparisons:
per-source `is_latest_known_revision` is `None`. Omit `as_of` for present-store
comparison. Historical revision metadata is still resolved from the exact revision.
Unsupported ordinary history and legacy empty snapshots remain incomplete or
indeterminate; `projected_status=None` avoids exposing an unprovable present-day
status in a historical view. These APIs do not provide complete bitemporal history
or reconstruct every previous support/confidence value.

## Decisions from the actual processing pass

```python
from cogkura import Memory, SemanticDiagnosticsConfig

memory = Memory(semantic_diagnostics_config=SemanticDiagnosticsConfig(enabled=True, max_items=128))
result = await memory.process(tenant_id="tenant", as_of=clock)
report = result.semantics.decisions
```

Reporting defaults off (`decisions=None`). It adds no revision reads and does not
enable authority or corrections. When enabled, it observes the existing extraction,
authority qualification, deferral, reconciliation, and edit paths. Existing counters
and durable outcomes stay unchanged. Disabled mechanisms are marked unevaluated;
custom algorithms without supported hooks mark diagnostics incomplete.

`ADMIT`, `REPLACE`, and `WITHDRAW` records expose actual outcomes and typed reasons,
including unknown targets, competing edits, malformed declarations, and source
qualification failure. Malformed records can lack slot/target identity. An invalid
authority route does not prevent an otherwise valid fact from ordinary recurrence;
the diagnostics do not introduce a new rejection rule. Existing counter units can
differ from deduplicated report-item units. Reports are bounded and explicitly
truncated, with stable ordering using semantic and source identities.

A successful report is returned only after storage succeeds. It is local to that
processing pass; applications must save it if they need an audit. A later report
can repeat a deferral and does not constitute a persistent unresolved request.
In particular, the existing withdrawal qualifier can defer the original target
again on replay even while the durable semantic state stays withdrawn. The report
must reflect that branch rather than invent `ALREADY_APPLIED`.

## Storage, bounds, and limitations

Both built-in semantic stores implement optional `SemanticInspectionStore`.
Custom stores keep the original `SemanticMemoryStore` protocol; inject a separate
`semantic_inspection_store` if needed. Without the capability the new slot API
returns indeterminate/unavailable, never a false unknown. No migration is added.

PostgreSQL slot populations use a short read-only repeatable-read transaction.
Source/episode batches can follow through separate stores, so the full assessment
is not an atomic cross-store snapshot. Missing evidence reduces completeness;
current metadata never substitutes for a missing revision. The revision/relation
and evidence limits bound returned populations, not JSON bytes or all underlying
derivation rows. Large snapshots and the in-memory inventory scan remain costs to
measure for each workload.

The known **0.18.2 episodic-prose limitation remains**: a later re-adoption episode
can render its assertion even while the semantic claim stays withdrawn. These
reports distinguish semantic state from recalled prose; they do not redesign
retrieval or add suppression rules.

Applications own uncertainty and abstention policy. Complete attribution,
application authority, and strong retrieval never imply factual certainty.
Independent Bench validation is a separate release gate, not established by Core
unit or PostgreSQL parity tests. See the [design](design-source-aware-metamemory-0.18.3.md)
and [implementation findings](findings/0.18.3-source-aware-metamemory.md).
