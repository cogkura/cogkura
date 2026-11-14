# CogKura Core 0.18.3 — Source-aware metamemory and conflict observability

**Status:** D1–D6 approved as recommended on 14 November 2026; Phase 2 authorised.

The approved design below is retained as the engineering contract. Actual API details
and clarifications are documented in [source-aware metamemory](source-metamemory.md).

**Reviewed baseline:** Core `1a8bc04f7832c77962a6c2e99b2e8dca4dc19958` (`0.18.2`).

**Related validation:** CogKuraBench `ce29fa677a2ca4c861b839c18e74e7079861e29c` (`0.3.10`).

**Date:** 12 November 2026.

**Branch:** `feat/source-aware-metamemory-0.18.3` for design and subsequent approved implementation.

## 1. Purpose and release boundary

Expose what Core can establish about evidence, semantic state, and unresolved
decisions, without making its memory mechanisms decide what an application should
believe. Authority is permission for a supported operation; it is not factual
certainty. Accessibility is not evidence quality. Different source records are
not necessarily independent sources.

The recommended release has three additive surfaces:

1. Source-aware evidence assessment of an existing prepared context.
2. Exact semantic-slot inspection that can find withdrawal, contention, and history
   even when retrieval returns nothing.
3. Optional typed admission/correction/withdrawal decision reports from the actual
   consolidation pass, including reasons for deferral.

The third surface is deliberately **pass-local**. There is no existing durable
ledger of deferred decisions. This design does not pretend that an empty query
result or a past counter proves that no unresolved declaration exists. Durable
unresolved-decision enumeration would require a separately approved storage
extension (review decision D2).

This document records the Phase 1 review and the approved implementation contract.
The user authorised Core implementation, incremental commits on this branch, and
one PR after Core validation. Bench, Demo, and Agent remain outside this task.
Independent Bench validation remains a release gate and will be authorised separately.

## 2. Architecture review: reuse and gaps

| Existing capability | Actual implementation and limitation | 0.18.3 extension |
| --- | --- | --- |
| Retrieval metamemory | `algorithms/metamemory.py`: `DeterministicMemoryMonitor` assesses the bounded threshold-qualified pool. `MemoryAssessment` includes conflict, confidence, context, and interference diagnostics. | Preserve the monitor, its protocol, signals, and flags. Add a separate source report. |
| Evidence confidence | `MetamemorySignals.evidence_confidence` is retrieval-weighted stored confidence. The semantic confidence formula uses recurrence, agreement, and namespace diversity. | Describe evidence resolution and relation coverage separately; do not reweight confidence. |
| Provenance diversity | Existing metamemory counts distinct observation IDs; consolidation's diversity term counts namespaces. | New reports count `(source_namespace, source_record_id)` pairs and label the population. Neither existing calculation changes. |
| Exact provenance | `provenance.py`: `SourceProvenanceResolver`, `SourceEvidence`, `MemoryProvenance`, `ContextProvenance`; exact revision reads via `ObservationStore.get_revisions_many()`. | Reuse the resolver and its allow-listed attribution, batching, evidence order, and context-member representation. |
| Resolution completeness | `SOURCE_REF_MISSING` retains historical identity/version/time, but existing `is_complete` and resolved counts require status `RESOLVED`. Empty evidence can be vacuously complete. | Add independent revision, locator, relation, and evidence-presence measures; retain old fields unchanged. |
| Source revision currentness | `SourceEvidence.is_latest_known_revision` compares cited revision with the owning observation's present `current_revision`. `StoredObservationRevision.is_deleted` also comes from that present row. | Explicitly label comparison with the store view at inspection. Do not present either field as historical source state. |
| Fact-scoped authority | `algorithms/semantic.py`: exact-revision qualification, opt-in single-episode promotion, and competing-authority deferral. Promotion metadata is returned by provenance. | Explain admission basis separately from confidence and support count. |
| Semantic conflicts | `algorithms/reconsolidation.py`: deterministic `CONFLICTS`, `SUPERSEDES`, `COEXISTS`, `REINFORCES`; status `CONTESTED`. Stored relations include conflicts, supersession, and retraction. | Inspect the existing statuses/relations and their evidence; do not introduce another reconciler. |
| Explicit edits | `algorithms/corrections.py`: `qualify_explicit_corrections()` and `apply_explicit_semantic_edits()`; `memory.py` orchestrates them. | Observe the actual qualification branches and final committed outcome, with typed reasons. |
| Deferred edits | `SemanticConsolidationResult` has `authority_deferred_conflict`, `corrections_deferred`, and `corrections_invalid`; no per-request persisted outcome. | Add bounded optional pass-local records. Never reconstruct a past decision by running consolidation in an assessment. |
| History and knowledge boundary | Versioned revision snapshots from migration `011`; correction projection and stale-episode filtering in `corrections.py`. Legacy empty snapshots fall back to a current-row overlay. | Reuse supported projection rules; report incomplete historical attribution instead of certifying the fallback. |
| Storage | `storage/base.py` has tenant-scoped list/revision and batch evidence APIs, but no bounded slot-and-relation read. In-memory stores relations privately; PostgreSQL persists them. | Add an optional read capability in the existing storage package. Do not read private store dictionaries from the facade. |

Primary review sources: [metamemory](metamemory.md),
[architecture](architecture.md), [source provenance](design-source-provenance-0.18.0.md),
[source authority](design-source-authority-0.18.1.md),
[corrections](design-authoritative-corrections-0.18.2.md), and
[0.18.2 findings](findings/0.18.2-authoritative-corrections.md).

### Verified release context and concerns

The local Core and Bench checkouts match the supplied commits. Bench's checked-in
`docs/findings/authoritative-corrections-0.3.10.md` records the supplied 35-event,
26-checkpoint results, all declared accuracy rates at 1.000, zero declared
violations, unchanged frozen results, and `NO CORE CHANGE INDICATED`.
These are prior measurements, not a new Bench run for this design.

At the Phase 1 review on 12 November, **PostgreSQL adopt/replace/withdraw/replay
parity was NOT EXECUTED** and `COGKURA_POSTGRES_MEMORY_URL` was unset. Existing
Core parity tests were in `tests/test_postgres_integration.py`; source review was
not execution. This Core gap was subsequently executed and closed during Phase 2;
see the [0.18.3 findings](findings/0.18.3-source-aware-metamemory.md). The separate
Bench 0.3.10 measurement retains its original not-executed status.

Bench also explicitly documents that a later re-adoption episode's text can appear
in context even though the withdrawn semantic claim stays withdrawn. Its zero
stale-context violations cover the declared cases; they do not establish a
universal absence of such prose. Record and expose the semantic/episodic distinction;
do not expand 0.18.3 into a retrieval suppression fix.

## 3. Scope and exclusions

### Included

- Resolution of exact referenced evidence, missing evidence, optional locators,
  and known support/contradiction relations.
- Distinct-source disagreement supported by existing semantic conflict records
  or by an actual pass's competing-authority decisions.
- Separate representations of unknown, currently known, contested, withdrawn,
  and superseded historical knowledge.
- Existing admission basis, applied edits, pending effective boundaries, and
  pass-local deferral/invalidity reasons.
- Explicit unknown/unavailable/truncated values, with evidence references that
  let an application make its own abstention policy.
- Deterministic, read-only context and slot reports on in-memory and PostgreSQL
  stores; optional decision reporting on the existing write path.

### Excluded

- Inferred source trust, source ranking, calibrated truth probabilities, an
  overall certainty score, or a Core `should_answer`/`should_abstain` decision.
- Inferring independence from namespaces, record IDs, prose, URLs, confidence,
  repetition, actors, or authority. No new independence declarations in this
  proposed release; independence remains unknown (D3).
- New semantic extraction, scanning arbitrary source prose for disagreement,
  or treating every different object as a contradiction. Unencoded/unpromoted
  claims are outside slot inventory except where an actual pass reports them.
- Automatic conflict resolution, source deletion as withdrawal, authority
  removal as retraction, readmission after withdrawal, or backdated corrections.
- A durable decision journal, retrospective audit reconstruction, or complete
  bitemporal storage. No fabricated legacy snapshots or backfill.
- Changes to recurrence, consolidation/reconciliation decisions, confidence,
  activation, retrieval, forgetting, interference, inhibition, context selection,
  chunking, rendering, learning, checkpoints, or customer schemas.
- CogKuraBench implementation, Demo, Agent, external connectors, embeddings,
  network fetches, new runtime dependencies, and automatic diagnostic prompting.

## 4. Proposed public contracts

All new records are frozen, slotted dataclasses in `models.py`, with `StrEnum`
states, constructor validation, immutable collection values, and exports through
`cogkura.__init__`. Use existing `MemoryIdentity`, `ObservationRevisionIdentity`,
`SourceEvidence`, `SemanticPromotion`, and semantic enums instead of parallel
versions. Proposed algorithms belong under `algorithms/`; source resolution stays
in `provenance.py`; orchestration stays in `memory.py`.

### 4.1 Prepared-context assessment

```python
async def assess_source_memory(
    self,
    context: MemoryContext,
    *,
    max_evidence: int = 256,
) -> SourceMemoryAssessment: ...
```

This assesses exactly the selected primaries and included chunk members, in the
same order as `resolve_provenance()`. It calls the shared resolver once and
computes additional pure diagnostics. It does not recall again, inspect other
slots, or change `context.assessment`. Tenant, subject, and semantic evaluation
times come from the supplied context. An empty context yields an empty report,
not evidence that a requested fact is unknown or withdrawn.

```python
@dataclass(frozen=True, slots=True)
class SourceMemoryAssessment:
    tenant_id: str
    subject_id: str | None
    assessed_at: datetime  # context.prepared_at; cognitive clock
    valid_at: datetime | None
    provenance: ContextProvenance  # existing contract, unchanged
    items: tuple[SourceMemoryItem, ...]
    quality: EvidenceQuality
    reasons: tuple[SourceAssessmentReason, ...]


@dataclass(frozen=True, slots=True)
class SourceMemoryItem:
    memory_identity: MemoryIdentity
    representation: MemoryProvenanceRepresentation
    revision_key: str | None  # semantic identity within history
    quality: EvidenceQuality
    currentness: SourceCurrentness
    promotion: SemanticPromotion | None
```

`SourceMemoryItem` is a diagnostic companion to `MemoryProvenance`, not a second
source-evidence representation. Source currency is explicitly relative to the
store view at inspection, which may be later than context preparation.

### 4.2 Exact slot inspection

```python
async def inspect_semantic_slot(
    self,
    slot: SemanticSlotQuery,
    *,
    tenant_id: str,
    subject_id: str | None = None,
    as_of: datetime | None = None,
    valid_at: datetime | None = None,
    limits: SourceInspectionLimits | None = None,
) -> SemanticSlotAssessment: ...


@dataclass(frozen=True, slots=True)
class SemanticSlotQuery:
    subject_entity_id: str | None
    predicate: str
    qualifiers: Mapping[str, Any]  # default empty immutable mapping


@dataclass(frozen=True, slots=True)
class SourceInspectionLimits:
    max_revisions: int = 128
    max_relations: int = 256
    max_evidence: int = 256


@dataclass(frozen=True, slots=True)
class SemanticSlotAssessment:
    tenant_id: str
    subject_id: str | None
    slot_key: str
    assessed_at: datetime
    valid_at: datetime | None
    state: SemanticKnowledgeState
    conflict: SourceConflictAssessment
    revisions: tuple[SemanticRevisionAssessment, ...]
    quality: EvidenceQuality
    inventory_complete: bool
    decision_history: DecisionHistoryAvailability  # UNAVAILABLE in 0.18.3
    reasons: tuple[SourceAssessmentReason, ...]
```

Reuse the current NFKC/text/qualifier canonicalization and hash inputs in
`semantic.py`. Extract a shared slot-key helper if needed; assert unchanged hashes.
The slot excludes object value, polarity, and cardinality, as it does today.
`subject_id` is the existing storage scope, distinct from `subject_entity_id`.
Omitting it permits inspection within the named tenant; it never removes the
tenant requirement. No natural-language slot inference or fuzzy slot matching.

`SemanticRevisionAssessment` fields: `memory_identity: MemoryIdentity`,
`revision_key: str`, `projected_status: SemanticMemoryStatus | None`,
`view_state: SemanticRevisionViewState`, `valid_from/valid_until: datetime | None`,
`sources: tuple[SourceEvidence, ...]`, `quality: EvidenceQuality`,
`currentness: SourceCurrentness`, `promotion: SemanticPromotion | None`,
`transitions: tuple[SemanticTransitionAssessment, ...]`, and
`historical_attribution_complete: bool`.
`SemanticRevisionViewState` values are `VISIBLE_CURRENT`, `VISIBLE_HISTORICAL`,
`WITHDRAWN`, `PENDING_EFFECTIVE`, and `INDETERMINATE`.
`SemanticTransitionAssessment` contains `relation: SemanticUpdateRelation`,
`predecessor_revision_key: str`, `successor_revision_key: str`,
`effective_at: datetime | None`, `knowledge_at: datetime | None`, and
`asserting_evidence: ObservationRevisionIdentity | None`. A withdrawal successor
is the existing retracted revision. Direction comes from snapshots/intervals,
not lexical ordering of relation endpoints; omit an unprovable transition and
report `TRANSITION_ATTRIBUTION_UNAVAILABLE`.
`projected_status` is the status in the permitted temporal view. Do not expose a
present-day predecessor's `SUPERSEDED`/`RETRACTED` status in an earlier view where
that transition is still unknown.

Inspection covers stored semantics and their retained evidence independently of
activation and recall thresholds. It does not make forgotten memories retrievable.
`KNOWN` means an applicable semantic assertion is stored; it does not mean it was
retrieved or established as true. There is no database-wide unpromoted-claim scan.

### 4.3 Actual-pass decision diagnostics

```python
@dataclass(frozen=True, slots=True)
class SemanticDiagnosticsConfig:
    enabled: bool = False
    max_items: int = 128


# Proposed Memory constructor keyword:
# semantic_diagnostics_config: SemanticDiagnosticsConfig | None = None

# Proposed additive SemanticConsolidationResult field:
# decisions: SemanticDecisionReport | None = None


@dataclass(frozen=True, slots=True)
class SemanticDecisionReport:
    tenant_id: str
    subject_id: str | None
    evaluated_at: datetime
    authority_evaluated: bool
    corrections_evaluated: bool
    items: tuple[SemanticDecision, ...]
    total_decisions: int
    is_truncated: bool
    diagnostics_complete: bool


@dataclass(frozen=True, slots=True)
class SemanticDecision:
    operation: SemanticDecisionOperation  # ADMIT, REPLACE, WITHDRAW
    outcome: SemanticDecisionOutcome  # APPLIED, ALREADY_APPLIED, DEFERRED, INVALID
    reasons: tuple[SemanticDecisionReason, ...]
    slot_key: str | None
    claim_key: str | None
    target_revision_key: str | None
    resulting_revision_key: str | None
    asserting_evidence: ObservationRevisionIdentity | None
    effective_at: datetime | None
```

The report is produced by the **actual existing pass**, using already-loaded
inputs, the qualification result, the edit plan, and successful storage result.
`APPLIED` is emitted only after the existing operation commits; an already applied
edit is `ALREADY_APPLIED`. A storage failure raises `StorageError` and returns no
successful report. No diagnostic journal write is introduced.

Required reason codes mirror existing branches: `AUTHORITY_COMPETES_WITH_CURRENT`,
`COMPETING_AUTHORITY_CLAIMS`, `READMISSION_AFTER_WITHDRAWAL`,
`TARGET_UNAVAILABLE`, `TARGET_NOT_UNIQUE_CURRENT`, `TARGET_CONTESTED`,
`COMPETING_CORRECTIONS`, `COMPETING_WITHDRAWALS`, `REPLACE_WITHDRAW_CONFLICT`,
`AUTHORITY_REQUIRED`, `MALFORMED_DECLARATION`, `EXACT_REVISION_UNAVAILABLE`,
`STALE_OR_DELETED_AUTHORIZING_SOURCE`, `WRONG_SLOT`, `UNSUPPORTED_CARDINALITY`,
`SAME_CLAIM_REPLACEMENT`, `INVALID_EFFECTIVE_TIME`, `TARGET_ALREADY_CLOSED`,
`APPLIED`, and `ALREADY_APPLIED`.

These explain existing behavior; they do not change branch order or outcomes.
For example, an unknown predecessor is currently **deferred**, as verified by
`test_unknown_predecessor_is_rejected`, despite that test's name. A different
reason taxonomy must not silently turn it into an invalid operation.

Malformed records may lack slot or target identity; retain a record with nullable
fields rather than attach it to a guessed slot. Disabled mechanisms are marked
unevaluated without scanning ignored metadata. A custom extractor/consolidator
that cannot supply a reason must set `diagnostics_complete=False`, not report a
default success. Truncation and diagnostic capability completeness are distinct.

Deduplicate display items by operation, exact asserting revision, canonical claim,
target, and effective time; preserve existing aggregate counters, including their
present counting units. `total_decisions` counts unique diagnostic items and is
not guaranteed to equal a legacy counter. Bound output after deterministic
ordering, with totals computed over the actual evaluated pass. Reprocessing may
repeat a deferral report; it never creates a durable unresolved request.

Applications retain these reports if they need a processing audit. A report says
what happened at `evaluated_at`; it is not evidence that the same request is still
unresolved after another process call, source edit, restart, or configuration
change. Slot inspection does not accept old reports as current state. Without a
decision journal it returns `decision_history=UNAVAILABLE`, never an empty-history
claim. Neither source-read API runs `process()` to populate this field.

### Validation and errors

Reject blank tenant/predicate, naive datetimes, non-positive or boolean limits,
and malformed query values with `ValidationError`. Capture UTC `as_of` once if
omitted. `valid_at` is independently optional. Preserve existing storage errors;
missing historical evidence is report data, not an exception. An unsupported
inspection store returns explicit capability-unavailable diagnostics as described
in section 8; it must not return an empty inventory classified `UNKNOWN`.

## 5. Evidence-quality definitions

`EvidenceQuality` contains independent counts, not a score:

| Field | Exact meaning |
| --- | --- |
| `referenced_count: int` | Evidence occurrences referenced by the memory or report population, before capping. |
| `returned_count: int` | Referenced occurrences actually examined after the cap. |
| `revision_resolved_count: int` | Examined occurrences with the exact revision available, including `SOURCE_REF_MISSING`. |
| `locator_present_count: int` | Examined occurrences with a usable revision-specific `source_ref`. |
| `relation_known_count: int` | Examined semantic occurrences with proven support/contradiction relations; direct episodic references are not applicable. |
| `distinct_source_count: int` | Distinct resolved `(source_namespace, source_record_id)` pairs; a lower bound when incomplete. |
| `has_evidence: bool` | `referenced_count > 0`. Zero references do not establish evidence adequacy. |
| `is_truncated: bool` | At least one reference was omitted by the configured cap. |
| `revision_coverage_complete: bool` | No truncation and all referenced exact revisions resolved; report separately from `has_evidence`. |
| `locator_coverage_complete: bool` | No truncation and all references have a locator. |
| `relation_coverage_complete: bool | None` | All applicable semantic relations are known, including no omitted semantic references; episodic omissions do not reduce this coverage. `None` when no semantic references exist. |

No ratio is returned for a zero denominator. All returned counts are bounded by
`returned_count <= referenced_count`. Missing locators lower locator coverage,
not revision resolution. Missing revisions never borrow current metadata, version,
or observed time. The existing `ContextProvenance.is_complete`, counts, and status
meanings remain unchanged for compatibility.

Per-memory counts refer to that memory's occurrences. Context aggregate occurrence
counts include the same reference if it occurs in both a semantic primary and a
chunk member, matching existing provenance traversal. Distinct sources and store
lookups are deduplicated globally; occurrence counts are never presented as
independent corroborations. A semantic relation is known only when derivation
evidence proves it. Ambiguous mixed derivations remain relation-unknown.

## 6. Knowledge, conflict, and uncertainty semantics

### 6.1 Slot states

`SemanticKnowledgeState` values and classification order:

| State | Conditions in the visible, inspected slot |
| --- | --- |
| `INDETERMINATE` | Slot capability unavailable, inventory truncated, inconsistent projections, or insufficient temporal/history data to classify. |
| `CONTESTED` | An applicable visible revision is `CONTESTED`, or existing visible conflicting revisions establish unresolved semantic contention. |
| `KNOWN` | Applicable visible active assertion(s), no established unresolved semantic contention. Multiple cardinality-many values may coexist. |
| `WITHDRAWN` | No applicable active assertion; an explicit visible retraction establishes that the slot's applicable value was withdrawn. |
| `HISTORICAL_ONLY` | Retained superseded/expired assertions exist, with no applicable active or withdrawn projection at the requested time. |
| `UNKNOWN` | Complete slot inventory contains no semantic assertion applicable to the requested view and no visible retained history explaining its absence. |

Evidence-quality incompleteness alone does not erase an otherwise proven stored
status: a known assertion with a missing source remains `KNOWN` with incomplete
evidence. In contrast, missing inventory or temporal projection needed to choose
the status makes the slot `INDETERMINATE`. Positive visible conflict may still be
reported when the overall state is indeterminate.

Withdrawal is an explicit operation, not negation or absence. A stored `DENY`
assertion can be known. A superseded revision is historical, not an ongoing
conflict merely because it differs from its successor. One withdrawn alternative
does not make the whole slot withdrawn if another applicable active value exists;
expose its retraction on that revision. Inconsistent cardinality-one occupants
without enough information to establish a recorded conflict are indeterminate.

An empty recall result, threshold suppression, forgotten dynamics, a missing
locator, source deletion, and absent evidence do not establish `UNKNOWN` or
`WITHDRAWN`. Unknown is explicitly relative to the stored semantic inventory,
not all observations, external reality, or possible unpromoted facts.

### 6.2 Conflict and source disagreement

`SourceConflictAssessment` fields:
`state: SourceConflictState`, `conflicting_revision_keys: tuple[str, ...]`,
`distinct_supporting_sources: int`, `distinct_contradicting_sources: int`,
`cross_source_disagreement: bool | None`, `independence: SourceIndependenceState`,
and `is_complete: bool`.

- `SourceConflictState` is `NONE_OBSERVED`, `SEMANTIC_CONFLICT`, or `INDETERMINATE`.
  `NONE_OBSERVED` requires complete inspected conflict inventory and means no
  conflict was observed in that population. It does not rule out deferred or
  unpromoted source assertions.
- `cross_source_disagreement=True` requires an existing semantic incompatibility
  plus exact evidence establishing different source identities on opposing
  claims. `False` requires complete attribution and no cross-source opposition;
  `None` means missing/ambiguous evidence prevents the decision. Known disagreement
  remains true even if other evidence is missing.
- Repeated episodes or revisions of one source do not establish cross-source
  disagreement. Cardinality-many coexistence and non-overlapping temporal
  succession do not establish disagreement. Reuse recorded semantic relation
  meanings; do not apply retrieval competition pairs as factual contradictions.
- `SourceIndependenceState.UNKNOWN` is the only supported independence value in
  this release. Two different source records can quote the same upstream source.
  The application can use externally known independence in its own policy; Core
  supplies the evidence identities without claiming to verify it.
- Pass reports identify competing authorised admissions and incompatible edits
  even when these never became contested semantic rows. This is unresolved
  **operation** conflict at that pass, distinct from stored semantic contention.
  A deferred operation does not change an active memory to `CONTESTED`.

`SourceAssessmentReason` is a fixed-order diagnostic enum: `NO_EVIDENCE`,
`SOURCE_UNAVAILABLE`, `REVISION_UNAVAILABLE`, `LOCATOR_MISSING`,
`RELATION_UNAVAILABLE`, `EVIDENCE_TRUNCATED`, `INVENTORY_TRUNCATED`,
`LEGACY_HISTORY_INCOMPLETE`, `TEMPORAL_HISTORY_UNAVAILABLE`,
`TRANSITION_ATTRIBUTION_UNAVAILABLE`, `SOURCE_REVISION_SUPERSEDED`,
`SOURCE_CURRENTNESS_UNKNOWN`, `SEMANTIC_CONTESTED`, `CROSS_SOURCE_DISAGREEMENT`,
`FACT_WITHDRAWN`, `HISTORICAL_ONLY`, `PENDING_EFFECTIVE`,
`DECISION_HISTORY_UNAVAILABLE`, and `INSPECTION_CAPABILITY_UNAVAILABLE`.
Reasons are independent of the existing `MemoryAssessmentFlag` enum and never
feed ranking or render output.

### 6.3 Application uncertainty and abstention

Applications can combine retrieval assessment, source assessment, exact-slot
state, and the current process result. For example, an application may decline
to answer when the requested slot is withdrawn/contested/indeterminate, when the
latest pass deferred its requested correction, or when required evidence cannot
be resolved. That is application policy; no such decision is built into Core.

An application may require independence that Core cannot establish, or a freshness
service-level rule that Core does not know. `UNKNOWN` and `None` remain visible
so that absence of a diagnostic is not mistaken for certainty. High accessibility,
complete attribution, and application authority never override those unknowns.

## 7. Currentness and temporal boundaries

`SourceCurrentness` contains `basis: SourceCurrentnessBasis` (only
`STORE_VIEW_AT_INSPECTION`), `latest_count: int`, `older_revision_count: int`,
`unknown_count: int`, `is_truncated: bool`, and `is_complete: bool`. Counts cover
examined evidence occurrences; truncation prevents complete currentness.
Map existing `is_latest_known_revision=True/False/None` directly. No inference
from `source_version` lexical ordering, a missing version, or evidence age.

Being the latest revision retained by Core does not mean the external source is
fresh or the fact true. Being an older revision does not invalidate a semantic
fact, supersede it, or withdraw it. Optional existing metamemory freshness remains
an age/accessibility diagnostic, with its old configuration and formula.

The following clocks must remain separate:

1. `valid_at` selects the factual interval, half-open `[valid_from, valid_until)`.
2. `as_of` is the cognitive/knowledge projection clock used by the supported
   0.18.2 correction path. A future asserting observation must not reveal its
   successor, withdrawal, authorising evidence, transition link, or reason early.
3. Source revision comparison reflects the store view when inspected. The
   observation API has no historical latest-pointer query; it cannot honestly
   reconstruct that pointer at an earlier `as_of`.

For slot inspection, project transitions with the existing correction helpers,
then classify. A future-effective correction known by `as_of` leaves its
predecessor applicable until the effective boundary and may expose a
`PENDING_EFFECTIVE` transition. At the boundary the successor applies. A visible
withdrawal leaves the old revision historical before the boundary and the slot
withdrawn at/after it. Before the asserting observation, expose neither pending
status nor administrative details. Do not infer direction from sorted keys.

Exact evidence whose `observed_at > as_of` must not contribute to the slot's
historical evidence/conflict report. If excluding later evidence leaves the
stored status's historical basis unprovable, return `INDETERMINATE` with
`TEMPORAL_HISTORY_UNAVAILABLE`, rather than report a reconstructed certainty.
Legacy empty snapshots do not certify historical authority or correction links.
Ordinary reinforcement can update counts without preserving every prior state;
this release does not promise full historical reconstruction of all confidence,
support counts, or ordinary contention. Where that matters to classification,
report the limitation explicitly. No changes to existing retrieval are implied.

The context API preserves the existing provenance contract, including retained
evidence and present-store currency. It is not a new historical inventory query.
The slot API is the temporal classification surface. When `as_of` is explicitly
supplied, set all examined source currency comparisons to unknown, including
`SourceEvidence.is_latest_known_revision=None` on returned copies. This uniform
rule avoids letting the presence of a later source update change an earlier
report's currentness flag. Omitted `as_of` permits present-store comparison.
Its currentness basis remains explicit even on a live query. Return only visible
revision/transition identities and visible-population counts; do not expose raw
hidden future rows or their administrative metadata through the report.
Neither API promises an external ingestion-time audit: application `observed_at`
and stored history are the available evidence, not proof of wall-clock ingestion.

## 8. Storage and performance

### Read capability

Add an optional `SemanticInspectionStore` protocol in `storage/base.py` with:

```python
async def inspect_slot(
    self,
    *,
    tenant_id: str,
    subject_id: str | None,
    slot_key: str,
    max_revisions: int,
    max_relations: int,
) -> SemanticSlotRecords: ...
```

`SemanticSlotRecords` contains current projection rows, revision rows, relation
rows, and independent overflow indicators for each population. Projection rows
are bounded by `max_revisions` too, with their own overflow bit. Load one extra
row per bounded population to detect overflow; do not pretend a capped count is
an exact total. Apply bounds before decoding large snapshot payloads where
possible. Relations retain both endpoint keys; missing endpoints due to a bound
make classification incomplete. Do not filter by current `ACTIVE` status before
loading history, since withdrawn and contested slots are required inputs.

Default in-memory and PostgreSQL semantic stores implement this capability.
Existing `SemanticMemoryStore` implementations remain valid without it. Inject an
optional `semantic_inspection_store` into `Memory` for custom backends; otherwise
use the semantic store if it advertises the capability. If unavailable, report
`INDETERMINATE` and `INSPECTION_CAPABILITY_UNAVAILABLE`. Do not fall back to an
unbounded tenant scan or access backend-private storage.

PostgreSQL reads only CogKura tables, joined on tenant and memory identity.
Use existing `(tenant_id, slot_key)` and `(tenant_id, memory_key)` indexes from
migrations `003` and `006`. Read the slot populations in a short read-only
repeatable-read transaction; copy the in-memory populations before pure analysis.
Use endpoint joins to read stored relation records, not a quadratic all-pairs
reconstruction. Source/episode batch reads may occur afterward through their
existing protocols; the full report is therefore not promised to be one atomic
cross-store snapshot. Under concurrent source changes, missing/mismatched
attribution reduces completeness; never repair a missing revision with current
metadata. Strong cross-store snapshot isolation is excluded from this release.

### Persistence and compatibility

- No proposed table, column, migration, write-time source index, persisted
  assessment, or decision journal. Reuse migration `011` snapshot data.
- No changed fingerprints, claim/slot/revision keys, stored statuses, admission
  thresholds, or relation meanings.
- Decision reports are optional result data; their default is `None`. Disabled
  reporting adds no source reads and does not evaluate ignored declarations.
- No optional-field addition to `MemoryMonitor.assess()` or mandatory new method
  on existing store protocols. Custom monitors and stores retain old behavior.
- New reports expose only typed memory/revision identities, existing attribution,
  counts, states, and declared times. Never expose arbitrary metadata or source
  content. `source_ref` is inert data and is never opened or fetched.

### Bounds and cost

Context assessment retains the existing global evidence occurrence cap and order:
selected primaries, then included members, with each memory's evidence ordered
by sequence number/observation ID/revision. Reuse deduplicated batch observation,
revision, and episode evidence reads. Do not create a `MemoryContext` containing
unselected history merely to call the public context resolver; share its
lower-level evidence resolution helpers instead.

Slot population ordering is `(memory_key, revision_number, revision_key)`;
relations order by `(left_revision_key, right_revision_key, relation)`. Pure
analysis is linear in returned records, relations, and references apart from
deterministic sorting. Conflict evidence sets use existing relationships rather
than enumerate every possible source pair. Storage round trips are a fixed
small number of bounded batch calls, never one per source. The context resolver's
existing derivation-evidence loading can precede the output cap; do not advertise
`max_evidence` as a hard database-row or byte budget. Slot evidence and JSON
snapshot sizes have the same limitation and must be measured.

Record pass counts before truncation, but do not materialize an unbounded list of
display decisions: keep bounded deterministically ordered records plus counts.
If query-plan or payload measurements require a new index, byte budget, or
snapshot representation, report the concern for review instead of silently
expanding schema scope. No global cache is proposed; stale cached diagnostic
state must not survive source edits, reprocessing, or tenant clearance.

## 9. Non-interference guarantees

For identical stored state, configuration, query, `as_of`, and `valid_at`:

1. Calling either assessment API any number of times performs no writes to
   observations, episodes, semantics, revisions/relations, activation references,
   dynamics, learning, relationships, inhibition traces, or connector checkpoints.
2. `recall`, `inspect_recall`, `select_working_memory`, `assess_memory`, and
   `prepare_context` retain the same results, scores, order, eligibility,
   components, flags, working-memory membership, budgets, and rendered bytes.
3. Assessments never call `record_access`, `record_context_use`, `learn`,
   `apply_forgetting`, `encode_episodes`, `consolidate_semantics`, or `process`.
4. Enabling decision diagnostics changes only optional diagnostic output.
   All old pass counters and durable rows, correction boundaries, authority
   decisions, replay behavior, and recurrence outcomes remain identical.
5. No source-aware report, enum, or reason enters a retrieval feature, confidence
   formula, reconciler input, forgetting rule, inhibition rule, or render path.
6. The frozen `0.17.6.1`/`0.17.7` cognitive contract and the validated 0.18.2
   correction exceptions remain the reference behavior. Default switches remain
   off. No benchmark baseline is retuned or redefined.

Use store spies that fail on writes, state comparisons before/after inspection,
and diagnostics-on/off paired runs with fixed clocks. Compare logical durable
identities and evidence, not randomly generated row UUIDs across fresh backends.

## 10. Detailed test matrix

| Area | Cases | Required assertion |
| --- | --- | --- |
| Public validation | Blank tenant/predicate, naive times, invalid limits, immutable maps/tuples, Unicode/qualifier normalization | Deterministic validation; unchanged existing canonical keys. |
| Exact revision | Current source v2, evidence v1; absent v1; absent source; deleted present source | Exact historical attribution or explicit missing state; never substitute v2. |
| Quality | Missing locator, no references, mixed known/unknown relations, contradictory derivations | Locator and revision coverage differ; zero references has `has_evidence=False`; no invented relation. |
| Counts | Shared source in primary/member; repeated episodes; several revisions of one record; two records in one namespace | Correct occurrence counts, unique lookups, source-pair count; no independence claim. |
| Context representation | Semantic primary, episode member, duplicate representations, empty context | Preserve existing provenance order and member/render distinction; no inferred unknown slot. |
| Bounds | Limits 1, exact limit, overflow; early item exhausts evidence budget; truncated relation endpoints | Stable prefix, explicit overflow, lower-bound counts, no false complete/unknown/clear state. |
| Unknown vs inaccessible | Empty slot; unpromoted observation; stored semantic below threshold; forgotten semantic | Unknown is relative to inventory; stored semantic remains known when retrieval is empty. |
| Current/historical | Known assertion, superseded-only interval, expired claim, negative assertion, legacy snapshot | Correct distinct state; legacy attribution unavailable rather than fabricated. |
| Withdrawal | Before/at/after boundary; source deletion without withdrawal; explicit retraction with retained history | Half-open semantics; only explicit withdrawal produces withdrawn state. |
| Temporal knowledge | `as_of` before asserting observation with earlier/equal/later `valid_at`; known future-effective correction | No future identity/link/reason/evidence leaks; predecessor remains visible until applicable. |
| Historical limitations | Later ordinary reinforcement/conflict, later source revision, incomplete snapshots | Mark unsupported historic classification indeterminate; no historical latest-pointer claim. |
| Semantic conflict | Recorded contested slot; overlapping cardinality-one alternatives; many-valued coexistence; non-overlapping succession | Only existing incompatibility is conflict; coexistence/succession is not disagreement. |
| Sources | Opposing claims with distinct records; one source across revisions; shared upstream unknown; missing attribution | True/false/unknown follows evidence; independence always unknown. |
| Authority deferral | New value competing with current; incompatible same-pass authority; readmission after withdrawal | Existing authority counters/state unchanged; precise pass-local reason. |
| Invalid edits | Missing authority, malformed declaration, stale/deleted asserting revision, wrong slot/cardinality, invalid effective time | Existing rejection branch retained; typed reason and exact asserting identity when known. |
| Deferred edits | Unknown target; contested/non-unique target; competing replacements; competing withdrawals; replace plus withdraw; same-batch recurrence | Existing deferral/counts unchanged; no winner or partial application. |
| Success/replay | Adopt, replace, withdraw, same-batch predecessor, future-effective operation, repeat each at least three times | Committed vs already-applied outcomes accurate; no new revision/relation on replay. |
| Report lifetime | Diagnostics off/on; truncated report; restart without saved report; later source edit/process | Off is `None` and no extra I/O; no old report promoted to current unresolved state. |
| Isolation | Same source/slot names across tenants, subject filters, unavailable foreign target, custom unsupported store | No cross-tenant result/existence disclosure; unsupported is indeterminate, never unknown. |
| Concurrency/errors | Source update/deletion between batches; unavailable evidence; SQL/storage failure | Honest completeness and propagated storage failure; no fabricated success or latest substitution. |
| Pure reads | Repeated context/slot inspection; all supported feature switches on/off | Zero writes; no extra inhibition reads simply for diagnostics; unchanged recall/context. |
| Cognitive freeze | Recurrence and authority, default and enabled corrections, context reinstatement, forgetting, interference and inhibition suites | Exact old results/components/render and durable state; no warm-up or threshold retuning. |
| Backend parity | In-memory vs PostgreSQL for all above applicable cases, including migration-011 and legacy snapshots | Same logical reports/states/order/counts/replay; explicit skipped/not-executed status if database absent. |
| Performance | Many unrelated slots, long revision history, repeated shared references, large snapshots | Bounded returned populations, constant batch-call count, measured query plans and payload limits. |

Extend existing `test_source_provenance.py`, `test_metamemory_api.py`,
`test_018_1_source_authority.py`, `test_018_2_authoritative_corrections.py`,
and PostgreSQL integration coverage; add a focused source-assessment test module.
Do not rewrite frozen fixtures to fit new outputs.

## 11. Acceptance and implementation gates after approval

1. Approve or revise D1–D6 below before changing Core code.
2. Implement only approved additive contracts and tests on this feature branch;
   commit coherent increments. Update docs to distinguish shipped behavior from
   known gaps only after verification. No version bump during design review.
3. Every deterministic matrix case passes. At fixed inputs, repeated reports
   match field-for-field, excluding backend-generated IDs not in the public
   diagnostic contract. Every unknown state has a defined reason/population.
4. Non-interference paired comparisons have zero differences in prior outputs,
   old pass counters, durable semantic decisions, and cognitive state. All
   baseline tests pass without fixture, formula, or threshold changes.
5. Run the repository CI gates:

   ```bash
   uv sync --all-extras --dev --locked
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest
   ```

   Required CI covers Python 3.11, 3.12, and 3.13. The new API must also import and
   run with the dependency-free base installation, without PostgreSQL extras.
6. Execute PostgreSQL tests against an available disposable CogKura database,
   including adopt/replace/withdraw/replay parity. Record command, backend
   version, commit, passed/failed/skipped counts, and comparison results. If no
   database is available, retain **NOT EXECUTED** prominently: other checks can
   pass, but backend parity and release validation are incomplete. Do not count
   skipped tests as compatibility confirmation.
7. Compare the frozen benchmark results in the separately authorised validation
   work described below. Document measured performance and any unsupported
   historical or custom-backend cases; do not claim full evidence completeness
   from a limited or incomplete population.
8. Open **one PR only after the complete approved implementation and required
   validation are finished**. This design-only checkpoint does not open a PR.
   Do not push or publish merely to obtain design approval.

### Phase 1 checks actually executed

At the reviewed Core baseline, the following focused existing tests passed:

```bash
.venv/bin/pytest --no-cov -q \
  tests/test_metamemory.py \
  tests/test_metamemory_api.py \
  tests/test_source_provenance.py \
  tests/test_018_1_source_authority.py \
  tests/test_018_2_authoritative_corrections.py \
  tests/test_interference_inhibition_architecture_freeze.py
```

Result: **94 passed**. These are existing baseline tests, not 0.18.3 tests.
No Core implementation was changed. Full lint/type/test release gates, a new
Bench run, and PostgreSQL parity were not executed in this design phase.

## 12. Subsequent Bench validation plan (separate authorisation)

No Bench files change in this task. Propose a subsequent Bench release, with its
own review, to add a versioned `source_aware_metamemory_v1` fixture and a neutral
source-assessment capability. Keep Core-specific calls inside its adapter;
metrics use public outputs mapped to fixture event IDs, not private stores or a
second implementation of Core's qualification rules.

Fixtures should declare unknown/known/contested/withdrawn/historical states,
missing locators vs missing revisions, repeated vs distinct sources, unknown
independence, pass-local deferred edits, pending boundaries, and unavailable
decision history. Include empty retrieval with stored knowledge, conflicts
outside the recall limit, truncated inventories, and unavailable historical
attribution. Author gold independently; label unsupported surfaces N/A rather
than treating them as a clean bill of health.

Measure state accuracy; conflict precision/recall; reason-code accuracy;
exact-revision attribution accuracy; completeness/truncation accuracy; temporal
boundary accuracy; and unknown/unsupported-state accuracy. Require 1.000 on
declared deterministic supported cases and zero false certainty, cross-tenant
leaks, future transition/evidence leaks, or inspection-induced mutations. Do not
score LLM truth or automatic abstention. Independently asserted source
independence cannot be scored as Core knowledge while it is unsupported.

Run paired diagnostics-off/on profiles with otherwise identical configurations,
event processing, clocks, and explicit use. Require zero differences in semantic
decisions, broad recall identities/order/scores, rendered context, activation,
interference, and inhibition effects. Preserve the existing correction control
and enabled profiles and all released fixture ground truth.

Re-run `source_authority_v1`, `authoritative_corrections_v1`, mini,
`software_project_v1`, `helios_v1`, `customer_decision_context_v1`,
`interference_v1`, `transient_interference_v1/v2`, and the controlled/software/
customer persistent-inhibition workloads. Compare with Bench `0.3.10`:
authority and correction declared accuracies 1.000; control recall@5 mini 1.000,
software 1.000, helios 0.800, customer 0.600; transient v2 precision/recall/F1/
direction 1.000; persistent trace/effect precision and recall 1.000;
`interference_v1` competition F1 0.435. Preserve per-case outputs as well as
aggregate metrics so an unchanged average cannot hide a regression.

Repeat supported source/slot assessment and the outstanding correction parity on
PostgreSQL. If that remains unexecuted, retain the verification gap in the new
findings and do not declare the release fully validated. Bench findings must
distinguish frozen retrieval regressions from limitations intentionally inherited
from 0.18.2, including later re-adoption episode prose.

## 13. Decisions requiring human review

| ID | Decision | Recommendation and consequence |
| --- | --- | --- |
| D1 | Separate APIs or automatic enrichment of `assess_memory`/`prepare_context`? | Approve separate context assessment and exact-slot inspection. Existing APIs, monitor protocols, latency, and flags remain stable; applications explicitly request source diagnostics. |
| D2 | Is pass-local correction/authority observability sufficient for 0.18.3, or must unresolved requests survive restarts and be enumerated? | Approve pass-local typed reports plus explicit unavailable history. If durable unresolved inventory is required, revise the design first for a tenant-scoped journal, lifecycle, atomic writes, replay, retention, and migrations. That is not hidden inside this proposal. |
| D3 | Must Core assert independence between sources? | Report different source identities and disagreement, with independence unknown. Application-asserted origin groups or independence attestations require a separate contract; neither can be inferred from present provenance. |
| D4 | What historical guarantees should the new inspection API claim? | Preserve supported 0.18.2 transition projection; mark incomplete legacy/ordinary historical state indeterminate. Currentness is present-store comparison, not a past latest-pointer reconstruction. Full bitemporal history is excluded. |
| D5 | Is the optional bounded storage capability acceptable for custom stores? | Approve it with explicit unsupported diagnostics and no unbounded fallback. Implement both built-in backends; no mandatory break to existing store protocols and no initial migration. |
| D6 | Are PostgreSQL parity and separate Bench confirmation release gates? | Yes. Close the outstanding database gap and complete separate authorised Bench validation before claiming release readiness. Do not change Bench, Demo, or Agent in this Core task. |

D1–D6 were approved as recommended on 14 November 2026. The user also clarified
that relation coverage must distinguish non-applicable semantic relations, decision
diagnostics must observe actual existing outcomes without duplicated decisions or
counter changes, and the 0.18.2 stale episodic prose limitation must remain.
These clarifications do not require renewed approval unless a material architectural
change is discovered. Core validation precedes the one implementation PR;
independent Bench validation remains a separately authorised release gate.
