# CogKura Core 0.18.4 — Source provenance architecture freeze

**Status:** Audit and non-behavioral implementation authorised by the user's
15 November 2026 request. The user also explicitly approved the separate
[withdrawn-readmission correction](design-withdrawn-readmission-context-correction.md).
Implementation/verification is in progress; freeze acceptance remains pending.
**Baseline:** validated Core 0.18.3, `f62096a`; Bench 0.3.11, `fca1f90`.
**Release actions:** direct commits on `main`; tags/publication remain separate.

## Scope and prior contracts

This is integration verification, documentation and an architecture freeze, with
no new cognitive capability. The normative source contracts remain the preceding
[provenance](design-source-provenance-0.18.0.md),
[authority](design-source-authority-0.18.1.md),
[correction](design-authoritative-corrections-0.18.2.md),
[metamemory](design-source-aware-metamemory-0.18.3.md), and
[canonical slot ordering](review-source-inspection-ordering-0.18.3.md) designs.
The user's supplied 0.18.4 proposal is the acceptance specification; this note
records its audit, implementation boundary and required release decisions.

Add `test_source_provenance_architecture_freeze.py` for combined feature paths.
Retain focused algorithm tests rather than copying them to inflate counts. The
invariant matrix maps the full contract to existing coverage plus these additions.
No runtime algorithm, storage protocol, schema, dependency or default is changed
by the freeze itself. SQLite/embedded persistence belongs to 0.19.x; production
Agent lifecycle/API integration belongs to 0.20.0. Demo and Agent are untouched.

## Contract/invariant matrix

Selectors below are pytest function names in `tests/`. Existing suites remain
part of the full required gate. New combined checks are in the freeze suite.

| Contract | Existing verified coverage to retain | New combined coverage / disposition |
| --- | --- | --- |
| Exact source revisions; missing revision never substituted; locator distinct | `test_source_provenance.py`: `test_episode_provenance_uses_the_encoded_revision_after_source_update`, `test_missing_revision_is_reported_without_current_revision_substitution`, `test_missing_source_ref_keeps_resolved_evidence_and_tenant_scope_is_enforced` | Admission/correction/history assessment across both backends |
| Tenant/subject scope; logical identity, not UUID | `test_018_1_source_authority.py`: `test_authority_does_not_cross_tenants_or_subjects`; `test_source_assessment.py`: `test_tenant_subject_isolation_and_no_cognitive_state_changes` | Combined lifecycle parity, foreign-tenant/subject empty slots |
| Selected primary/member provenance and legacy ordering | `test_source_provenance.py`: `test_semantic_support_and_chunk_member_representation_are_explicit`; `test_source_assessment.py`: `test_slot_prefix_ignores_uuid_derived_sequences_and_preserves_context` | Source reads interleaved with rendering preserve context |
| Recurrence default; explicit enabled authority only; malformed claims rejected | `test_018_1_source_authority.py`: ordinary one/two-episode, disabled-authority, malformed/deleted/missing/later-revision tests | Default configuration freeze; no new admission path |
| Authority does not increase confidence/relevance/activation/evidence strength | `test_018_1_source_authority.py`: `test_authority_confidence_and_recall_match_single_episode_consolidation`; `test_same_claim_keeps_authority_evidence_despite_higher_confidence` | Read-only interleaving and diagnostics off/on parity |
| Named correction/withdrawal; historical evidence; idempotent replay | `test_018_2_authoritative_corrections.py`: named replacement, explicit withdrawal, replay and authorising-observation snapshot tests | Admission → future-effective correction → history → withdrawal → replay → slot/source reports |
| Ambiguous edits never silently select a winner; actual pass diagnostics | `test_semantic_decision_diagnostics.py`: competing corrections, replace/withdraw conflicts, malformed authority, failed storage and replay | Competing declarations → unchanged current slot → pass-local deferral reports |
| Known/unknown/withdrawn/contested/historical-only/indeterminate are distinct | `test_source_assessment.py`: state, negative/expired facts, inaccessible recall, legacy history and capability tests | Combined lifecycle state assertions; structured conflict despite recall inaccessibility |
| No inferred source independence/trust/automatic certainty | `test_source_assessment.py`: `test_contested_sources_do_not_claim_independence`; relation/locator/currentness tests | Conflict remains contested with independence `UNKNOWN` |
| Temporal boundaries: factual vs knowledge vs present-store currentness | `test_018_2_authoritative_corrections.py`: half-open intervals, earlier knowledge, future-effective correction; `test_source_assessment.py`: `test_multiple_correction_boundaries_restore_only_the_known_predecessor`, `test_slot_future_evidence_cannot_consume_cap_or_determine_visible_order` | Same lifecycle inspected before knowledge, pending, at effective time, and historical factual time |
| Missing evidence does not necessarily erase proven semantic state | `test_source_assessment.py`: `test_missing_evidence_does_not_erase_known_status_or_substitute_current` | Missing exact metadata remains incomplete; supported status may stay known |
| Unsupported history is indeterminate, not fabricated | `test_source_assessment.py`: `test_legacy_history_is_indeterminate_without_invented_promotion`, `test_unsupported_ordinary_history_exposes_no_present_status_or_false_clear_conflict` | Legacy history combined with missing exact evidence |
| Optional inspection capability; custom stores remain compatible | `test_source_assessment.py`: `test_custom_store_without_inspection_is_unavailable_not_unknown` | Existing focused coverage retained; no mandatory protocol extension |
| Canonical slot order and global occurrence cap | `test_source_assessment.py`: numeric/case ordering, UUID-derived sequences, repeated occurrences, mixed availability; `test_postgres_source_assessment.py`: opposite-sequence parity | Conflict/cap checks across backends at 1, 2, exact, overflow |
| Revision/relation bounds and incomplete coverage | `test_source_assessment.py`: `test_inventory_and_evidence_limits_are_not_false_completeness`; `test_postgres_source_assessment.py`: bounded revision prefixes | Combined conflict with revision/relation/evidence bounds |
| Read-only diagnostics; no reinforcement/learning/lifecycle writes | `test_source_assessment.py`: complete store-state snapshot; `test_semantic_decision_diagnostics.py`: off/on history/counter/recall/render parity | Paired independent stores with controlled identities and interleaved diagnostics |
| Existing 0.17.7 cognitive baseline | `test_interference_inhibition_architecture_freeze.py`, `test_contextual_memory_architecture_freeze.py`, activation/forgetting/learning/context tests | Run unchanged full suite and independent frozen Bench signatures |
| Batched reads; output cap does not bound bytes | PostgreSQL constant-query/read-only tests; 0.18.3 ordering-cost measurements | Repeat fixture against exact baseline/current sources; report source rows, bytes, queries and latency |
| No locator fetching | Allow-listed resolver reads only observation/episode stores; locators remain inert strings | Resolver audit plus deliberately non-fetchable locator in lifecycle fixture |
| Later readmission cannot expose withdrawn assertion as current prose | Existing test covers semantic deferral only; original supporting-episode redaction tests do not cover later episode | **GAP / RELEASE BLOCKER:** separately designed correction needs approval; do not freeze leaked prose |

## Clarifications from the audit

An explicit historical `as_of` masks latest-source comparisons in exact slot
inspection. `assess_source_memory(context)` retains the existing provenance
resolver's **present-store** comparison, even for a context prepared earlier;
its API has no historical-currentness selector. Do not silently change that
interface to make the proposal's wording broader. `valid_at` is factual time;
`as_of` retains each operation's supported cognitive/knowledge-time meaning.

Missing exact evidence alone can leave an otherwise provable stored slot `KNOWN`
with incomplete attribution. It is unsupported temporal reconstruction or legacy
history, rather than every missing revision, that requires `INDETERMINATE`.

Exact slot ordering is case-sensitive namespace/record/numeric revision within
the existing canonical revision population, then one global occurrence cap.
Unavailable identities form a per-revision suffix without cross-UUID portability.
Context provenance and unrelated retrieval retain their existing order contracts.

Source identities must be resolved before capping, so `max_evidence` bounds output
occurrences rather than all metadata/content reads. Preserve deduplicated batches
and report bytes/rows as well as round trips. Do not add caching/indexes/protocols
to optimize a small synthetic timing observation. Atomic cross-store reads and
complete bitemporal history are not promised.

## Stale prose disposition

The existing readmission gap is a correctness defect and is not safe to certify
as an invariant for production Agent use. A deterministic public-API reproduction
shows `WITHDRAWN`, a deferred readmission, and the same assertion in current rendered
context. The separate corrective design specifies a narrow read-side projection
change and an explicit approval boundary for any frozen output deviation.

## Validation and release closure

Run locked install, Ruff, format, MyPy and full pytest on Python 3.11–3.13 with
all PostgreSQL fixtures configured; retain actual counts and zero unexpected skips.
Verify dependency-free use, both backend populations, read-only behavior, and the
same repeatable cost fixture against 0.18.3. Record exact source commits and runtime
versions in `findings/0.18.4-source-provenance-architecture-freeze.md`.

Only after the correctness decision and Core gates are resolved should Bench pin
the exact candidate and run its unchanged strict validator and full CLI/runtime
matrix. Preserve original gold and prior findings. If a deliberate correction
changes frozen signatures, report exact differences and obtain separate approval.

Version metadata advances consistently to the 0.18.4 candidate after approval of
the correction. The 0.18.3 baseline remains the comparison point; the architecture
is not declared frozen before all gates and any deviation decisions are resolved.
No release is published by this work.
