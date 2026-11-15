# Corrective design: withdrawn readmission in current context

**Status:** Approved explicitly by the user on 15 November 2026; implementation
and Core/independent Bench validation complete. The resulting exact one-query
frozen-output deviation and scoped acceptance were separately approved that day.
**Reviewed baseline:** Core `f62096a` / runtime `0.18.3`, 15 November 2026.
**Relationship:** Correctness prerequisite to the verified 0.18.4 architecture freeze.

## Finding and release decision

An application admits `migration_execution_strategy=application_startup`, explicitly
withdraws its named revision, then observes a new authoritative episode asserting
the same claim. Consolidation correctly defers readmission and exact slot inspection
returns `WITHDRAWN`. Nevertheless, current `recall()` and `prepare_context().render()`
can expose the later assertion as ordinary current evidence.

This is a correctness defect, not an acceptable frozen invariant. It violates
the intended current-context boundary in the
[0.18.2 design](design-authoritative-corrections-0.18.2.md). It is not safe to call
the architecture dependable for production Agent integration while this remains
unresolved. This finding originally blocked the freeze. The approved correction
and its validation now resolve that blocker; production Agent readiness remains
the separate 0.20.0 integration milestone.

Existing `test_readmission_after_withdrawal_does_not_restore_the_assertion` checks
deferral and durable semantic history, but never recalls or renders context.
Other withdrawal tests cover the original supporting episode and mixed prose,
not a later episode that was never admitted as support. Thus the original defect
is documented but lacks a rendering correctness regression.

## Cause

`corrections._stale_episode_assertions()` identifies visible closed assertions,
then attaches redaction rules only to `predecessor.derivations`. A deferred later
episode is absent from those derivations. `selectively_filter_stale_episode_content()`
therefore passes it through unchanged. Both recall and inspection use this helper;
working memory and rendering subsequently consume the result.

## Proposed correction

Extend only the existing read-side stale-assertion projection. Keep the current
support-episode path. Additionally identify candidate episodes carrying an exact
structured assertion of a known closed claim, even if never admitted as support.

- Reuse existing claim canonicalization: tenant, subject entity, predicate,
  qualifiers, object identity/value, polarity and cardinality. Do not match solely
  on a generic value, source name, confidence, authority label or query text.
- Read the episode's exact referenced observation revisions through existing
  deduplicated batch APIs when structured identity is needed. Do not substitute
  latest source metadata or run semantic qualification/consolidation during reads.
- Apply a suppression rule only where the explicit closing transition is known
  at `as_of` and effective at `valid_at` (or the operation's current boundary).
  Before those boundaries retain the predecessor. Do not broaden bitemporal claims.
- Reuse the existing sentence-fragment filtering and conservative whole-episode
  omission if safe separation is not provable. Keep separable unrelated facts,
  including an ordinary fact sharing the same object value.
- Preserve raw stored episodes, exact source evidence, revision history, semantic
  admission/deferral counters, processing diagnostics, and explicit-use behavior.
  No migration, dependency, mandatory protocol, trust ranking or new public API.
- Do not infer arbitrary unstructured prose as a semantic claim. Incomplete or
  malformed structured identities do not justify a new broad lexical filter.
  Any resulting residual limitations must be reported explicitly.

This intentionally changes eligible/rendered episodic content for the identified
stale-claim cases. It does not change activation formulas, thresholds, learning,
interference or inhibition. It must not be presented as behavioral non-interference.

## Verification and approval boundary

Add deterministic positive correctness tests for original and later evidence,
ordinary and authoritative repeated assertions, mixed unrelated facts, same-value
different predicates, qualifier/entity/tenant/subject isolation, and historical
knowledge/effective boundaries. Test recall, recall inspection, prepared context,
stored-history immutability and in-memory/PostgreSQL logical parity.

Run the full Python 3.11–3.13 Core matrix with PostgreSQL and dependency-free import.
Measure any added exact-revision reads against 0.18.3. Independently reproduce the
case in Bench through public APIs; preserve released dataset gold and comparison
definitions. Run existing strict frozen and diagnostics comparisons unchanged.

If intentional correction changes any existing frozen fingerprint, retain the
failed comparison and enumerate the exact affected queries, before/after content,
and metrics for separate approval. Do not rebaseline, weaken comparisons, or
declare the strict gate passed without that explicit decision.

**Approval and outcome:** the user approved this narrow retrieval correction, then
separately approved the one observed `ac-q-c20` deviation with exact version,
commit, gold digest and before/after fingerprints. The original failed comparison
remains intact; the opt-in approved gate passes without other drift. See
[freeze findings](findings/0.18.4-source-provenance-architecture-freeze.md) for exact
commits, full gates, read costs and residual limitations. Future retrieval changes
or baseline deviations require their own review.
