# Cogkura 0.18.1 — Source authority and single-observation consolidation

**Status:** Shipped  
**Baseline:** Cogkura `0.18.0`

Recurrence remains the default route from episodic evidence to semantic memory. `0.18.1` adds a second, explicit route: an application may declare that one structured fact is authoritative, and that fact can be consolidated from a single supporting episode.

Authority changes the evidence required for admission. It does not change confidence, activation, forgetting, retrieval, or automatic correction.

## Two routes

Learning through recurrence:

```text
Observation → Episode → repeated supporting episodes → semantic memory
```

`minimum_supporting_episodes` stays `2` unless the application sets it.

Explicit source authority:

```text
Application declaration on one fact
  → observation revision
  → episode
  → revision check
  → semantic memory
```

Both routes produce ordinary semantic memories. A promoted memory is not guaranteed to be retrieved.

## Declaration

Authority is an optional object on one `semantic_facts` entry:

```python
"authority": {"basis": "application_asserted"}
```

`SemanticAuthorityBasis.APPLICATION_ASSERTED` is the only accepted basis. The declaration applies only to that fact. Cogkura does not infer it from `source_type`, `source_ref`, `status`, `actor_id`, confidence, or prose.

A malformed `authority` object does not authorise the fact and does not drop an otherwise valid fact. Ordinary recurrence can still apply. `observe()` does not reject the observation for a bad authority block; extraction counts it as `authority_invalid` when the feature is enabled.

The calling application is the trust boundary. Core does not authenticate a person or verify a source. `cogkura-agent` must not translate a `confirmed` status into this declaration.

## Opt-in

```python
ComplementaryLearningSemanticConsolidator(
    minimum_supporting_episodes=2,
    allow_authoritative_single_episode=True,
)
```

The default is `False`. With the flag off, authority metadata is ignored, no extra revision reads are performed, and consolidation matches `0.18.0`.

## Revision gate

Fact values are still extracted from the current observation, as in `0.18.0`. Only the authority path reads the episode's cited revision through `ObservationStore.get_revisions_many`.

Admission requires all of the following:

- the cited revision row exists;
- that revision's `semantic_facts` contain the same canonical claim with `basis=application_asserted`;
- the cited revision is the observation's current revision;
- the observation is not deleted.

Anything else clears authority and increments `authority_invalid`. A later revision cannot authorise an episode that still cites an earlier one. Authority does not leak across facts, subjects, predicates, or tenants.

A new observation revision is still opened only when source version, content hash, deleted flag, or encoding context changes. A metadata-only resubmit stays `UNCHANGED`, so the declaration has to be stored on a revision that ingest actually writes.

## Admission and conflicts

A claim is promoted when either distinct supporting episodes meet `minimum_supporting_episodes`, or the authority flag is on and one verified binding exists. `support_count` is the real episode count. Confidence and importance use the existing formulas.

Within one episode, a verified authority candidate is kept ahead of a higher-confidence copy of the same claim. When neither candidate is authorised, the previous confidence comparison is unchanged.

If the claim already meets the recurrence threshold, the promotion basis is `recurrence` even when authority is also present.

Single-support authority does not enter reconciliation when it would compete:

- a cardinality-one authority claim against a different current, non-superseded claim in the same slot is deferred;
- two or more incompatible cardinality-one authority claims in one batch are all deferred;
- the same claim reinforces the existing memory;
- recurrence-qualified candidates, including ordinary corrections, still use `DeterministicSemanticReconciler`;
- cardinality-many authority claims follow existing coexistence.

Removing authority from a later source revision does not retract the semantic memory. Correction and withdrawal are `0.18.2`.

## Provenance and diagnostics

`SemanticPromotionBasis` is `recurrence` or `explicit_source_authority`. The record is stored at `metadata["semantic"]["promotion"]` on the semantic memory row. Authority records also keep the asserting observation id, revision, claim key, and the support count at admission. That block is not part of `content_fingerprint`, so an unchanged reprocess stays `UNCHANGED`. Later reinforcement keeps the original block.

`semantic_promotion_from_metadata()` reads the record. `Memory.resolve_provenance()` copies it onto `MemoryProvenance.promotion`. Source evidence and source diversity are unchanged. `MemoryContext.render()` does not describe authority.

`SemanticConsolidationResult` adds `recurrence_promoted`, `authority_promoted`, `authority_deferred_conflict`, and `authority_invalid` for the current pass. A repeated `process()` of one authoritative observation can report `authority_promoted=1` together with `unchanged=1`.

## Out of scope

No trust rankings, inferred authority, retrieval or activation boosts, new confidence weights, automatic supersession, retraction, new historical validity rules, schema migration, SQLite, or agent changes.
