# Cogkura 0.18.2 — Authoritative corrections and withdrawals

**Status:** Shipped  
**Baseline:** Cogkura `0.18.1`

`0.18.1` can adopt one authorised fact. It defers that fact when another current value already occupies the slot, and it does not treat a removed authority declaration as a withdrawal. `0.18.2` adds two explicit operations, both opt-in:

- **Replace** one named semantic revision with a new authorised assertion.
- **Withdraw** one named semantic revision without supplying a successor.

Authority still does not change confidence, activation, forgetting, interference, or inhibition. Historical observations are not deleted or rewritten.

## Enabling the path

Both flags are required for a single-support change:

```python
Memory(
    semantic_consolidator=ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True,
    ),
    semantic_correction_config=SemanticCorrectionConfig(enabled=True),
)
```

`SemanticCorrectionConfig(enabled=False)` is the default. While it is off, a `correction` object is ignored and `0.18.1` competing-authority deferral is unchanged.

## Replacement

On one `semantic_facts` entry:

```python
"authority": {"basis": "application_asserted"},
"correction": {
    "replaces_revision_key": "<revision from list_semantic_revisions()>",
    "effective_at": "<timezone-aware ISO-8601>",
}
```

The predecessor may have been promoted by recurrence or by explicit authority. The key is that revision's `revision_key`, not a document id. `generate_revision_key` is available when an application needs the deterministic key for a claim it is about to consolidate in the same batch.

A replacement is applied only when the declaration is on the exact current, non-deleted observation revision, the predecessor is the only current cardinality-one revision in that slot, the new claim is different, and `effective_at` is not earlier than the asserting observation and is after the predecessor's start. `process()` time is not the effective time.

The predecessor revision stays the same identity, becomes `SUPERSEDED`, and closes at `effective_at`. The successor is a new semantic memory, `ACTIVE` from `effective_at`, with its own evidence and support count. The relation is `SUPERSEDES`.

## Withdrawal

```python
"semantic_withdrawals": [{
    "target_revision_key": "<revision key>",
    "effective_at": "<timezone-aware ISO-8601>",
    "authority": {"basis": "application_asserted"},
}]
```

A withdrawal does not assert a new value and is not stored as `SUPERSEDED`. The original revision remains historically readable until `effective_at`. A `RETRACTED` revision becomes the current projection and is excluded from current answers. The relation is `RETRACTS`.

Editing source text, dropping `authority`, deleting a document, forgetting an episode, or falling below the retrieval threshold does not withdraw a semantic fact.

## History and knowledge time

`valid_at` selects the factual interval. Intervals are half-open: a query at `effective_at` returns the successor, and the predecessor matches only before that instant. `as_of` remains the cognitive clock. A correction or withdrawal whose asserting observation is later than `as_of` is not visible yet; the predecessor stays the known current value. Stored rows are not rewritten for that read.

Each new semantic revision stores a versioned snapshot (`semantic-revision-snapshot-v1`) with the statement, claim, evidence, promotion metadata, and any correction or withdrawal link. `valid_at` reconstruction uses that snapshot. Older revisions with an empty snapshot keep the previous current-row overlay and do not gain a fabricated authority explanation. There is no backfill.

`MemoryContext.render()` does not include correction administration. `resolve_provenance()` still reports the observation revisions on the selected memory, which now come from the historical snapshot when one exists.

Repeated processing of the same correction or withdrawal does not add another revision or relation.

## Diagnostics

`SemanticConsolidationResult` adds:

- `authoritative_corrections_applied`
- `authoritative_withdrawals_applied`
- `corrections_deferred`
- `corrections_invalid`

Applied counts are new durable writes on that pass. Ambiguous or competing declarations are deferred and do not partially supersede a predecessor.
