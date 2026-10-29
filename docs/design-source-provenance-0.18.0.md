# Source identity and provenance monitoring (0.18.0)

Cogkura 0.18.0 adds revision-aware source attribution as a read-only Core capability. It answers where the evidence for a prepared context originated without assigning source authority or changing retrieval.

## Public contract

Applications first prepare normal memory context, then resolve provenance explicitly:

```python
context = await memory.prepare_context(
    "What is the deployment policy?",
    tenant_id="repository",
)
provenance = await memory.resolve_provenance(context)
```

`ContextProvenance.items` contains a `MemoryProvenance` record for every selected primary representation and every included chunk member. `SourceEvidence` contains the observation ID and the exact evidence revision, stable source identity (`source_namespace`, `source_record_id`), the revision's `source_version`, optional `metadata["source_ref"]`, observation time, contribution, and known relation.

The resolver returns only this allow-listed attribution data. It never returns observation content or arbitrary metadata, follows no source reference, and makes no external call.

## Historical semantics

`ObservationStore.get_revisions_many()` resolves `(observation_id, observation_revision)` in a tenant-scoped batch. It reads the matching revision row, not the observation's current values. `source_namespace` and `source_record_id` are stable source identity; revision metadata, source version, and observed time are historical fields. Current-only fields such as `source_type` are not presented as historical facts.

An unavailable historical revision is `REVISION_UNAVAILABLE`; it is never replaced with the latest revision. A deleted current source remains distinguishable from an absent source because historical revision rows can still resolve. `is_latest_known_revision` only compares the evidence revision to the current observation revision. It is not a truth, freshness, or authority judgement.

`metadata["source_ref"]` is the supported optional external locator convention. It is untrusted descriptive data, not a path to open or an instruction to fetch. A missing locator yields `SOURCE_REF_MISSING` while retaining the resolved source identity and revision evidence.

## Memory representation and relationships

Episode evidence has `DIRECT_EPISODIC_EVIDENCE`. Semantic evidence is labelled `SEMANTIC_SUPPORT` or `SEMANTIC_CONTRADICTION` only when its existing episode derivation proves that relation; otherwise the relation is absent.

`MemoryProvenance.representation` separates `SELECTED_RENDERED` primary memories from `CHUNK_MEMBER_NOT_INDEPENDENTLY_RENDERED`. A supporting episode can therefore be reported as a chunk member without being claimed to have appeared independently in `MemoryContext.render()`.

Source diversity counts distinct `(source_namespace, source_record_id)` pairs, not episodes or database IDs. More sources and a newer source revision do not confer authority. Authority-aware promotion, correction, and ranking remain out of scope.

## Bounds and non-interference

`resolve_provenance(..., max_evidence=256)` applies a deterministic global cap in prepared-context order. It preserves each memory's total referenced-evidence count and explicitly sets truncation warnings rather than silently omitting sources. Revision lookups are deduplicated and batched.

The resolver is downstream of `prepare_context()`. It does not modify declarative activation, relevance, ranking, working-memory selection, consolidation, reconsolidation, learning, inhibition, interference, or render output.
