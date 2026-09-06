# Encoding context — 0.16.0

**Target release:** `0.16.0`  
**Status:** Shipped  
**Primary implementation:** [`src/cogkura/observations/encoding_context.py`](../src/cogkura/observations/encoding_context.py), [`src/cogkura/models.py`](../src/cogkura/models.py) (`MemoryContextSignature`), [`src/cogkura/algorithms/episodic.py`](../src/cogkura/algorithms/episodic.py)

## Summary

Cogkura `0.16.0` introduces explicit **encoding context**: structured circumstances present when observations are ingested and episodic memories are formed.

This release is **retrieval-neutral**. Encoding context is captured, normalised, aggregated, persisted, and inspectable. It does not alter recall, activation, admission, ranking, or working-memory selection.

## Public contract

- `ObservationContext` — optional on `ObservationInput`; persisted on `StoredObservation.context`
- `MemoryContextSignature` — on `EpisodeInput.encoding_context` and `StoredEpisode.encoding_context`
- Episode grouping/segmentation remains metadata-driven (`conversation_id`, `thread_id`, `session_id`, `episode_id` in observation metadata)

## Non-goals (0.16.0)

- context-aware retrieval or ranking
- context reinstatement scoring
- LLM-inferred context from observation text
- semantic-memory encoding-context aggregation

## Persistence

Postgres migration [`009_encoding_context.sql`](../src/cogkura/migrations/postgres/009_encoding_context.sql) adds `encoding_context JSONB` to observations, observation revisions, and memories. Legacy rows hydrate as empty signatures.

## Tests

- `tests/test_encoding_context_models.py`
- `tests/test_encoding_context_aggregation.py`
- `tests/test_encoding_context_persistence.py`
- `tests/test_encoding_context_lifecycle.py`
- `tests/test_016_encoding_context_retrieval_neutrality.py` (mandatory gate)

## Next

- **0.16.1** — retrieval context matching
- **0.16.2** — context reinstatement influences accessibility
