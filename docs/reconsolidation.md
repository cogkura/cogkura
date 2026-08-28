# Reconsolidation and memory updating

Target release: 0.8  
Status: Shipped

## Summary

Cogkura `0.8.0` adds **temporal semantic reconsolidation**: revision history, deterministic relation classification (`REINFORCES`, `COEXISTS`, `SUPERSEDES`, `CONFLICTS`), and historical retrieval via `valid_at`. Consolidation still runs through `Memory.consolidate_semantics()`; there is no separate `reconsolidate()` API.

`observed_at` never implies world validity when explicit `valid_from` / `valid_until` are present. For cardinality-one competitors with **unspecified** world windows, supporting evidence chronology (`last_supported_at`, then `first_supported_at`) determines supersession; equal evidence times still yield `CONFLICTS`, not latest-write-wins supersession.

## Pipeline

```text
extract semantic facts
  → consolidate revision candidates
  → load existing memories + revisions
  → DeterministicSemanticReconciler
  → SemanticReconciliationPlan
  → SemanticMemoryStore.apply_reconciliation() (atomic)
```

## Temporal validity on observations

Structured `valid_from` / `valid_until` may appear on `semantic_facts` entries (ISO-8601, timezone-aware). Naive or malformed values are rejected. Natural-language date extraction is out of scope.

## Relation matrix (deterministic)

| Situation | Relation |
|-----------|----------|
| Same proposition, compatible validity | `REINFORCES` |
| `MANY` cardinality or distinct slots | `COEXISTS` |
| Sequential non-overlapping ONE competitors | `SUPERSEDES` |
| Unspecified ONE competitors with ordered evidence chronology | `SUPERSEDES` |
| Overlap, tied evidence, or unknown chronology for ONE competitors | `CONFLICTS` |

Supersession closes the predecessor interval (`valid_until = successor.valid_from`), marks the predecessor `SUPERSEDED`, and preserves its confidence.

## Public API

- `Memory.consolidate_semantics()` — now reconciles revisions and writes history (no `deactivate_missing` on this path).
- `Memory.list_semantic_revisions(...)` — revision history for a tenant.
- `valid_at: datetime | None` on `list_semantic_memories()`, `recall()`, and `select_working_memory()`.
  - `valid_at=None`: current projection; excludes `SUPERSEDED`.
  - `valid_at=<timestamp>`: half-open interval match; includes historical superseded revisions.
- `as_of` remains cognitive time (activation / forgetting), independent of `valid_at`.

## Storage

- In-memory store: `_revisions`, `_relations`, `apply_reconciliation()`.
- PostgreSQL migration `006_semantic_reconsolidation.sql`:
  - `semantic_claim_revisions`
  - `semantic_claims.current_revision_key`
  - `semantic_revision_relations` (persists `SUPERSEDES` / `CONFLICTS` only)
  - `memory_derivations.revision_key` with legacy backfill (`legacy:{memory_id}`)
  - legacy semantic rows: world `valid_from` / `valid_until` reset to `NULL`

## Non-goals (0.8)

No LLM/embeddings, NL temporal extraction, source trust learning, ACT-R equation changes, or forgetting equation changes. Explicit learning/reinforcement on reconsolidation remains planned for `0.9`.

## Example

See [`examples/reconsolidation.py`](../examples/reconsolidation.py).
