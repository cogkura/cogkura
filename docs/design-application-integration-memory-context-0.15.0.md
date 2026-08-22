# CogKura 0.15.0 - Application integration and memory context

## Summary

CogKura `0.15.0` adds application-facing orchestration APIs that sit above the existing cognitive stack. Applications can process observations into recallable memory, prepare bounded assessed context for an external LLM, and record context use without CogKura owning model invocation.

## Added

- `Memory.process()` — episodic encoding and semantic consolidation with one shared `as_of` timestamp.
- `Memory.prepare_context()` — bounded working memory and metamemory assessment in one read operation.
- `MemoryContext` — immutable provider-neutral boundary with structured fields and deterministic `render()`.
- `Memory.record_context_use()` — records use of selected context memories via existing access semantics.
- Shared declarative retrieval inside `prepare_context()` (one rank pass per call).
- [`docs/application-integration.md`](application-integration.md) and [`examples/application_context.py`](../../examples/application_context.py).

## Behavioral constraints

- `observe()` / `ingest()` store observations only; recallable memory requires `process()` or explicit `encode_episodes()` / `consolidate_semantics()`.
- `prepare_context()` is presentation only; it does not write activation references, dynamics, semantic state, or learning state.
- `record_context_use()` reinforces only memories selected into working memory, not unselected recall candidates.
- Lower-level APIs (`recall()`, `select_working_memory()`, `assess_memory()`, `record_access()`) remain available and unchanged in contract.
- `MemoryContext` requires exact `subject_id` alignment across context, working memory, and assessment, including `None`, and `assessment.valid_at == context.valid_at`.
- Naive `valid_at` is rejected at the shared declarative ranking boundary for all retrieval APIs.

## `process()` deactivation

Episodic encoding deactivates episodes that no longer have backing observations. Episodes are marked inactive, not deleted.

- Tenant-wide `process(tenant_id=...)` deactivates stale episodes across the tenant.
- Subject-scoped `process(tenant_id=..., subject_id=...)` only deactivates stale episodes for that subject.
- `result.episodes.deactivated` reports inactive transitions during the call.

## Notes

- No migration required.
- `prepare_context()` raises when metamemory is disabled, matching `assess_memory()`.
- PyPI `Development Status` classifier updated from `2 - Pre-Alpha` to `4 - Beta`.
