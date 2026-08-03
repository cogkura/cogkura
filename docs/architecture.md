# Architecture

## Design direction

Cognema is designed as a thin cognitive layer between application data and AI reasoning components.

The architecture separates concerns so applications can keep their own persistence and model infrastructure.

## Layers

### Public API

The public package API starts with:

- `Memory`
- `MemoryEvent`
- `RecallResult`

This API should remain stable even as internals evolve.

### Memory models

`MemoryEvent` and `RecallResult` capture event observations and scored recall outputs.

These models are typed and validated to keep behavior explicit.

### Storage adapters

Storage is abstracted behind a small protocol:

- `store(event)`
- `get(event_id)`
- `list()`
- `clear()`

`0.0.1` implements only `InMemoryStorage`.

SQLite and PostgreSQL adapters are planned and should not require public API changes.

### Cognitive algorithms

Future modules will implement consolidation, decay, attention, and association mechanics.

In `0.0.1`, this layer is intentionally a stub.

### Retrieval

`0.0.1` uses deterministic token overlap scoring as a placeholder retrieval strategy.

It is dependency-free and transparent so it can be replaced by cognitive retrieval methods in later versions.

### Embeddings and LLM integrations

Embedding providers and LLM providers are planned integration points, not implemented features in `0.0.1`.

The goal is for Cognema to orchestrate memory behavior without forcing specific providers.

## Current implementation boundary

Implemented now:

- public API;
- core models and exceptions;
- storage abstraction and in-memory backend;
- deterministic recall;
- tests and CI.

Planned later:

- episodic and semantic memory subsystems;
- consolidation pipelines;
- spreading activation and associative recall;
- goal-aware filtering and working-memory selection;
- external adapters and integrations.
