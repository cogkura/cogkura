# Cogkura — agent guide

Primary entry point for coding agents working in this repository.

## What this repo is

Cogkura is a research-driven cognitive memory library for AI systems (Python).
It sits between application data and LLM reasoning.

Cogkura owns observations, revisions, checkpoints, and (later) derived memories.
It does **not** own or modify customer application schemas.

- Current release focus: `0.15.2` long-horizon semantic recall
- Next: see [`docs/roadmap.md`](docs/roadmap.md) (Later milestones)

## Read first

1. [`AGENTS.md`](AGENTS.md) (this file)
2. [`README.md`](README.md)
3. [`docs/architecture.md`](docs/architecture.md)
4. [`docs/roadmap.md`](docs/roadmap.md)
5. [`docs/vision.md`](docs/vision.md)

## Layout

```text
src/cogkura/
  memory.py          # public facade
  observations/      # models, pipeline, policy, retention, hashing
  sources/           # SourceConnector + PostgresTableSource
  mappers/           # ObservationMapper protocol
  storage/           # ObservationStore, CheckpointStore, EpisodeStore, SemanticMemoryStore, ActivationStore, MemoryDynamicsStore, LearningStore
  migrations/        # Cogkura-owned Postgres schema SQL
  algorithms/        # episodic.py, semantic.py, activation.py, spreading.py, forgetting.py, working_memory.py, reconsolidation.py, learning.py, relevance.py, metamemory.py
tests/
examples/
  basic_memory.py
  working_memory.py
  reconsolidation.py
  learning.py
  metamemory.py
  application_context.py
  postgres_datasource/
```

## Public API (single path)

| API | Purpose |
|-----|---------|
| `observe(ObservationInput)` | Ingest one normalized observation |
| `ingest(source, mapper, tenant_id=...)` | Batch ingest from a source connector |
| `process(tenant_id=..., subject_id=..., as_of=...)` | Orchestrate episodic encoding and semantic consolidation |
| `prepare_context(query, tenant_id=..., goal=..., prompt_budget_tokens=...)` | Bounded working memory + metamemory assessment in one read |
| `record_context_use(context, ...)` | Record use of selected context memories (delegates to `record_access`) |
| `recall(query, tenant_id=..., valid_at=..., as_of=...)` | ACT-R declarative activation over episodic + semantic memories |
| `inspect_recall(query, tenant_id=..., valid_at=..., as_of=...)` | Bounded recall inspection with terminal dispositions and trace detail |
| `select_working_memory(query, tenant_id=..., goal=..., previous=...)` | Bounded goal-aware working-memory selection from recall candidates |
| `assess_memory(query, tenant_id=..., goal=..., valid_at=...)` | Read-only metamemory assessment; `MISSING_KNOWLEDGE` for unresolved slot-like queries or weak non-slot retrieval (`0.14.2+`) |
| `record_access(results, tenant_id=..., min_score=...)` | Explicitly reinforce used memories (reactivates forgotten dynamics); optional score floor and burst limits |
| `learn(feedback)` | Apply HELPFUL/UNHELPFUL/INCORRECT outcome feedback (idempotent by `feedback_id`) |
| `list_learning_state(tenant_id=..., identities=..., goal=...)` | Inspect persisted learning counts |
| `apply_forgetting(tenant_id=..., as_of=...)` | Evaluate forgetting lifecycle and compact old activation references |
| `encode_episodes(tenant_id=..., as_of=...)` | Build episodic memories from stored observations |
| `list_episodes(tenant_id=...)` | List encoded episodes for a tenant |
| `consolidate_semantics(tenant_id=..., as_of=...)` | Build semantic memories from active episodes (includes reconsolidation) |
| `list_semantic_memories(tenant_id=..., valid_at=...)` | List consolidated semantic memories for a tenant |
| `list_semantic_revisions(tenant_id=..., valid_at=...)` | List semantic revision history |
| `clear(tenant_id=...)` | Remove learning, activation refs, dynamics, semantic memories, episodes, observations |

There is **no** parallel `MemoryEvent` / string-`observe` path. Cognitive work builds on observations.

## PRD / Design Note vs codebase

When a PRD or Design Note initiates a feature:

- Use the PRD/Design Note for **concept, behavior, and acceptance criteria**.
- Use the **existing repo** for naming, package layout, protocols, and API shape.
- Prefer extending `storage/`, `algorithms/`, `observations/`, etc.
- Do **not** invent parallel trees from a PRD sketch (e.g. `stores/`, `cognitive/`) unless the user explicitly asks for a structural change.
- Docs describe what shipped; do not rewrite documentation to match a fictional PRD layout.

## Commands (match CI)

```bash
uv sync --all-extras --dev --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

Postgres integration tests (`@pytest.mark.postgres`) skip unless env URLs are set.
See [`examples/postgres_datasource/README.md`](examples/postgres_datasource/README.md).

After substantive code changes, run the validation commands above before finishing.

## Hard constraints

- Core package stays dependency-free; Postgres only via `cogkura[postgres]`.
- No arbitrary SQL as the primary source public API.
- Do not modify customer source tables.
- Do not advance connector checkpoints before a successful batch.
- Do not permit unscoped cross-tenant queries in the public API.
- Do not strip useful README/example/doc content without an explicit reason; prefer surgical edits over full-file rewrites.
- Do not commit or push unless the user asks.

## Style

- Frozen dataclasses, `typing.Protocol`, validation in constructors.
- Raise `ValidationError` / `StorageError` from `cogkura.exceptions`.
- Async for observation, ingest, recall, and Postgres paths.
- Match existing ruff/mypy settings in `pyproject.toml`.
