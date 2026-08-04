# Cognema — agent guide

Primary entry point for coding agents working in this repository.

## What this repo is

Cognema is a research-driven cognitive memory library for AI systems (Python).
It sits between application data and LLM reasoning.

Cognema owns observations, revisions, checkpoints, and (later) derived memories.
It does **not** own or modify customer application schemas.

- Current release focus: `0.2` episodic memory encoding + tenant-scoped recall placeholder
- Next: `0.3` semantic consolidation — see [`docs/roadmap.md`](docs/roadmap.md)

## Read first

1. [`AGENTS.md`](AGENTS.md) (this file)
2. [`README.md`](README.md)
3. [`docs/architecture.md`](docs/architecture.md)
4. [`docs/roadmap.md`](docs/roadmap.md)
5. [`docs/vision.md`](docs/vision.md)

## Layout

```text
src/cognema/
  memory.py          # public facade
  observations/      # models, pipeline, policy, retention, hashing
  sources/           # SourceConnector + PostgresTableSource
  mappers/           # ObservationMapper protocol
  storage/           # ObservationStore, CheckpointStore, EpisodeStore + backends
  migrations/        # Cognema-owned Postgres schema SQL
  algorithms/        # DeterministicEpisodicEncoder (episodic.py)
tests/
examples/
  basic_memory.py
  postgres_datasource/
docs/
```

## Public API (single path)

| API | Purpose |
|-----|---------|
| `observe(ObservationInput)` | Ingest one normalized observation |
| `ingest(source, mapper, tenant_id=...)` | Batch ingest from a source connector |
| `recall(query, tenant_id=...)` | Tenant-scoped placeholder retrieval over observations |
| `encode_episodes(tenant_id=...)` | Build episodic memories from stored observations |
| `list_episodes(tenant_id=...)` | List encoded episodes for a tenant |
| `clear(tenant_id=...)` | Remove episodes and observations for a tenant |

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

- Core package stays dependency-free; Postgres only via `cognema[postgres]`.
- No arbitrary SQL as the primary source public API.
- Do not modify customer source tables.
- Do not advance connector checkpoints before a successful batch.
- Do not permit unscoped cross-tenant queries in the public API.
- Do not strip useful README/example/doc content without an explicit reason; prefer surgical edits over full-file rewrites.
- Do not commit or push unless the user asks.

## Style

- Frozen dataclasses, `typing.Protocol`, validation in constructors.
- Raise `ValidationError` / `StorageError` from `cognema.exceptions`.
- Async for observation, ingest, recall, and Postgres paths.
- Match existing ruff/mypy settings in `pyproject.toml`.
