# Contributing to Cogkura

Contributions are welcome. For a proposed behavioral or architectural change, start by opening an issue or discussing the intended contract before implementing it. Review the [architecture](docs/architecture.md), [roadmap](docs/roadmap.md) and relevant design notes in the [documentation index](docs/README.md).

## Development setup

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras --dev --locked
```

Core has no required runtime dependencies. PostgreSQL support is optional through `cogkura[postgres]`; keep optional dependencies out of the default installation.

## Validation

Run the same checks used by the project:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

PostgreSQL integration tests require database URLs and are otherwise skipped. See the [PostgreSQL example setup](examples/postgres_datasource/README.md) for test databases, roles and environment variables. Report skipped tests rather than describing them as passing database coverage.

For changes to storage, memory behavior, or public contracts, include relevant regression tests and update the applicable design and findings. Preserve released benchmark gold and record any comparison differences explicitly. The independent [CogKuraBench](https://github.com/cogkura/cogkura-bench) project contains the benchmark workloads.

When sharing changes, describe the behavior, rationale, checks run and any remaining limitations. Releases, tags and package publication are handled separately from ordinary changes.
