# Cognema

Research-driven cognitive memory framework for AI systems.

## Tagline

> Research-driven cognitive memory framework for AI systems.

## Why Cognema exists

Most AI applications keep useful data, but retrieval is often shallow. You either do direct lookup, keyword search, or vector similarity, and then pass results to an LLM with little memory structure.

Cognema explores how research-backed cognitive memory mechanisms can improve how AI systems encode, consolidate, associate, and recall information.

## What Cognema is not

Cognema is not:

- a vector database;
- a RAG framework;
- an LLM provider;
- a hosted memory API;
- tied to one model, database, or agent framework.

## How Cognema differs

- Storage systems optimize persistence and querying.
- Vector search optimizes similarity matching.
- RAG frameworks optimize context assembly for prompts.

Cognema focuses on cognitive memory algorithms that sit between your data and your AI system.

You bring your own storage, ingestion, embeddings, and LLM provider. Cognema supplies memory behavior and orchestration.

## Installation

```bash
pip install cognema
```

## Quick start

```python
from cognema import Memory

memory = Memory()

event = memory.observe(
    "George discussed cognitive memory algorithms",
    metadata={"source": "conversation", "topic": "cognitive-memory"},
)

results = memory.recall("What was discussed about cognitive memory?")

for result in results:
    print(result.score, result.event.content, result.reason)

memory.sleep()
```

## Current status

Cognema is in early development. Version `0.0.1` is an initial foundation release intended to establish package structure, public API shape, and project standards.

## Scope of 0.0.1

Implemented in `0.0.1`:

- package structure and typed public API;
- `MemoryEvent` and `RecallResult` models;
- storage abstraction and in-memory backend;
- deterministic token-overlap recall;
- basic tests, CI workflows, and documentation.

Not implemented in `0.0.1`:

- episodic-to-semantic consolidation;
- spreading activation;
- memory decay and forgetting curves;
- goal-aware retrieval and working-memory selection.

## Long-term cognitive architecture

Target conceptual flow:

```text
Data and experiences
        ↓
Event encoding
        ↓
Episodic memory
        ↓
Semantic consolidation
        ↓
Associative world model
        ↓
Spreading activation
        ↓
Attention and goal filtering
        ↓
Working memory
        ↓
LLM reasoning and planning
```

## Roadmap

- `0.0.x`: foundation and package hardening.
- `0.1`: richer episodic memory and salience scoring.
- `0.2`: semantic consolidation pipelines.
- `0.3`: cognitive retrieval and working-memory selection.
- later: additional storage adapters, model interfaces, and integrations.

See [`docs/roadmap.md`](docs/roadmap.md) for details.

## Development setup with uv

```bash
uv sync --dev
```

## Validation commands

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

## Build commands

```bash
uv build
uvx twine check dist/*
```

## Contributing

Contributions are welcome. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md), then open an issue or pull request.

## Release and PyPI namespace claim notes

To secure the `cognema` PyPI namespace with Trusted Publishing:

1. Configure GitHub environment `pypi`.
2. Add a pending Trusted Publisher in PyPI for owner `cognema`, repo `cognema`, workflow `publish.yml`, environment `pypi`.
3. Publish `v0.0.1` through the GitHub release workflow.

A pending publisher does not reserve the package name until the first successful upload.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
