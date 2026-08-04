# Cognema

Research-driven cognitive memory framework for AI systems.

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

Cognema owns observations and derived memories, not customer application records. Source connectors read customer data; Cognema writes only to Cognema-owned storage.

## Installation

```bash
pip install cognema
```

PostgreSQL support:

```bash
pip install "cognema[postgres]"
```

## Quick start (transitional recall)

```python
from cognema import Memory

memory = Memory()

event = memory.observe(
    "George discussed cognitive memory algorithms",
    metadata={"source": "conversation", "topic": "cognitive-memory"},
    tags=["research", "memory"],
)

results = memory.recall("What was discussed about cognitive memory?")

for result in results:
    print(result.score, result.event.content, result.reason)

memory.sleep()
```

## Observation ingestion (PostgreSQL)

```python
from sqlalchemy.ext.asyncio import create_async_engine

from cognema import Memory
from cognema.sources.postgres import PostgresTableSource
from cognema.storage.postgres import PostgresCheckpointStore, PostgresObservationStore

memory_engine = create_async_engine("postgresql+asyncpg://...")
source_engine = create_async_engine("postgresql+asyncpg://...")

memory = Memory(
    observation_store=PostgresObservationStore(memory_engine),
    checkpoint_store=PostgresCheckpointStore(memory_engine),
)

source = PostgresTableSource(
    connector_id="application-messages",
    engine=source_engine,
    table="public.messages",
    cursor_columns=("updated_at", "id"),
)

result = await memory.ingest(
    source=source,
    mapper=MessageMapper("company_123"),
    tenant_id="company_123",
)
```

Direct observation:

```python
from datetime import UTC, datetime

from cognema import ObservationInput

status = await memory.observe_input(
    ObservationInput(
        tenant_id="company_123",
        subject_id="user_456",
        source_namespace="chat.messages",
        source_record_id="message_789",
        source_version="1",
        event_type="message",
        content="I prefer PostgreSQL for production services.",
        observed_at=datetime.now(UTC),
    )
)
```

See [`examples/postgres_datasource/README.md`](examples/postgres_datasource/README.md) for the Docker-based demo.

## Current status

Cognema is in early development. Version `0.1.0` adds PostgreSQL observation ingestion while keeping transitional token-overlap recall via `observe()` / `recall()`.

## Scope of 0.1.0

Implemented in `0.1.0`:

- observation models and ingestion pipeline;
- `ObservationStore` and `CheckpointStore` protocols;
- PostgreSQL schema, migrations, and stores;
- `PostgresTableSource` with compound cursor pagination;
- `Memory.observe()`, `Memory.observe_input()`, and `Memory.ingest()`;
- revision history for create, update, delete, and restore;
- Docker PostgreSQL example with seed and mutation scripts;
- unit tests and optional PostgreSQL integration tests.

Not implemented in `0.1.0`:

- episodic-to-semantic consolidation;
- spreading activation;
- memory decay and forgetting curves;
- goal-aware retrieval and working-memory selection;
- full REDACTED / REFERENCE_ONLY retention modes;
- non-PostgreSQL source connectors.

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

- `0.1`: PostgreSQL observation ingestion and provenance.
- `0.2`: episodic memory encoding, salience, and temporal context.
- `0.3`: semantic consolidation pipelines.
- `0.4`: cognitive retrieval and working-memory selection.
- later: additional source connectors, model interfaces, and integrations.

See [`docs/roadmap.md`](docs/roadmap.md) and [`docs/architecture.md`](docs/architecture.md) for details.

## Development setup with uv

```bash
uv sync --all-extras --dev
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

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
