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

## Quick start

```python
import asyncio
from datetime import UTC, datetime

from cognema import Memory, ObservationInput


async def main() -> None:
    memory = Memory()
    tenant_id = "local"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            source_namespace="direct",
            source_record_id="1",
            content="George discussed cognitive memory algorithms",
            observed_at=datetime.now(UTC),
            metadata={"source": "conversation", "tags": ["research", "memory"]},
        )
    )

    results = await memory.recall(
        "What was discussed about cognitive memory?",
        tenant_id=tenant_id,
    )

    for result in results:
        print(result.score, result.observation.content, result.reason)

    memory.sleep()


asyncio.run(main())
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

status = await memory.observe(
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

See [`examples/postgres_datasource/README.md`](examples/postgres_datasource/README.md) for the full Docker-based demo.

### Postgres example environment

Unit tests and the basic in-memory example do not need Docker or env vars.

For the Postgres demo and `@pytest.mark.postgres` integration tests:

```bash
cd examples/postgres_datasource
docker compose up -d
cp .env.example .env
```

Example `.env` (also in [`.env.example`](examples/postgres_datasource/.env.example)):

```bash
# Read-only source DB (demo + most integration tests)
COGNEMA_POSTGRES_SOURCE_URL=postgresql+asyncpg://cognema_reader:cognema_reader@localhost:5432/cognema_source

# Cognema write DB (demo + most integration tests)
COGNEMA_POSTGRES_MEMORY_URL=postgresql+asyncpg://cognema_writer:cognema_writer@localhost:5432/cognema_memory

# Optional: write access for mutate.py / admin test inserts
COGNEMA_POSTGRES_SOURCE_ADMIN_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cognema_source

# Optional: same-DB schema mode tests
COGNEMA_POSTGRES_SAME_DB_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cognema_source
```

Load the file into your shell before running the demo or Postgres tests:

```bash
set -a && source examples/postgres_datasource/.env && set +a
uv run python examples/postgres_datasource/demo.py
uv run pytest -m postgres
```

`mutate.py` needs write access to the source database. Prefer `COGNEMA_POSTGRES_SOURCE_ADMIN_URL`, or run with the script default (`postgres` on `cognema_source`), not the read-only `cognema_reader` URL.

## Current status

Cognema is in early development. Version `0.1.0` uses a single observation-based API for ingest and tenant-scoped recall (token-overlap placeholder until cognitive retrieval).

## Scope of 0.1.0

Implemented in `0.1.0`:

- observation models and ingestion pipeline;
- `ObservationStore` and `CheckpointStore` protocols;
- in-memory and PostgreSQL observation stores;
- `PostgresTableSource` with compound cursor pagination;
- `Memory.observe()`, `Memory.ingest()`, and tenant-scoped `Memory.recall()`;
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

Agent and editor guidance lives in [`AGENTS.md`](AGENTS.md) (primary). [`CLAUDE.md`](CLAUDE.md) points there.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
