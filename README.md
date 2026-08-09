# Cogkura

Research-driven cognitive memory framework for AI systems.

## Why Cogkura exists

Most AI applications keep useful data, but retrieval is often shallow. You either do direct lookup, keyword search, or vector similarity, and then pass results to an LLM with little memory structure.

Cogkura explores how research-backed cognitive memory mechanisms can improve how AI systems encode, consolidate, associate, and recall information.

## What Cogkura is not

Cogkura is not:

- a vector database;
- a RAG framework;
- an LLM provider;
- a hosted memory API;
- tied to one model, database, or agent framework.

## How Cogkura differs

- Storage systems optimize persistence and querying.
- Vector search optimizes similarity matching.
- RAG frameworks optimize context assembly for prompts.

Cogkura focuses on cognitive memory algorithms that sit between your data and your AI system.

You bring your own storage, ingestion, embeddings, and LLM provider. Cogkura supplies memory behavior and orchestration.

Cogkura owns observations and derived memories, not customer application records. Source connectors read customer data; Cogkura writes only to Cogkura-owned storage.

## Installation

```bash
pip install cogkura
```

PostgreSQL support:

```bash
pip install "cogkura[postgres]"
```

## Quick start

```python
import asyncio
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput


async def main() -> None:
    memory = Memory()
    tenant_id = "local"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="george",
            source_namespace="direct",
            source_record_id="1",
            content="George discussed cognitive memory algorithms",
            observed_at=datetime.now(UTC),
            metadata={"conversation_id": "research", "source": "conversation"},
        )
    )

    await memory.encode_episodes(tenant_id=tenant_id)

    results = await memory.recall(
        "What was discussed about cognitive memory?",
        tenant_id=tenant_id,
    )

    for result in results:
        print(result.score, result.memory.statement, result.reason)

    memory.sleep()


asyncio.run(main())
```

## Episodic memory encoding

After observations are stored, encode them into context-bound episodes:

```python
from datetime import UTC, datetime

from cogkura import Memory, ObservationInput

memory = Memory()

await memory.observe(
    ObservationInput(
        tenant_id="company_123",
        subject_id="customer_42",
        source_namespace="direct",
        source_record_id="message_1",
        content="Redis would add too much operational complexity.",
        observed_at=datetime.now(UTC),
        metadata={"conversation_id": "architecture_123"},
    )
)

result = await memory.encode_episodes(tenant_id="company_123", subject_id="customer_42")
episodes = await memory.list_episodes(tenant_id="company_123", subject_id="customer_42")

print(result.created, len(episodes[0].evidence))
```

## Semantic consolidation

Attach structured facts to observation metadata, encode episodes, then consolidate:

```python
semantic_fact = {
    "predicate": "preferred_database",
    "object_value": "postgresql",
    "object_entity_id": "postgresql",
    "cardinality": "one",
    "polarity": "affirm",
    "qualifiers": {"environment": "production"},
}

await memory.observe(
    ObservationInput(
        tenant_id="company_123",
        subject_id="customer_42",
        source_namespace="direct",
        source_record_id="message_1",
        content="PostgreSQL fits our operational constraints.",
        observed_at=datetime.now(UTC),
        metadata={
            "conversation_id": "architecture_123",
            "semantic_facts": [semantic_fact],
        },
    )
)

await memory.encode_episodes(tenant_id="company_123", subject_id="customer_42")
result = await memory.consolidate_semantics(tenant_id="company_123", subject_id="customer_42")
memories = await memory.list_semantic_memories(tenant_id="company_123", subject_id="customer_42")

print(result.promoted, memories[0].statement)
```

## Declarative activation (recall)

After encoding (and optionally consolidating), recall ranks episodic and semantic memories with ACT-R base-level and partial matching:

```python
from cogkura import ActivationConfig, RetrievalCue

results = await memory.recall(
    RetrievalCue(text="preferred database for production", subject_id="customer_42"),
    tenant_id="company_123",
)

for result in results:
    print(result.activation, result.score, result.memory.statement)

await memory.record_access(results, tenant_id="company_123")
```

Tune retrieval with `activation_config=ActivationConfig(retrieval_threshold=-1.0)` on `Memory(...)`.

For PostgreSQL, pass `PostgresObservationStore`, `PostgresEpisodeStore`, `PostgresSemanticMemoryStore`, and `PostgresActivationStore` to `Memory`.

## Observation ingestion (PostgreSQL)

```python
from sqlalchemy.ext.asyncio import create_async_engine

from cogkura import Memory
from cogkura.sources.postgres import PostgresTableSource
from cogkura.storage.postgres import (
    PostgresActivationStore,
    PostgresCheckpointStore,
    PostgresEpisodeStore,
    PostgresObservationStore,
    PostgresSemanticMemoryStore,
)

memory_engine = create_async_engine("postgresql+asyncpg://...")
source_engine = create_async_engine("postgresql+asyncpg://...")

memory = Memory(
    observation_store=PostgresObservationStore(memory_engine),
    checkpoint_store=PostgresCheckpointStore(memory_engine),
    episode_store=PostgresEpisodeStore(memory_engine),
    semantic_store=PostgresSemanticMemoryStore(memory_engine),
    activation_store=PostgresActivationStore(memory_engine),
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

from cogkura import ObservationInput

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
COGKURA_POSTGRES_SOURCE_URL=postgresql+asyncpg://cogkura_reader:cogkura_reader@localhost:5432/cogkura_source

# Cogkura write DB (demo + most integration tests)
COGKURA_POSTGRES_MEMORY_URL=postgresql+asyncpg://cogkura_writer:cogkura_writer@localhost:5432/cogkura_memory

# Optional: write access for mutate.py / admin test inserts
COGKURA_POSTGRES_SOURCE_ADMIN_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_source

# Optional: owner role for schema migrations / upgrade tests
COGKURA_POSTGRES_MEMORY_ADMIN_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_memory

# Optional: same-DB schema mode tests
COGKURA_POSTGRES_SAME_DB_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_source
```

Load the file into your shell before running the demo or Postgres tests:

```bash
set -a && source examples/postgres_datasource/.env && set +a
uv run python examples/postgres_datasource/demo.py
uv run pytest -m postgres
```

`mutate.py` needs write access to the source database. Prefer `COGKURA_POSTGRES_SOURCE_ADMIN_URL`, or run with the script default (`postgres` on `cogkura_source`), not the read-only `cogkura_reader` URL.

## Current status

Cogkura is in early development. Through `0.4.0`, the library provides observation ingestion, episodic encoding, semantic consolidation, and ACT-R declarative activation over memories, with explicit `record_access()` reinforcement.

## Scope of 0.4.0

Implemented through `0.4.0`:

- observation models and ingestion pipeline (`0.1`);
- `ObservationStore` and `CheckpointStore` protocols with in-memory and PostgreSQL backends (`0.1`);
- `PostgresTableSource` with compound cursor pagination (`0.1`);
- `Memory.observe()`, `Memory.ingest()`, revision history, and tenant-scoped storage (`0.1`);
- deterministic episodic encoding, `Memory.encode_episodes()`, and `Memory.list_episodes()` (`0.2`);
- semantic consolidation, `Memory.consolidate_semantics()`, and `Memory.list_semantic_memories()` (`0.3`);
- ACT-R declarative activation, `Memory.recall()` over episodic + semantic memories, and `Memory.record_access()` (`0.4`);
- Docker PostgreSQL example with seed and mutation scripts;
- unit tests and optional PostgreSQL integration tests.

Not implemented in `0.4.0`:

- spreading activation (planned `0.5`);
- memory decay and forgetting curves (`0.6`);
- goal-aware retrieval and working-memory selection (`0.7`);
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
- `0.2`: episodic memory encoding, salience, temporal context, and evidence links (done).
- `0.3`: semantic consolidation from episodic memories (done).
- `0.4`: declarative activation (ACT-R recall over episodic + semantic memories) (done).
- `0.5`: spreading activation.
- later: forgetting dynamics, working-memory selection, additional connectors, and integrations.

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
