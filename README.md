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

Pass `as_of=` when replaying a frozen timeline; omit it for live encoding.

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

Pass the same `as_of=` used for encoding when consolidating a simulated timeline.

## Declarative activation (recall)

After encoding (and optionally consolidating), recall ranks episodic and semantic memories with ACT-R base-level, spreading activation, and partial matching. String queries seed spreading sources from cue tokens that overlap candidate entity ids; structured `RetrievalCue.entity_ids` still take precedence. Matching `ACTIVE` slot semantics (and their `SUPPORT` episodes) can be admitted before the retrieval threshold cut. Text matching downweights tokens that are common in the current candidate set. Near-duplicate statements are collapsed before the rank limit is applied, ignoring purely numeric tokens. Active semantic slot values are preferred over superseded ones; episodes that support a superseded slot are penalized on current-state cues.

`recall()` is presentation. `record_access()` records use.

```python
from datetime import UTC, datetime

from cogkura import ActivationConfig, RetrievalCue

results = await memory.recall(
    RetrievalCue(text="preferred database for production", subject_id="customer_42"),
    tenant_id="company_123",
)

# String cues can seed spreading from candidate entity overlap.
# Explicit entity_ids keep 0.11 associative behaviour.
results = await memory.recall(
    RetrievalCue(
        text="What database was involved?",
        entity_ids=("alice",),
    ),
    tenant_id="company_123",
)

# Historical recall: semantics use revision windows; episodes need started_at <= valid_at
as_of = datetime(2026, 1, 6, tzinfo=UTC)
results = await memory.recall(
    "What did we currently use for job coordination?",
    tenant_id="company_123",
    as_of=as_of,
    valid_at=as_of,
)

for result in results:
    print(result.activation, result.score, result.memory.statement)

# record_access is use, not presentation — filter weak rows when needed
await memory.record_access(results, tenant_id="company_123", min_score=0.5)

# Forgetting maintenance (explicit; sleep() is a no-op)
result = await memory.apply_forgetting(tenant_id="company_123", as_of=as_of)
```

For simulated replay, pass the same `as_of` to `encode_episodes()` and `consolidate_semantics()` so `created_at` is not wall clock. Live callers can omit it.

Tune retrieval with `activation_config=ActivationConfig(retrieval_threshold=-1.0)` on `Memory(...)`. See [`docs/design-ranking-time-current-state.md`](docs/design-ranking-time-current-state.md) and [`docs/design-string-cues-current-state.md`](docs/design-string-cues-current-state.md).

For PostgreSQL, pass `PostgresObservationStore`, `PostgresEpisodeStore`, `PostgresSemanticMemoryStore`, `PostgresActivationStore`, `PostgresMemoryDynamicsStore`, and `PostgresLearningStore` to `Memory`.

See [`docs/forgetting.md`](docs/forgetting.md) for lifecycle thresholds and compaction details.

## Metamemory (memory assessment)

`assess_memory()` reports the state of currently retrievable memory. It does not record access, apply forgetting, or create learning feedback, and it does not produce an overall confidence score.

```python
assessment = await memory.assess_memory(
    "What database did we select for production?",
    tenant_id="company_123",
    goal="Recall the production database decision.",
)

print(assessment.signals.cue_coverage)
print(assessment.signals.top_retrieval_strength)
print(assessment.signals.evidence_confidence)
print(assessment.signals.semantic_conflict)
print(assessment.flags)
```

See [`docs/metamemory.md`](docs/metamemory.md) and [`examples/metamemory.py`](examples/metamemory.py).

## Observation ingestion (PostgreSQL)

```python
from sqlalchemy.ext.asyncio import create_async_engine

from cogkura import Memory
from cogkura.sources.postgres import PostgresTableSource
from cogkura.storage.postgres import (
    PostgresActivationStore,
    PostgresCheckpointStore,
    PostgresEpisodeStore,
    PostgresLearningStore,
    PostgresMemoryDynamicsStore,
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
    dynamics_store=PostgresMemoryDynamicsStore(memory_engine),
    learning_store=PostgresLearningStore(memory_engine),
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

Cogkura is in early development. Through `0.12.0`, the library provides observation ingestion, episodic encoding, semantic consolidation with temporal reconsolidation, ACT-R declarative activation, spreading activation, Ebbinghaus-inspired forgetting dynamics, bounded working-memory selection, outcome-driven learning via `Memory.learn()`, and read-only metamemory assessment via `Memory.assess_memory()`, with explicit `record_access()` reinforcement (presentation vs use), `apply_forgetting()` maintenance, simulated `as_of` on encode/consolidate, episode `valid_at` filtering, candidate-set IDF ranking, near-duplicate collapse, current-state semantic bias, string-cue entity seeding, semantic slot admission, and superseded-slot episode penalties.

## Scope of 0.12.0

Implemented through `0.12.0`:

- observation models and ingestion pipeline (`0.1`);
- `ObservationStore` and `CheckpointStore` protocols with in-memory and PostgreSQL backends (`0.1`);
- `PostgresTableSource` with compound cursor pagination (`0.1`);
- `Memory.observe()`, `Memory.ingest()`, revision history, and tenant-scoped storage (`0.1`);
- deterministic episodic encoding, `Memory.encode_episodes()`, and `Memory.list_episodes()` (`0.2`);
- semantic consolidation, `Memory.consolidate_semantics()`, and `Memory.list_semantic_memories()` (`0.3`);
- ACT-R declarative activation, `Memory.recall()` over episodic + semantic memories, and `Memory.record_access()` (`0.4`);
- spreading activation with structured `RetrievalCue.entity_ids` (`0.5`);
- forgetting lifecycle, `Memory.apply_forgetting()`, weighted reference compaction, and `include_forgotten` on recall (`0.6`);
- bounded working-memory selection, `Memory.select_working_memory()`, goal relevance, inhibition, and prompt budgeting (`0.7`);
- temporal semantic reconsolidation, revision history, `Memory.list_semantic_revisions()`, and `valid_at` historical retrieval (`0.8`);
- outcome-driven learning, `Memory.learn()`, contextual utility, HELPFUL ACT-R traces, and learned associations (`0.9`);
- read-only metamemory, `Memory.assess_memory()`, independent monitoring signals, and diagnostic flags (`0.10`);
- simulated `as_of` on encode/consolidate, episode `valid_at` visibility, candidate-set IDF, near-duplicate collapse, current-state ranking, and importance-aware forgetting (`0.11`);
- string-cue entity seeding, semantic slot admission, superseded-support penalties, numeric duplicate collapse, and `record_access(..., min_score=...)` (`0.12`);
- Docker PostgreSQL example with seed and mutation scripts;
- unit tests and optional PostgreSQL integration tests.

Not implemented in `0.12.0`:

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
Goal relevance + inhibition
        ↓
Bounded working memory
        ↓
Memory assessment
        ↓
LLM reasoning and planning
        ↓
Outcome feedback
        ↓
Learning / reinforcement
```

## Roadmap

- `0.1`: PostgreSQL observation ingestion and provenance.
- `0.2`: episodic memory encoding, salience, temporal context, and evidence links (done).
- `0.3`: semantic consolidation from episodic memories (done).
- `0.4`: declarative activation (ACT-R recall over episodic + semantic memories) (done).
- `0.5`: spreading activation (done).
- `0.6`: forgetting / memory dynamics (done).
- `0.7`: working-memory selection and inhibition (done).
- `0.8`: temporal reconsolidation and memory updating (done).
- `0.9`: learning and reinforcement (done).
- `0.10`: metamemory / memory monitoring (done).
- `0.11`: ranking, simulated time, and current-state recall (done).
- `0.12`: string cues, slot admission, and access recording (done).
- later: additional connectors, and integrations.

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
