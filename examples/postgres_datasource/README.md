# PostgreSQL datasource example

This example runs PostgreSQL in Docker with:

- `cognema_source`: customer application schema (`public.*`)
- `cognema_memory`: Cognema-owned `cognema` schema

## Setup

```bash
cd examples/postgres_datasource
docker compose up -d
cp .env.example .env
```

Install Cognema with PostgreSQL support:

```bash
uv sync --all-extras --dev
```

## Run

From the repository root:

```bash
uv run python examples/postgres_datasource/demo.py
uv run python examples/postgres_datasource/demo.py
uv run python examples/postgres_datasource/scripts/mutate.py
uv run python examples/postgres_datasource/demo.py
```

Or, if you are already in `examples/postgres_datasource`:

```bash
uv run python demo.py
uv run python demo.py
uv run python scripts/mutate.py
uv run python demo.py
```

Expected pattern:

1. First run: `created > 0`
2. Second run without mutation: no new discoveries (checkpoint at end)
3. After mutation: `created > 0`, `updated > 0`, `deleted > 0`

## Roles

- `cognema_reader`: read-only on source tables
- `cognema_writer`: read/write on `cognema.*` in the memory database

## Reset

```bash
./scripts/reset.sh
```

## Integration tests

```bash
export COGNEMA_POSTGRES_SOURCE_URL=postgresql+asyncpg://cognema_reader:cognema_reader@localhost:5432/cognema_source
export COGNEMA_POSTGRES_MEMORY_URL=postgresql+asyncpg://cognema_writer:cognema_writer@localhost:5432/cognema_memory
# Optional for same-DB / admin mutation tests:
# export COGNEMA_POSTGRES_SOURCE_ADMIN_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cognema_source
# export COGNEMA_POSTGRES_SAME_DB_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cognema_source
uv run pytest -m postgres
```
