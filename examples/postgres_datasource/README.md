# PostgreSQL datasource example

This example runs PostgreSQL in Docker with:

- `cogkura_source`: customer application schema (`public.*`)
- `cogkura_memory`: Cogkura-owned `cogkura` schema

## Setup

```bash
cd examples/postgres_datasource
docker compose up -d
cp .env.example .env
```

Install Cogkura with PostgreSQL support:

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

- `cogkura_reader`: read-only on source tables
- `cogkura_writer`: read/write on `cogkura.*` in the memory database

## Reset

```bash
./scripts/reset.sh
```

## Integration tests

```bash
export COGKURA_POSTGRES_SOURCE_URL=postgresql+asyncpg://cogkura_reader:cogkura_reader@localhost:5432/cogkura_source
export COGKURA_POSTGRES_MEMORY_URL=postgresql+asyncpg://cogkura_writer:cogkura_writer@localhost:5432/cogkura_memory
# Optional for same-DB / admin mutation tests:
# export COGKURA_POSTGRES_SOURCE_ADMIN_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_source
# export COGKURA_POSTGRES_SAME_DB_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_source
uv run pytest -m postgres
```
