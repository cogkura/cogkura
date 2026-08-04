"""Unit tests for Memory ingestion API."""

from datetime import UTC, datetime

import pytest

from cognema.exceptions import ValidationError
from cognema.memory import Memory
from cognema.observations.models import IngestStatus, ObservationInput
from cognema.storage.in_memory_observation import InMemoryCheckpointStore, InMemoryObservationStore


class _ListSource:
    connector_id = "test-source"

    def __init__(self, records: list[dict[str, object]]) -> None:
        self._records = records

    def records(self, checkpoint: dict[str, object] | None):  # noqa: ANN201
        async def _iterate():  # noqa: ANN202
            for record in self._records:
                yield record

        return _iterate()

    def checkpoint_for(self, record: dict[str, object]) -> dict[str, object]:
        updated_at = record["updated_at"]
        if hasattr(updated_at, "isoformat"):
            ts_str = updated_at.isoformat()  # type: ignore[union-attr]
        else:
            ts_str = str(updated_at)
        return {"updated_at": ts_str, "id": str(record["id"])}


class _Mapper:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def map(self, record: dict[str, object]) -> ObservationInput:
        return ObservationInput(
            tenant_id=self.tenant_id,
            subject_id="user_1",
            source_namespace="public.messages",
            source_record_id=str(record["id"]),
            source_version=str(record["updated_at"]),
            event_type="message",
            content=str(record["body"]),
            observed_at=record["updated_at"],  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_ingest_advances_checkpoint() -> None:
    ts = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    source = _ListSource(
        [
            {"id": "1", "body": "George prefers PostgreSQL.", "updated_at": ts},
            {"id": "2", "body": "ok", "updated_at": ts},
        ]
    )
    observation_store = InMemoryObservationStore()
    checkpoint_store = InMemoryCheckpointStore()
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=checkpoint_store,
    )
    result = await memory.ingest(
        source=source,
        mapper=_Mapper("company_123"),
        tenant_id="company_123",
        batch_size=10,
    )
    assert result.discovered == 2
    assert result.created == 1
    assert result.rejected == 1
    checkpoint = await checkpoint_store.get(tenant_id="company_123", connector_id="test-source")
    assert checkpoint == {"updated_at": ts.isoformat(), "id": "2"}


@pytest.mark.asyncio
async def test_observe_creates_observation() -> None:
    memory = Memory()
    obs = ObservationInput(
        tenant_id="company_123",
        source_namespace="public.messages",
        source_record_id="1",
        content="George prefers PostgreSQL.",
        observed_at=datetime.now(UTC),
    )
    status = await memory.observe(obs)
    assert status is IngestStatus.CREATED


@pytest.mark.asyncio
async def test_observe_rejects_short_content() -> None:
    memory = Memory()
    obs = ObservationInput(
        tenant_id="company_123",
        source_namespace="public.messages",
        source_record_id="1",
        content="ok",
        observed_at=datetime.now(UTC),
    )
    with pytest.raises(ValidationError, match="rejected by policy"):
        await memory.observe(obs)
