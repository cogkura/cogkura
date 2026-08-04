from datetime import UTC, datetime

import pytest

from cognema import Memory, ObservationInput
from cognema.exceptions import ValidationError
from cognema.observations.models import IngestStatus


def _obs(
    content: str,
    *,
    record_id: str,
    tenant_id: str = "local",
) -> ObservationInput:
    return ObservationInput(
        tenant_id=tenant_id,
        source_namespace="direct",
        source_record_id=record_id,
        content=content,
        observed_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_observe_stores_observation() -> None:
    memory = Memory()

    status = await memory.observe(_obs("Cognema explores cognitive recall.", record_id="1"))

    assert status is IngestStatus.CREATED
    results = await memory.recall("cognitive", tenant_id="local")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_recall_matching_observations() -> None:
    memory = Memory()
    await memory.observe(_obs("George discussed cognitive memory algorithms.", record_id="1"))
    await memory.observe(_obs("A weather report from London.", record_id="2"))

    results = await memory.recall("cognitive memory", tenant_id="local")

    assert len(results) == 1
    assert results[0].observation.content == "George discussed cognitive memory algorithms."
    assert results[0].score > 0.0


@pytest.mark.asyncio
async def test_recall_is_deterministic_for_ties() -> None:
    memory = Memory()
    await memory.observe(_obs("alpha beta", record_id="a"))
    await memory.observe(_obs("alpha beta", record_id="b"))

    results = await memory.recall("alpha beta", tenant_id="local", limit=2)
    result_ids = [result.observation.id for result in results]

    assert result_ids == sorted(result_ids)


@pytest.mark.asyncio
async def test_recall_respects_limit() -> None:
    memory = Memory()
    await memory.observe(_obs("alpha one", record_id="1"))
    await memory.observe(_obs("alpha two", record_id="2"))
    await memory.observe(_obs("alpha three", record_id="3"))

    results = await memory.recall("alpha", tenant_id="local", limit=2)

    assert len(results) == 2


@pytest.mark.parametrize("query", ["", "   "])
@pytest.mark.asyncio
async def test_recall_rejects_invalid_query(query: str) -> None:
    memory = Memory()

    with pytest.raises(ValidationError, match="Query must not be empty"):
        await memory.recall(query, tenant_id="local")


@pytest.mark.parametrize("limit", [0, -1])
@pytest.mark.asyncio
async def test_recall_rejects_invalid_limit(limit: int) -> None:
    memory = Memory()
    await memory.observe(_obs("alpha one", record_id="1"))

    with pytest.raises(ValidationError, match="Limit must be greater than zero"):
        await memory.recall("alpha", tenant_id="local", limit=limit)


@pytest.mark.asyncio
async def test_recall_requires_tenant() -> None:
    memory = Memory()
    with pytest.raises(ValidationError, match="tenant_id"):
        await memory.recall("alpha", tenant_id="  ")


@pytest.mark.asyncio
async def test_sleep_is_safe_to_call() -> None:
    memory = Memory()
    await memory.observe(_obs("alpha one", record_id="1"))

    memory.sleep()

    assert len(await memory.recall("alpha", tenant_id="local")) == 1


@pytest.mark.asyncio
async def test_clear_removes_observations() -> None:
    memory = Memory()
    await memory.observe(_obs("alpha one", record_id="1"))
    await memory.observe(_obs("alpha two", record_id="2"))

    await memory.clear(tenant_id="local")

    assert await memory.recall("alpha", tenant_id="local") == []


@pytest.mark.asyncio
async def test_tenant_isolation_in_recall() -> None:
    memory = Memory()
    await memory.observe(_obs("shared topic alpha", record_id="1", tenant_id="tenant_a"))
    await memory.observe(_obs("shared topic beta", record_id="1", tenant_id="tenant_b"))

    results = await memory.recall("shared topic", tenant_id="tenant_a")

    assert len(results) == 1
    assert results[0].observation.tenant_id == "tenant_a"
