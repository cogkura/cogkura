from datetime import UTC, datetime

import pytest

from cognema import Memory, ObservationInput
from cognema.exceptions import ValidationError
from cognema.models import ActivationConfig, MemoryKind
from cognema.observations.models import IngestStatus


def _obs(
    content: str,
    *,
    record_id: str,
    tenant_id: str = "local",
    conversation_id: str = "conv-1",
) -> ObservationInput:
    return ObservationInput(
        tenant_id=tenant_id,
        source_namespace="direct",
        source_record_id=record_id,
        content=content,
        observed_at=datetime.now(UTC),
        metadata={"conversation_id": conversation_id},
    )


async def _encode(
    memory: Memory,
    content: str,
    *,
    record_id: str,
    tenant_id: str = "local",
    conversation_id: str | None = None,
) -> None:
    conv = conversation_id or f"conv-{record_id}"
    await memory.observe(
        _obs(content, record_id=record_id, tenant_id=tenant_id, conversation_id=conv)
    )
    await memory.encode_episodes(tenant_id=tenant_id)


@pytest.mark.asyncio
async def test_observe_stores_observation() -> None:
    memory = Memory()

    status = await memory.observe(_obs("Cognema explores cognitive recall.", record_id="1"))

    assert status is IngestStatus.CREATED
    await _encode(memory, "Cognema explores cognitive recall.", record_id="1")
    results = await memory.recall(
        "cognitive",
        tenant_id="local",
        limit=5,
    )
    assert len(results) == 1
    assert results[0].memory_kind is MemoryKind.EPISODE


@pytest.mark.asyncio
async def test_recall_matching_episodes() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    await _encode(memory, "George discussed cognitive memory algorithms.", record_id="1")
    await _encode(
        memory,
        "A weather report from London.",
        record_id="2",
        tenant_id="local",
    )

    results = await memory.recall("cognitive memory", tenant_id="local")

    assert len(results) >= 1
    assert "cognitive memory" in results[0].memory.statement.lower()
    assert results[0].score > 0.0
    if len(results) > 1:
        assert results[0].activation > results[1].activation


@pytest.mark.asyncio
async def test_recall_is_deterministic_for_ties() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    await _encode(memory, "alpha beta", record_id="a", tenant_id="local")
    await _encode(
        memory,
        "alpha beta",
        record_id="b",
        tenant_id="local",
    )

    results = await memory.recall("alpha beta", tenant_id="local", limit=2)
    keys = [_result_memory_key(result) for result in results]

    assert keys == sorted(keys)


@pytest.mark.asyncio
async def test_recall_respects_limit() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    await _encode(memory, "alpha one", record_id="1")
    await _encode(memory, "alpha two", record_id="2")
    await _encode(memory, "alpha three", record_id="3")

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
    await _encode(memory, "alpha one", record_id="1")

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
    await _encode(memory, "alpha one", record_id="1")

    memory.sleep()

    assert len(await memory.recall("alpha", tenant_id="local", limit=5)) >= 1


@pytest.mark.asyncio
async def test_clear_removes_memories() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    await _encode(memory, "alpha one", record_id="1")
    await _encode(memory, "alpha two", record_id="2")

    await memory.clear(tenant_id="local")

    assert await memory.recall("alpha", tenant_id="local") == []


@pytest.mark.asyncio
async def test_tenant_isolation_in_recall() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    await _encode(
        memory,
        "shared topic alpha",
        record_id="1",
        tenant_id="tenant_a",
    )
    await _encode(
        memory,
        "shared topic beta",
        record_id="1",
        tenant_id="tenant_b",
    )

    results = await memory.recall("shared topic", tenant_id="tenant_a")

    assert len(results) == 1
    assert results[0].memory.tenant_id == "tenant_a"


@pytest.mark.asyncio
async def test_record_access_increases_subsequent_activation() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10.0))
    await _encode(memory, "payment incident resolved", record_id="1")
    first = await memory.recall("payment incident", tenant_id="local", limit=1)
    await memory.record_access(first, tenant_id="local", request_id="run-1")
    second = await memory.recall("payment incident", tenant_id="local", limit=1)
    assert second[0].activation > first[0].activation


def _result_memory_key(result: object) -> str:
    from cognema.models import RecallResult

    assert isinstance(result, RecallResult)
    return result.memory.memory_key
