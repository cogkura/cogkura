import pytest

from cognema import Memory
from cognema.exceptions import ValidationError
from cognema.storage import InMemoryStorage


def test_observe_stores_event() -> None:
    memory = Memory()

    event = memory.observe("Cognema explores cognitive recall.")

    assert event.content == "Cognema explores cognitive recall."
    assert len(memory.recall("cognitive")) == 1


def test_recall_matching_events() -> None:
    memory = Memory()
    first = memory.observe("George discussed cognitive memory algorithms.")
    second = memory.observe("A weather report from London.")

    results = memory.recall("cognitive memory")

    assert len(results) == 1
    assert results[0].event == first
    assert results[0].score > 0.0
    assert second not in [result.event for result in results]


def test_recall_is_deterministic_for_ties() -> None:
    memory = Memory()
    first = memory.observe("alpha beta")
    second = memory.observe("alpha beta")

    results = memory.recall("alpha beta", limit=2)
    result_ids = [result.event.id for result in results]

    assert result_ids == sorted([first.id, second.id])


def test_recall_respects_limit() -> None:
    memory = Memory()
    memory.observe("alpha one")
    memory.observe("alpha two")
    memory.observe("alpha three")

    results = memory.recall("alpha", limit=2)

    assert len(results) == 2


@pytest.mark.parametrize("query", ["", "   "])
def test_recall_rejects_invalid_query(query: str) -> None:
    memory = Memory()

    with pytest.raises(ValidationError, match="Query must not be empty"):
        memory.recall(query)


@pytest.mark.parametrize("limit", [0, -1])
def test_recall_rejects_invalid_limit(limit: int) -> None:
    memory = Memory()
    memory.observe("alpha one")

    with pytest.raises(ValidationError, match="Limit must be greater than zero"):
        memory.recall("alpha", limit=limit)


def test_sleep_is_safe_to_call() -> None:
    memory = Memory()
    memory.observe("alpha one")

    memory.sleep()

    assert len(memory.recall("alpha")) == 1


def test_clear_removes_events() -> None:
    memory = Memory()
    memory.observe("alpha one")
    memory.observe("alpha two")

    memory.clear()

    assert memory.recall("alpha") == []


def test_storage_injection() -> None:
    storage = InMemoryStorage()
    memory = Memory(storage=storage)
    event = memory.observe("injected storage")

    assert storage.get(event.id) == event
