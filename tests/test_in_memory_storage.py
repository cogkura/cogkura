from cognema.event import MemoryEvent
from cognema.storage.in_memory import InMemoryStorage


def test_store_and_get_event() -> None:
    storage = InMemoryStorage()
    event = MemoryEvent(content="event content")

    storage.store(event)

    assert storage.get(event.id) == event


def test_list_events() -> None:
    storage = InMemoryStorage()
    first = MemoryEvent(content="first event")
    second = MemoryEvent(content="second event")
    storage.store(first)
    storage.store(second)

    assert storage.list() == [first, second]


def test_clear_events() -> None:
    storage = InMemoryStorage()
    storage.store(MemoryEvent(content="first event"))
    storage.store(MemoryEvent(content="second event"))

    storage.clear()

    assert storage.list() == []
