from datetime import UTC

import pytest

from cognema.event import MemoryEvent
from cognema.exceptions import ValidationError


def test_create_valid_event() -> None:
    event = MemoryEvent(
        content="George discussed cognitive memory algorithms",
        metadata={"source": "conversation"},
        importance=0.75,
        tags=("memory", "research"),
    )

    assert event.id
    assert event.content == "George discussed cognitive memory algorithms"
    assert event.metadata["source"] == "conversation"
    assert event.importance == 0.75
    assert event.tags == ("memory", "research")
    assert event.created_at.tzinfo is UTC


def test_event_rejects_empty_content() -> None:
    with pytest.raises(ValidationError, match="content must not be empty"):
        MemoryEvent(content="   ")


@pytest.mark.parametrize("importance", [-0.1, 1.1])
def test_event_validates_importance_range(importance: float) -> None:
    with pytest.raises(ValidationError, match="Importance must be between 0.0 and 1.0"):
        MemoryEvent(content="valid", importance=importance)


def test_event_ids_are_unique() -> None:
    first = MemoryEvent(content="event one")
    second = MemoryEvent(content="event two")

    assert first.id != second.id
