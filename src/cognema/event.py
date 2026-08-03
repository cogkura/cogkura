"""Memory event models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from cognema.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    """A single observed memory event."""

    content: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    importance: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        normalized_content = self.content.strip()
        if not normalized_content:
            raise ValidationError("Event content must not be empty.")
        object.__setattr__(self, "content", normalized_content)

        if self.importance is not None and not 0.0 <= self.importance <= 1.0:
            raise ValidationError("Importance must be between 0.0 and 1.0.")

        metadata_dict = dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata_dict))

        normalized_tags = tuple(tag.strip() for tag in self.tags if tag.strip())
        object.__setattr__(self, "tags", normalized_tags)

        if not self.id.strip():
            raise ValidationError("Event id must not be empty.")

        if self.created_at.tzinfo is None:
            raise ValidationError("Event timestamp must be timezone-aware.")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
