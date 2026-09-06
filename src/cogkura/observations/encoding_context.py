"""Encoding-context models and deterministic normalisation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from cogkura.exceptions import ValidationError


def _normalise_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _normalise_text_tuple(values: Sequence[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalised: list[str] = []
    for value in values:
        if value is None:
            continue
        trimmed = value.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        normalised.append(trimmed)
    return tuple(sorted(normalised))


def _normalise_string_attributes(
    attributes: Mapping[str, Any],
) -> MappingProxyType[str, str]:
    normalised: dict[str, str] = {}
    for key, value in attributes.items():
        if not isinstance(key, str):
            raise ValidationError("context attribute keys must be strings.")
        trimmed_key = key.strip()
        if not trimmed_key:
            raise ValidationError("context attribute keys must not be empty.")
        if not isinstance(value, str):
            raise ValidationError("context attribute values must be strings.")
        trimmed_value = value.strip()
        if not trimmed_value:
            continue
        normalised[trimmed_key] = trimmed_value
    return MappingProxyType(dict(sorted(normalised.items())))


def _normalise_attribute_map(
    attributes: Mapping[str, Sequence[str | None]],
) -> MappingProxyType[str, tuple[str, ...]]:
    normalised: dict[str, tuple[str, ...]] = {}
    for key in sorted(attributes):
        value = attributes[key]
        if not isinstance(key, str):
            raise ValidationError("context attribute keys must be strings.")
        trimmed_key = key.strip()
        if not trimmed_key:
            raise ValidationError("context attribute keys must not be empty.")
        values = _normalise_text_tuple(value)
        if values:
            normalised[trimmed_key] = values
    return MappingProxyType(normalised)


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """Structured context supplied with an observation before episodic encoding."""

    conversation_id: str | None = None
    thread_id: str | None = None
    session_id: str | None = None
    goal: str | None = None
    activity: str | None = None
    domain: str | None = None
    location: str | None = None
    temporal_context: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "conversation_id", _normalise_optional_text(self.conversation_id))
        object.__setattr__(self, "thread_id", _normalise_optional_text(self.thread_id))
        object.__setattr__(self, "session_id", _normalise_optional_text(self.session_id))
        object.__setattr__(self, "goal", _normalise_optional_text(self.goal))
        object.__setattr__(self, "activity", _normalise_optional_text(self.activity))
        object.__setattr__(self, "domain", _normalise_optional_text(self.domain))
        object.__setattr__(self, "location", _normalise_optional_text(self.location))
        object.__setattr__(
            self,
            "temporal_context",
            _normalise_text_tuple(self.temporal_context),
        )
        object.__setattr__(
            self,
            "attributes",
            _normalise_string_attributes(self.attributes),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serialisable representation."""
        return {
            "conversation_id": self.conversation_id,
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "goal": self.goal,
            "activity": self.activity,
            "domain": self.domain,
            "location": self.location,
            "temporal_context": list(self.temporal_context),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_canonical_dict(cls, payload: Mapping[str, Any] | None) -> ObservationContext:
        """Hydrate from persisted JSON; missing payload yields empty context."""
        if not payload:
            return cls()
        raw_attributes = payload.get("attributes", {})
        attributes: dict[str, str] = {}
        if isinstance(raw_attributes, Mapping):
            for key, value in raw_attributes.items():
                if isinstance(key, str) and isinstance(value, str):
                    attributes[key] = value
        raw_temporal = payload.get("temporal_context", ())
        temporal: tuple[str, ...] = ()
        if isinstance(raw_temporal, Sequence) and not isinstance(raw_temporal, (str, bytes)):
            temporal = tuple(str(item) for item in raw_temporal)
        return cls(
            conversation_id=_optional_text(payload.get("conversation_id")),
            thread_id=_optional_text(payload.get("thread_id")),
            session_id=_optional_text(payload.get("session_id")),
            goal=_optional_text(payload.get("goal")),
            activity=_optional_text(payload.get("activity")),
            domain=_optional_text(payload.get("domain")),
            location=_optional_text(payload.get("location")),
            temporal_context=temporal,
            attributes=attributes,
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def observation_contexts_equal(
    left: ObservationContext | None,
    right: ObservationContext | None,
) -> bool:
    left_context = left if left is not None else ObservationContext()
    right_context = right if right is not None else ObservationContext()
    return left_context.to_canonical_dict() == right_context.to_canonical_dict()
