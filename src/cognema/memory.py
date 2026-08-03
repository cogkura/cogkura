"""Public memory API."""

from __future__ import annotations

import re

from cognema.event import MemoryEvent
from cognema.exceptions import ValidationError
from cognema.models import RecallResult
from cognema.storage import InMemoryStorage, MemoryStorage

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class Memory:
    """A minimal cognitive memory facade for observing and recalling events."""

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        self._storage = storage if storage is not None else InMemoryStorage()

    def observe(
        self,
        content: str,
        *,
        metadata: dict[str, object] | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEvent:
        """Create and store a memory event."""
        event = MemoryEvent(
            content=content,
            metadata={} if metadata is None else metadata,
            importance=importance,
            tags=tuple(tags or ()),
        )
        self._storage.store(event)
        return event

    def recall(self, query: str, *, limit: int = 5) -> list[RecallResult]:
        """Recall events by deterministic token-overlap scoring.

        This `0.0.1` retrieval method is a placeholder for future cognitive
        retrieval algorithms such as spreading activation and goal filtering.
        """
        normalized_query = query.strip()
        if not normalized_query:
            raise ValidationError("Query must not be empty.")
        if limit <= 0:
            raise ValidationError("Limit must be greater than zero.")

        query_tokens = _tokenize(normalized_query)
        if not query_tokens:
            raise ValidationError("Query must contain at least one alphanumeric token.")

        results: list[RecallResult] = []
        for event in self._storage.list():
            event_tokens = _tokenize(event.content)
            score, matched_tokens = _score_overlap(query_tokens, event_tokens)
            if score <= 0.0:
                continue
            reason = f"Matched tokens: {', '.join(matched_tokens)}" if matched_tokens else None
            results.append(RecallResult(event=event, score=score, reason=reason))

        results.sort(key=lambda result: (-result.score, result.event.id))
        return results[:limit]

    def sleep(self) -> None:
        """Run deferred memory maintenance.

        For `0.0.1` this is a safe no-op. Future versions will use this hook for
        consolidation, association strengthening, decay, summarization, and
        duplicate merging.
        """

    def clear(self) -> None:
        """Clear all events from the configured storage backend."""
        self._storage.clear()


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _score_overlap(query_tokens: set[str], event_tokens: set[str]) -> tuple[float, tuple[str, ...]]:
    matched_tokens = tuple(sorted(query_tokens.intersection(event_tokens)))
    if not matched_tokens:
        return 0.0, matched_tokens
    score = len(matched_tokens) / len(query_tokens)
    return score, matched_tokens
