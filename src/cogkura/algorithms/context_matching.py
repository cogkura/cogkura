"""Deterministic encoding-context matching for retrieval diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cogkura.models import (
    ContextDimensionMatch,
    ContextMatch,
    ContextMatchState,
    MemoryContextSignature,
)
from cogkura.observations.encoding_context import RetrievalContext

_PRIMARY_DIMENSIONS: tuple[tuple[str, str, str], ...] = (
    ("conversation", "conversation_id", "conversation_ids"),
    ("thread", "thread_id", "thread_ids"),
    ("session", "session_id", "session_ids"),
    ("goal", "goal", "goals"),
    ("activity", "activity", "activities"),
    ("domain", "domain", "domains"),
    ("location", "location", "locations"),
    ("temporal_context", "temporal_context", "temporal_contexts"),
)


class ContextMatcher(Protocol):
    """Compare retrieval context against encoded episodic context."""

    def match(
        self,
        retrieval_context: RetrievalContext | None,
        encoding_context: MemoryContextSignature,
    ) -> ContextMatch:
        """Return deterministic context-match diagnostics."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicContextMatcher:
    """Exact normalised comparison without fuzzy or semantic matching."""

    def match(
        self,
        retrieval_context: RetrievalContext | None,
        encoding_context: MemoryContextSignature,
    ) -> ContextMatch:
        if retrieval_context is None or retrieval_context.is_empty():
            return ContextMatch(
                dimensions=(),
                attribute_dimensions=(),
                score=None,
                comparable_count=0,
                cue_dimension_count=0,
                cue_coverage=0.0,
            )

        dimensions: list[ContextDimensionMatch] = []
        comparable_scores: list[float] = []
        cue_dimension_count = 0

        for dimension_name, cue_field, memory_field in _PRIMARY_DIMENSIONS:
            cue_values = _cue_values(retrieval_context, cue_field)
            if not cue_values:
                continue
            cue_dimension_count += 1
            memory_values = getattr(encoding_context, memory_field)
            if not memory_values:
                dimensions.append(
                    ContextDimensionMatch(
                        dimension=dimension_name,
                        state=ContextMatchState.UNAVAILABLE,
                        score=None,
                        cue_values=cue_values,
                        memory_values=(),
                    )
                )
                continue

            dimension_score = _dimension_score(cue_values=cue_values, memory_values=memory_values)
            comparable_scores.append(dimension_score)
            dimensions.append(
                ContextDimensionMatch(
                    dimension=dimension_name,
                    state=_state_from_score(dimension_score, multi_value=len(cue_values) > 1),
                    score=dimension_score,
                    cue_values=cue_values,
                    memory_values=memory_values,
                )
            )

        attribute_dimensions: list[ContextDimensionMatch] = []
        for attribute_key in sorted(retrieval_context.attributes):
            cue_value = retrieval_context.attributes[attribute_key]
            memory_values = encoding_context.attributes.get(attribute_key, ())
            if not memory_values:
                attribute_dimensions.append(
                    ContextDimensionMatch(
                        dimension=attribute_key,
                        state=ContextMatchState.UNAVAILABLE,
                        score=None,
                        cue_values=(cue_value,),
                        memory_values=(),
                    )
                )
                continue
            attribute_score = 1.0 if cue_value in memory_values else 0.0
            attribute_dimensions.append(
                ContextDimensionMatch(
                    dimension=attribute_key,
                    state=_state_from_score(attribute_score, multi_value=False),
                    score=attribute_score,
                    cue_values=(cue_value,),
                    memory_values=memory_values,
                )
            )

        comparable_count = len(comparable_scores)
        score = sum(comparable_scores) / comparable_count if comparable_count > 0 else None
        cue_coverage = comparable_count / cue_dimension_count if cue_dimension_count > 0 else 0.0
        return ContextMatch(
            dimensions=tuple(dimensions),
            attribute_dimensions=tuple(attribute_dimensions),
            score=score,
            comparable_count=comparable_count,
            cue_dimension_count=cue_dimension_count,
            cue_coverage=cue_coverage,
        )


def _cue_values(retrieval_context: RetrievalContext, cue_field: str) -> tuple[str, ...]:
    if cue_field == "temporal_context":
        return retrieval_context.temporal_context
    value = getattr(retrieval_context, cue_field)
    if value is None:
        return ()
    return (value,)


def _dimension_score(
    *,
    cue_values: tuple[str, ...],
    memory_values: tuple[str, ...],
) -> float:
    if len(cue_values) == 1:
        return 1.0 if cue_values[0] in memory_values else 0.0
    intersection = set(cue_values).intersection(memory_values)
    return len(intersection) / len(cue_values)


def _state_from_score(score: float, *, multi_value: bool) -> ContextMatchState:
    if score >= 1.0:
        return ContextMatchState.MATCH
    if score <= 0.0:
        return ContextMatchState.MISMATCH
    if multi_value:
        return ContextMatchState.PARTIAL_MATCH
    return ContextMatchState.MISMATCH
