"""Context reinstatement policy for episodic activation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cogkura.models import ContextMatch, ContextReinstatement, ContextReinstatementReason


def reinstatement_strength_from_match(match: ContextMatch | None) -> float:
    """Compute R = M × V; None score or missing match yields 0."""
    if match is None or match.score is None:
        return 0.0
    return match.score * match.cue_coverage


class ContextReinstatementPolicy(Protocol):
    """Convert context-match evidence into bounded activation contribution."""

    def evaluate(
        self,
        *,
        match: ContextMatch | None,
        weight: float,
        episodic: bool,
        has_retrieval_context: bool,
    ) -> ContextReinstatement:
        """Return reinstatement diagnostics and activation contribution."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicContextReinstatementPolicy:
    """Positive-only reinstatement: R = match_score × cue_coverage, C = weight × R."""

    def evaluate(
        self,
        *,
        match: ContextMatch | None,
        weight: float,
        episodic: bool,
        has_retrieval_context: bool,
    ) -> ContextReinstatement:
        if not episodic:
            return ContextReinstatement(
                match_score=None,
                cue_coverage=0.0,
                strength=0.0,
                weight=weight,
                activation_contribution=0.0,
                applied=False,
                reason=ContextReinstatementReason.NOT_EPISODIC,
            )
        if not has_retrieval_context:
            return ContextReinstatement(
                match_score=None,
                cue_coverage=0.0,
                strength=0.0,
                weight=weight,
                activation_contribution=0.0,
                applied=False,
                reason=ContextReinstatementReason.NO_RETRIEVAL_CONTEXT,
            )

        resolved = match if match is not None else _empty_match()
        match_score = resolved.score
        cue_coverage = resolved.cue_coverage

        if match_score is None:
            return ContextReinstatement(
                match_score=None,
                cue_coverage=cue_coverage,
                strength=0.0,
                weight=weight,
                activation_contribution=0.0,
                applied=False,
                reason=ContextReinstatementReason.NO_COMPARABLE_CONTEXT,
            )

        strength = reinstatement_strength_from_match(resolved)
        if match_score == 0.0:
            return ContextReinstatement(
                match_score=match_score,
                cue_coverage=cue_coverage,
                strength=0.0,
                weight=weight,
                activation_contribution=0.0,
                applied=False,
                reason=ContextReinstatementReason.ZERO_MATCH,
            )

        if weight == 0.0:
            return ContextReinstatement(
                match_score=match_score,
                cue_coverage=cue_coverage,
                strength=strength,
                weight=weight,
                activation_contribution=0.0,
                applied=False,
                reason=ContextReinstatementReason.DISABLED,
            )

        contribution = weight * strength
        return ContextReinstatement(
            match_score=match_score,
            cue_coverage=cue_coverage,
            strength=strength,
            weight=weight,
            activation_contribution=contribution,
            applied=True,
            reason=ContextReinstatementReason.APPLIED,
        )


def _empty_match() -> ContextMatch:
    return ContextMatch(
        dimensions=(),
        attribute_dimensions=(),
        score=None,
        comparable_count=0,
        cue_dimension_count=0,
        cue_coverage=0.0,
    )
