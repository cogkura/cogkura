"""Semantic support-context propagation for declarative activation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from cogkura.algorithms.context_matching import ContextMatcher
from cogkura.models import (
    ContextMatch,
    SemanticDerivationRelation,
    SemanticSupportContextEvidence,
    SemanticSupportContextItem,
    SemanticSupportContextReason,
    StoredEpisode,
    StoredSemanticMemory,
)
from cogkura.observations.encoding_context import RetrievalContext


def reinstatement_strength_from_match(match: ContextMatch | None) -> float:
    """Compute R = M × V; None score or missing match yields 0."""
    if match is None or match.score is None:
        return 0.0
    return match.score * match.cue_coverage


def unique_support_episode_ids(memory: StoredSemanticMemory) -> tuple[str, ...]:
    """Return deduplicated SUPPORT episode ids in deterministic order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for derivation in memory.derivations:
        if derivation.relation is not SemanticDerivationRelation.SUPPORTS:
            continue
        if derivation.episode_id in seen:
            continue
        seen.add(derivation.episode_id)
        ordered.append(derivation.episode_id)
    return tuple(sorted(ordered))


@dataclass(slots=True)
class SupportContextMatchCache:
    """Request-scoped memoization of support episode context matches."""

    context_matcher: ContextMatcher
    retrieval_context: RetrievalContext
    episode_by_id: Mapping[str, StoredEpisode]
    _cache: dict[str, tuple[ContextMatch | None, float]] | None = None

    def match_strength(self, episode_id: str) -> tuple[ContextMatch | None, float, bool]:
        """Return match, reinstatement strength, and whether the support is unavailable."""
        episode = self.episode_by_id.get(episode_id)
        if episode is None:
            return None, 0.0, True
        if self._cache is None:
            self._cache = {}
        cached = self._cache.get(episode_id)
        if cached is not None:
            match, strength = cached
            return match, strength, False
        match = self.context_matcher.match(self.retrieval_context, episode.encoding_context)
        strength = reinstatement_strength_from_match(match)
        self._cache[episode_id] = (match, strength)
        return match, strength, False


class SemanticSupportContextPolicy(Protocol):
    """Aggregate support-level context evidence into semantic reinstatement."""

    def evaluate(
        self,
        *,
        memory: StoredSemanticMemory,
        retrieval_context: RetrievalContext | None,
        weight: float,
        match_cache: SupportContextMatchCache | None,
    ) -> SemanticSupportContextEvidence:
        """Return support-context diagnostics and activation contribution."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicSemanticSupportContextPolicy:
    """Positive-only semantic propagation: Rs = sum(Rj) / K, Cs = weight × Rs."""

    def evaluate(
        self,
        *,
        memory: StoredSemanticMemory,
        retrieval_context: RetrievalContext | None,
        weight: float,
        match_cache: SupportContextMatchCache | None,
    ) -> SemanticSupportContextEvidence:
        support_ids = unique_support_episode_ids(memory)
        if retrieval_context is None or retrieval_context.is_empty():
            return _empty_evidence(
                support_ids=support_ids,
                weight=weight,
                reason=SemanticSupportContextReason.NO_RETRIEVAL_CONTEXT,
            )
        if not support_ids:
            return _empty_evidence(
                support_ids=support_ids,
                weight=weight,
                reason=SemanticSupportContextReason.NO_SUPPORTS,
            )
        if match_cache is None:
            return _empty_evidence(
                support_ids=support_ids,
                weight=weight,
                reason=SemanticSupportContextReason.NOT_EVALUATED,
            )

        items: list[SemanticSupportContextItem] = []
        strengths: list[float] = []
        comparable_count = 0
        unavailable_count = 0
        matching_count = 0
        conflicting_count = 0

        for episode_id in support_ids:
            match, strength, unavailable = match_cache.match_strength(episode_id)
            strengths.append(strength)
            if unavailable:
                unavailable_count += 1
                items.append(
                    SemanticSupportContextItem(
                        episode_id=episode_id,
                        context_match=None,
                        reinstatement_strength=0.0,
                        unavailable=True,
                    )
                )
                continue
            if match is not None and match.score is not None:
                comparable_count += 1
                if strength > 0.0:
                    matching_count += 1
                if match.score == 0.0:
                    conflicting_count += 1
            else:
                unavailable_count += 1
            items.append(
                SemanticSupportContextItem(
                    episode_id=episode_id,
                    context_match=match,
                    reinstatement_strength=strength,
                    unavailable=False,
                )
            )

        support_count = len(support_ids)
        support_coverage = comparable_count / support_count if support_count else 0.0
        comparable_strengths = [
            item.reinstatement_strength
            for item in items
            if item.context_match is not None and item.context_match.score is not None
        ]
        mean_strength = (
            sum(comparable_strengths) / len(comparable_strengths) if comparable_strengths else 0.0
        )
        semantic_strength = sum(strengths) / support_count if support_count else 0.0

        if comparable_count == 0:
            return SemanticSupportContextEvidence(
                support_count=support_count,
                comparable_support_count=0,
                unavailable_support_count=unavailable_count,
                matching_support_count=0,
                conflicting_support_count=conflicting_count,
                support_coverage=support_coverage,
                mean_reinstatement_strength=mean_strength,
                strength=semantic_strength,
                weight=weight,
                activation_contribution=0.0,
                applied=False,
                reason=SemanticSupportContextReason.NO_COMPARABLE_CONTEXT,
                supports=tuple(items),
            )

        if weight == 0.0:
            return SemanticSupportContextEvidence(
                support_count=support_count,
                comparable_support_count=comparable_count,
                unavailable_support_count=unavailable_count,
                matching_support_count=matching_count,
                conflicting_support_count=conflicting_count,
                support_coverage=support_coverage,
                mean_reinstatement_strength=mean_strength,
                strength=semantic_strength,
                weight=weight,
                activation_contribution=0.0,
                applied=False,
                reason=SemanticSupportContextReason.DISABLED,
                supports=tuple(items),
            )

        contribution = weight * semantic_strength
        applied = contribution > 0.0
        return SemanticSupportContextEvidence(
            support_count=support_count,
            comparable_support_count=comparable_count,
            unavailable_support_count=unavailable_count,
            matching_support_count=matching_count,
            conflicting_support_count=conflicting_count,
            support_coverage=support_coverage,
            mean_reinstatement_strength=mean_strength,
            strength=semantic_strength,
            weight=weight,
            activation_contribution=contribution,
            applied=applied,
            reason=SemanticSupportContextReason.APPLIED
            if applied
            else SemanticSupportContextReason.DISABLED,
            supports=tuple(items),
        )


def _unavailable_support_count_for_reason(
    *,
    support_ids: tuple[str, ...],
    reason: SemanticSupportContextReason,
) -> int:
    if reason is SemanticSupportContextReason.NO_COMPARABLE_CONTEXT:
        return len(support_ids)
    return 0


def _empty_evidence(
    *,
    support_ids: tuple[str, ...],
    weight: float,
    reason: SemanticSupportContextReason,
) -> SemanticSupportContextEvidence:
    return SemanticSupportContextEvidence(
        support_count=len(support_ids),
        comparable_support_count=0,
        unavailable_support_count=_unavailable_support_count_for_reason(
            support_ids=support_ids,
            reason=reason,
        ),
        matching_support_count=0,
        conflicting_support_count=0,
        support_coverage=0.0,
        mean_reinstatement_strength=0.0,
        strength=0.0,
        weight=weight,
        activation_contribution=0.0,
        applied=False,
        reason=reason,
        supports=(),
    )
