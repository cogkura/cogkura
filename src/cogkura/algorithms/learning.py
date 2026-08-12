"""Deterministic learning and reinforcement planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from cogkura.exceptions import ValidationError
from cogkura.models import (
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    LearningPlan,
    MemoryFeedback,
    MemoryIdentity,
    RetrievalCue,
    StoredMemoryLearningState,
)

_WHITESPACE_PATTERN = re.compile(r"\s+")
_GLOBAL_CONTEXT_KEY = "global"


@dataclass(frozen=True, slots=True)
class LearningCounts:
    """Combined helpful, unhelpful, and incorrect learning counts."""

    helpful: int
    unhelpful: int
    incorrect: int


class LearningProcessor(Protocol):
    """Plans deterministic learning updates from application feedback."""

    def plan(
        self,
        *,
        feedback: LearningFeedback,
        config: LearningConfig,
    ) -> LearningPlan:
        """Produce an immutable learning plan without storage I/O."""


@dataclass(slots=True)
class DeterministicLearningProcessor:
    """Deterministic learning planner for utility, reinforcement, and associations."""

    def plan(
        self,
        *,
        feedback: LearningFeedback,
        config: LearningConfig,
    ) -> LearningPlan:
        if len(feedback.items) > config.max_feedback_items:
            raise ValidationError(
                f"items length {len(feedback.items)} exceeds "
                f"max_feedback_items {config.max_feedback_items}."
            )

        context_key = learning_context_key(feedback.goal)
        fingerprint = feedback_fingerprint(feedback, context_key=context_key)
        helpful_identities = tuple(
            sorted(
                (
                    item.identity
                    for item in feedback.items
                    if item.outcome is LearningOutcome.HELPFUL
                ),
                key=_identity_sort_key,
            )
        )
        association_pairs, association_items_skipped = _association_pairs(
            helpful_identities=helpful_identities,
            max_items=config.max_association_items_per_feedback,
        )

        return LearningPlan(
            feedback_id=feedback.feedback_id,
            feedback_fingerprint=fingerprint,
            tenant_id=feedback.tenant_id,
            subject_id=feedback.subject_id,
            context_key=context_key,
            occurred_at=feedback.occurred_at,
            items=feedback.items,
            association_pairs=association_pairs,
            association_items_skipped=association_items_skipped,
            metadata=feedback.metadata,
        )


def learning_context_key(goal: RetrievalCue | None) -> str:
    """Return a stable context key for contextual utility learning."""
    if goal is None:
        return _GLOBAL_CONTEXT_KEY
    canonical = _canonical_goal(goal)
    if not canonical:
        return _GLOBAL_CONTEXT_KEY
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest


def feedback_fingerprint(
    feedback: LearningFeedback,
    *,
    context_key: str,
) -> str:
    """Compute a deterministic fingerprint for idempotent feedback application."""
    payload = {
        "tenant_id": feedback.tenant_id,
        "feedback_id": feedback.feedback_id,
        "subject_id": feedback.subject_id,
        "context_key": context_key,
        "occurred_at": feedback.occurred_at.isoformat(),
        "items": [
            {
                "memory_kind": item.identity.memory_kind.value,
                "memory_key": item.identity.memory_key,
                "outcome": item.outcome.value,
                "revision_key": item.revision_key,
            }
            for item in sorted(feedback.items, key=_feedback_item_sort_key)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calculate_utility(
    *,
    helpful: int,
    unhelpful: int,
    incorrect: int,
    config: LearningConfig,
) -> float:
    """Calculate Beta-style smoothed utility from persisted counts."""
    positive_mass = helpful
    negative_mass = unhelpful + config.incorrect_utility_weight * incorrect
    numerator = config.utility_prior_positive + positive_mass
    denominator = (
        config.utility_prior_positive
        + config.utility_prior_negative
        + positive_mass
        + negative_mass
    )
    return numerator / denominator


def calculate_association_strength(
    coactivation_count: int,
    *,
    config: LearningConfig,
) -> float:
    """Calculate saturated association strength from coactivation count."""
    if coactivation_count < config.minimum_association_coactivations:
        return 0.0
    return 1.0 - math.exp(-coactivation_count / config.association_tau)


def canonical_association_pair(
    left: MemoryIdentity,
    right: MemoryIdentity,
) -> tuple[MemoryIdentity, MemoryIdentity]:
    """Return identities in deterministic lexicographic order."""
    if _identity_sort_key(left) <= _identity_sort_key(right):
        return left, right
    return right, left


def combine_learning_counts(
    *,
    global_state: StoredMemoryLearningState | None,
    context_state: StoredMemoryLearningState | None,
    context_key: str,
) -> LearningCounts:
    """Combine global and contextual counts without double-counting."""
    helpful = 0
    unhelpful = 0
    incorrect = 0
    if global_state is not None:
        helpful += global_state.helpful_count
        unhelpful += global_state.unhelpful_count
        incorrect += global_state.incorrect_count
    if context_state is not None and context_key != _GLOBAL_CONTEXT_KEY:
        helpful += context_state.helpful_count
        unhelpful += context_state.unhelpful_count
        incorrect += context_state.incorrect_count
    return LearningCounts(helpful=helpful, unhelpful=unhelpful, incorrect=incorrect)


def learning_counts_by_identity(
    *,
    identities: Sequence[MemoryIdentity],
    states: Sequence[StoredMemoryLearningState],
    context_key: str,
) -> Mapping[MemoryIdentity, LearningCounts]:
    """Combine global and contextual counts for each identity."""
    by_identity_context: dict[tuple[str, str, str], StoredMemoryLearningState] = {}
    for state in states:
        by_identity_context[
            (
                state.memory_kind.value,
                state.memory_key,
                state.context_key,
            )
        ] = state

    counts: dict[MemoryIdentity, LearningCounts] = {}
    for identity in identities:
        global_state = by_identity_context.get(
            (identity.memory_kind.value, identity.memory_key, _GLOBAL_CONTEXT_KEY)
        )
        context_state = by_identity_context.get(
            (identity.memory_kind.value, identity.memory_key, context_key)
        )
        counts[identity] = combine_learning_counts(
            global_state=global_state,
            context_state=context_state,
            context_key=context_key,
        )
    return counts


def build_learning_utilities(
    *,
    identities: Sequence[MemoryIdentity],
    states: Sequence[StoredMemoryLearningState],
    context_key: str,
    config: LearningConfig,
) -> dict[MemoryIdentity, float]:
    """Build combined contextual utilities for working-memory selection."""
    counts_by_identity = learning_counts_by_identity(
        identities=identities,
        states=states,
        context_key=context_key,
    )

    utilities: dict[MemoryIdentity, float] = {}
    for identity in identities:
        counts = counts_by_identity[identity]
        utilities[identity] = calculate_utility(
            helpful=counts.helpful,
            unhelpful=counts.unhelpful,
            incorrect=counts.incorrect,
            config=config,
        )
    return utilities


def _association_pairs(
    *,
    helpful_identities: Sequence[MemoryIdentity],
    max_items: int,
) -> tuple[tuple[tuple[MemoryIdentity, MemoryIdentity], ...], int]:
    selected = helpful_identities[:max_items]
    skipped = max(0, len(helpful_identities) - len(selected))
    pairs: list[tuple[MemoryIdentity, MemoryIdentity]] = []
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            pairs.append(canonical_association_pair(left, right))
    return tuple(pairs), skipped


def _canonical_goal(goal: RetrievalCue) -> str:
    parts: list[str] = []
    if goal.text and goal.text.strip():
        parts.append(f"text:{_normalise_text(goal.text)}")
    if goal.subject_id and goal.subject_id.strip():
        parts.append(f"subject_id:{goal.subject_id.strip()}")
    if goal.entity_ids:
        parts.append(f"entity_ids:{','.join(sorted(set(goal.entity_ids)))}")
    if goal.predicate and goal.predicate.strip():
        parts.append(f"predicate:{_normalise_text(goal.predicate)}")
    if goal.object_value and goal.object_value.strip():
        parts.append(f"object_value:{_normalise_text(goal.object_value)}")
    if goal.qualifiers:
        qualifier_pairs = sorted(
            (_normalise_text(str(key)), _normalise_text(str(value)))
            for key, value in goal.qualifiers.items()
        )
        parts.append("qualifiers:" + ";".join(f"{key}={value}" for key, value in qualifier_pairs))
    return "|".join(parts)


def _normalise_text(value: str) -> str:
    normalised = unicodedata.normalize("NFKC", value)
    normalised = _WHITESPACE_PATTERN.sub(" ", normalised).strip()
    return normalised.casefold()


def _identity_sort_key(identity: MemoryIdentity) -> tuple[str, str]:
    return identity.memory_kind.value, identity.memory_key


def _feedback_item_sort_key(item: MemoryFeedback) -> tuple[str, str, str, str | None]:
    return (
        item.identity.memory_kind.value,
        item.identity.memory_key,
        item.outcome.value,
        item.revision_key,
    )
