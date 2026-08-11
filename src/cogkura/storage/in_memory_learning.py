"""In-memory learning store for tests."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from cogkura.exceptions import StorageError
from cogkura.models import (
    ActivationReferenceTrace,
    LearningOutcome,
    LearningPlan,
    LearningWriteResult,
    MemoryIdentity,
    StoredMemoryAssociation,
    StoredMemoryLearningState,
)
from cogkura.storage.base import LearningStore


@dataclass(slots=True)
class _LearningEvent:
    tenant_id: str
    feedback_id: str
    feedback_fingerprint: str
    subject_id: str | None
    context_key: str
    occurred_at: datetime
    metadata: Mapping[str, object]


class InMemoryLearningStore(LearningStore):
    """In-memory store for learning events, state, and associations."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], _LearningEvent] = {}
        self._feedback_items: dict[
            tuple[str, str], tuple[tuple[MemoryIdentity, LearningOutcome], ...]
        ] = {}
        self._states: dict[tuple[str, str, str, str], StoredMemoryLearningState] = {}
        self._associations: dict[
            tuple[str, str, str, str, str],
            StoredMemoryAssociation,
        ] = {}

    async def apply(self, plan: LearningPlan) -> LearningWriteResult:
        event_key = (plan.tenant_id, plan.feedback_id)
        existing = self._events.get(event_key)
        if existing is not None:
            if existing.feedback_fingerprint == plan.feedback_fingerprint:
                return LearningWriteResult(
                    created=False,
                    unchanged=True,
                    helpful=0,
                    unhelpful=0,
                    incorrect=0,
                    associations_reinforced=0,
                )
            raise StorageError(
                f"Conflicting feedback fingerprint for feedback_id {plan.feedback_id!r}."
            )

        events = {
            key: _LearningEvent(
                tenant_id=event.tenant_id,
                feedback_id=event.feedback_id,
                feedback_fingerprint=event.feedback_fingerprint,
                subject_id=event.subject_id,
                context_key=event.context_key,
                occurred_at=event.occurred_at,
                metadata=dict(event.metadata),
            )
            for key, event in self._events.items()
        }
        feedback_items = copy.deepcopy(self._feedback_items)
        states = copy.deepcopy(self._states)
        associations: dict[tuple[str, str, str, str, str], StoredMemoryAssociation] = copy.deepcopy(
            self._associations
        )

        helpful = unhelpful = incorrect = 0
        for item in plan.items:
            if item.outcome is LearningOutcome.HELPFUL:
                helpful += 1
            elif item.outcome is LearningOutcome.UNHELPFUL:
                unhelpful += 1
            else:
                incorrect += 1
            _increment_state(
                states=states,
                tenant_id=plan.tenant_id,
                context_key=plan.context_key,
                identity=item.identity,
                outcome=item.outcome,
                at=plan.occurred_at,
            )

        associations_reinforced = 0
        for left, right in plan.association_pairs:
            _increment_association(
                associations=associations,
                tenant_id=plan.tenant_id,
                left=left,
                right=right,
                at=plan.occurred_at,
            )
            associations_reinforced += 1

        events[event_key] = _LearningEvent(
            tenant_id=plan.tenant_id,
            feedback_id=plan.feedback_id,
            feedback_fingerprint=plan.feedback_fingerprint,
            subject_id=plan.subject_id,
            context_key=plan.context_key,
            occurred_at=plan.occurred_at,
            metadata=dict(plan.metadata),
        )
        feedback_items[event_key] = tuple((item.identity, item.outcome) for item in plan.items)

        self._events = events
        self._feedback_items = feedback_items
        self._states = states
        self._associations = associations

        return LearningWriteResult(
            created=True,
            unchanged=False,
            helpful=helpful,
            unhelpful=unhelpful,
            incorrect=incorrect,
            associations_reinforced=associations_reinforced,
        )

    async def list_states(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        context_keys: Sequence[str],
    ) -> Sequence[StoredMemoryLearningState]:
        if not identities or not context_keys:
            return ()
        identity_keys = {
            (identity.memory_kind.value, identity.memory_key) for identity in identities
        }
        context_set = set(context_keys)
        return tuple(
            state
            for state in self._states.values()
            if state.tenant_id == tenant_id
            and (state.memory_kind.value, state.memory_key) in identity_keys
            and state.context_key in context_set
        )

    async def list_associations(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
    ) -> Sequence[StoredMemoryAssociation]:
        if not identities:
            return ()
        identity_set = set(identities)
        results: list[StoredMemoryAssociation] = []
        for association in self._associations.values():
            if association.tenant_id != tenant_id:
                continue
            if association.left in identity_set and association.right in identity_set:
                results.append(association)
        results.sort(key=_association_sort_key)
        return tuple(results)

    async def list_reinforcement_traces(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        if not identities:
            return {}
        identity_set = set(identities)
        cutoff = before_or_at.astimezone(UTC)
        grouped: dict[MemoryIdentity, list[ActivationReferenceTrace]] = {}
        for event_key, event in self._events.items():
            if event_key[0] != tenant_id:
                continue
            if event.occurred_at > cutoff:
                continue
            items = self._feedback_items.get(event_key, ())
            for identity, outcome in items:
                if outcome is not LearningOutcome.HELPFUL:
                    continue
                if identity not in identity_set:
                    continue
                grouped.setdefault(identity, []).append(
                    ActivationReferenceTrace(
                        referenced_at=event.occurred_at,
                        weight=1,
                    )
                )
        return {
            identity: tuple(sorted(traces, key=lambda trace: trace.referenced_at))
            for identity, traces in grouped.items()
        }

    async def clear(self, *, tenant_id: str) -> None:
        self._events = {key: value for key, value in self._events.items() if key[0] != tenant_id}
        self._feedback_items = {
            key: value for key, value in self._feedback_items.items() if key[0] != tenant_id
        }
        self._states = {key: value for key, value in self._states.items() if key[0] != tenant_id}
        self._associations = {
            key: value for key, value in self._associations.items() if key[0] != tenant_id
        }


def _state_key(
    *,
    tenant_id: str,
    context_key: str,
    identity: MemoryIdentity,
) -> tuple[str, str, str, str]:
    return tenant_id, context_key, identity.memory_kind.value, identity.memory_key


def _increment_state(
    *,
    states: dict[tuple[str, str, str, str], StoredMemoryLearningState],
    tenant_id: str,
    context_key: str,
    identity: MemoryIdentity,
    outcome: LearningOutcome,
    at: datetime,
) -> None:
    timestamp = at.astimezone(UTC)
    key = _state_key(tenant_id=tenant_id, context_key=context_key, identity=identity)
    existing = states.get(key)
    if existing is None:
        helpful = 1 if outcome is LearningOutcome.HELPFUL else 0
        unhelpful = 1 if outcome is LearningOutcome.UNHELPFUL else 0
        incorrect = 1 if outcome is LearningOutcome.INCORRECT else 0
        states[key] = StoredMemoryLearningState(
            tenant_id=tenant_id,
            context_key=context_key,
            memory_kind=identity.memory_kind,
            memory_key=identity.memory_key,
            helpful_count=helpful,
            unhelpful_count=unhelpful,
            incorrect_count=incorrect,
            first_feedback_at=timestamp,
            last_feedback_at=timestamp,
            updated_at=timestamp,
        )
        return

    helpful = existing.helpful_count + (1 if outcome is LearningOutcome.HELPFUL else 0)
    unhelpful = existing.unhelpful_count + (1 if outcome is LearningOutcome.UNHELPFUL else 0)
    incorrect = existing.incorrect_count + (1 if outcome is LearningOutcome.INCORRECT else 0)
    states[key] = StoredMemoryLearningState(
        tenant_id=existing.tenant_id,
        context_key=existing.context_key,
        memory_kind=existing.memory_kind,
        memory_key=existing.memory_key,
        helpful_count=helpful,
        unhelpful_count=unhelpful,
        incorrect_count=incorrect,
        first_feedback_at=existing.first_feedback_at,
        last_feedback_at=timestamp,
        updated_at=timestamp,
    )


def _association_storage_key(
    *,
    tenant_id: str,
    left: MemoryIdentity,
    right: MemoryIdentity,
) -> tuple[str, str, str, str, str]:
    return (
        tenant_id,
        left.memory_kind.value,
        left.memory_key,
        right.memory_kind.value,
        right.memory_key,
    )


def _increment_association(
    *,
    associations: dict[tuple[str, str, str, str, str], StoredMemoryAssociation],
    tenant_id: str,
    left: MemoryIdentity,
    right: MemoryIdentity,
    at: datetime,
) -> None:
    timestamp = at.astimezone(UTC)
    key = _association_storage_key(tenant_id=tenant_id, left=left, right=right)
    existing = associations.get(key)
    if existing is None:
        associations[key] = StoredMemoryAssociation(
            tenant_id=tenant_id,
            left=left,
            right=right,
            coactivation_count=1,
            first_reinforced_at=timestamp,
            last_reinforced_at=timestamp,
            updated_at=timestamp,
        )
        return
    associations[key] = StoredMemoryAssociation(
        tenant_id=existing.tenant_id,
        left=existing.left,
        right=existing.right,
        coactivation_count=existing.coactivation_count + 1,
        first_reinforced_at=existing.first_reinforced_at,
        last_reinforced_at=timestamp,
        updated_at=timestamp,
    )


def _association_sort_key(association: StoredMemoryAssociation) -> tuple[str, ...]:
    return (
        association.left.memory_kind.value,
        association.left.memory_key,
        association.right.memory_kind.value,
        association.right.memory_key,
    )
