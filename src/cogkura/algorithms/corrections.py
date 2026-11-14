"""Explicit authoritative replacement and withdrawal of semantic revisions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from cogkura.algorithms.retrieval_features import (
    canonical_content_features,
    predicate_content_features,
)
from cogkura.algorithms.semantic import (
    _parse_optional_datetime,
    generate_revision_key,
    promotion_basis_value,
)
from cogkura.algorithms.semantic_diagnostics import EditDecisionObserver, SemanticDecisionCollector
from cogkura.exceptions import ValidationError
from cogkura.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    SemanticCardinality,
    SemanticDecisionOperation,
    SemanticDecisionReason,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticPromotionBasis,
    SemanticReconciliationPlan,
    SemanticRevisionCandidate,
    SemanticRevisionInput,
    SemanticRevisionRelation,
    SemanticUpdateRelation,
    StoredEpisode,
    StoredSemanticMemory,
    StoredSemanticRevision,
    semantic_status_answers_currently,
)
from cogkura.observations.models import (
    ObservationRevisionIdentity,
    StoredObservation,
    StoredObservationRevision,
)

SNAPSHOT_VERSION = "semantic-revision-snapshot-v1"
_EPISODE_FRAGMENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True, slots=True)
class ExplicitSemanticCorrection:
    """A validated request to replace one semantic revision."""

    successor: SemanticRevisionCandidate
    predecessor_revision_key: str
    effective_at: datetime
    knowledge_at: datetime
    observation_id: str
    observation_revision: int


@dataclass(frozen=True, slots=True)
class ExplicitSemanticWithdrawal:
    """A validated request to retract one semantic revision without a successor."""

    tenant_id: str
    target_revision_key: str
    effective_at: datetime
    knowledge_at: datetime
    observation_id: str
    observation_revision: int
    episode_id: str


@dataclass(frozen=True, slots=True)
class CorrectionQualification:
    """Corrections and withdrawals that may be applied to one reconciliation plan."""

    corrections: tuple[ExplicitSemanticCorrection, ...] = ()
    withdrawals: tuple[ExplicitSemanticWithdrawal, ...] = ()
    withhold_memory_keys: frozenset[str] = frozenset()
    deferred: int = 0
    invalid: int = 0


@dataclass(frozen=True, slots=True)
class ExplicitEditOutcome:
    """Plan after explicit corrections and withdrawals, plus newly persisted counts."""

    plan: SemanticReconciliationPlan
    corrections_applied: int = 0
    withdrawals_applied: int = 0


@dataclass(frozen=True, slots=True)
class _StaleEpisodeAssertion:
    """Lexical evidence needed to isolate one stale assertion in source prose."""

    object_terms: frozenset[str]
    predicate_terms: frozenset[str]


def qualify_explicit_corrections(
    *,
    enabled: bool,
    authority_enabled: bool,
    episodes: Sequence[StoredEpisode],
    observations: Mapping[str, StoredObservation],
    observation_revisions: Mapping[ObservationRevisionIdentity, StoredObservationRevision],
    candidates: Sequence[SemanticRevisionCandidate],
    existing_memories: Sequence[StoredSemanticMemory],
    existing_revisions: Sequence[StoredSemanticRevision],
    admitted_candidates: Sequence[SemanticRevisionCandidate],
    diagnostics: SemanticDecisionCollector | None = None,
) -> CorrectionQualification:
    """Validate correction and withdrawal declarations. Disabled mode is a no-op."""
    if not enabled:
        return CorrectionQualification()

    memories_by_key = {memory.memory_key: memory for memory in existing_memories}
    revisions_by_key = {revision.revision_key: revision for revision in existing_revisions}
    admitted_by_key = {candidate.revision_key: candidate for candidate in admitted_candidates}
    raw_corrections: list[ExplicitSemanticCorrection] = []
    raw_withdrawals: list[ExplicitSemanticWithdrawal] = []
    withhold: set[str] = set()
    invalid = 0
    deferred = 0

    for episode in episodes:
        for evidence in episode.evidence:
            observation = observations.get(evidence.observation_id)
            if observation is None or observation.tenant_id != episode.tenant_id:
                continue
            identity = ObservationRevisionIdentity(
                observation_id=evidence.observation_id,
                observation_revision=evidence.observation_revision,
            )
            revision = observation_revisions.get(identity)
            if diagnostics is not None and revision is None:
                diagnostics.complete = False
            invalid_count, deferred_count = _collect_fact_corrections(
                diagnostics=diagnostics,
                observation=observation,
                revision=revision,
                evidence_revision=evidence.observation_revision,
                authority_enabled=authority_enabled,
                candidates=candidates,
                memories_by_key=memories_by_key,
                revisions_by_key=revisions_by_key,
                admitted_by_key=admitted_by_key,
                existing_memories=existing_memories,
                admitted_candidates=admitted_candidates,
                raw_corrections=raw_corrections,
                withhold=withhold,
            )
            invalid += invalid_count
            deferred += deferred_count
            withdrawal_invalid, withdrawal_deferred = _collect_withdrawals(
                diagnostics=diagnostics,
                observation=observation,
                revision=revision,
                evidence_revision=evidence.observation_revision,
                episode_id=episode.id,
                authority_enabled=authority_enabled,
                memories_by_key=memories_by_key,
                revisions_by_key=revisions_by_key,
                admitted_by_key=admitted_by_key,
                existing_memories=existing_memories,
                admitted_candidates=admitted_candidates,
                raw_withdrawals=raw_withdrawals,
            )
            invalid += withdrawal_invalid
            deferred += withdrawal_deferred

    kept_corrections, dropped, extra_deferred = _drop_competing_corrections(
        raw_corrections, diagnostics=diagnostics
    )
    deferred += extra_deferred
    withhold.update(dropped)
    kept_withdrawals, withdrawal_dropped, withdrawal_deferred = _drop_competing_withdrawals(
        raw_withdrawals, diagnostics=diagnostics
    )
    deferred += withdrawal_deferred
    kept_corrections, kept_withdrawals, cross_deferred = _drop_cross_operation_conflicts(
        kept_corrections,
        kept_withdrawals,
        diagnostics=diagnostics,
    )
    deferred += cross_deferred
    withhold.update(
        item.successor.memory_key for item in raw_corrections if item not in kept_corrections
    )
    return CorrectionQualification(
        corrections=tuple(kept_corrections),
        withdrawals=tuple(kept_withdrawals),
        withhold_memory_keys=frozenset(withhold),
        deferred=deferred,
        invalid=invalid + withdrawal_dropped,
    )


def apply_explicit_semantic_edits(
    plan: SemanticReconciliationPlan,
    corrections: Sequence[ExplicitSemanticCorrection],
    withdrawals: Sequence[ExplicitSemanticWithdrawal],
    *,
    diagnostics: SemanticDecisionCollector | None = None,
) -> ExplicitEditOutcome:
    """Apply already validated edits without creating a second reconciliation engine."""
    revisions = {revision.revision_key: revision for revision in plan.revisions}
    memories = {memory.memory_key: memory for memory in plan.current_memories}
    relations = {
        (
            relation.tenant_id,
            relation.left_revision_key,
            relation.right_revision_key,
            relation.relation,
        ): relation
        for relation in plan.relations
    }
    created = plan.revisions_created
    updated = plan.revisions_updated
    superseded = plan.superseded_count
    corrections_applied = 0
    withdrawals_applied = 0

    for correction in sorted(
        corrections,
        key=lambda item: (item.predecessor_revision_key, item.effective_at.isoformat()),
    ):
        applied, created, updated, superseded = _apply_correction(
            correction,
            revisions=revisions,
            memories=memories,
            relations=relations,
            created=created,
            updated=updated,
            superseded=superseded,
        )
        if applied:
            _edit_observer(diagnostics, correction).decide(
                "committed", SemanticDecisionReason.APPLIED
            )
        corrections_applied += applied

    for withdrawal in sorted(
        withdrawals,
        key=lambda item: (item.target_revision_key, item.effective_at.isoformat()),
    ):
        applied, created, updated = _apply_withdrawal(
            withdrawal,
            revisions=revisions,
            memories=memories,
            relations=relations,
            created=created,
            updated=updated,
        )
        _edit_observer(diagnostics, withdrawal).decide(
            "committed" if applied else "applied",
            SemanticDecisionReason.APPLIED if applied else SemanticDecisionReason.ALREADY_APPLIED,
        )
        withdrawals_applied += applied

    return ExplicitEditOutcome(
        plan=replace(
            plan,
            current_memories=tuple(memories[key] for key in sorted(memories)),
            revisions=tuple(revisions[key] for key in sorted(revisions)),
            relations=tuple(relations[key] for key in sorted(relations, key=str)),
            revisions_created=created,
            revisions_updated=updated,
            superseded_count=superseded,
        ),
        corrections_applied=corrections_applied,
        withdrawals_applied=withdrawals_applied,
    )


def build_snapshot_from_candidate(
    candidate: SemanticRevisionCandidate,
    *,
    status: SemanticMemoryStatus,
    valid_from: datetime | None,
    valid_until: datetime | None,
    confidence: float,
    importance: float,
    support_count: int,
    contradiction_count: int,
    derivations: Sequence[SemanticDerivationInput],
    knowledge_at: datetime | None = None,
    correction: Mapping[str, Any] | None = None,
    withdrawal: Mapping[str, Any] | None = None,
    closed_by: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready snapshot of one semantic revision."""
    return _snapshot(
        statement=candidate.statement,
        slot_key=candidate.slot_key,
        subject_entity_id=candidate.subject_entity_id,
        predicate=candidate.predicate,
        object_value=candidate.object_value,
        object_entity_id=candidate.object_entity_id,
        polarity=candidate.polarity,
        cardinality=candidate.cardinality,
        qualifiers=candidate.qualifiers,
        status=status,
        valid_from=valid_from,
        valid_until=valid_until,
        confidence=confidence,
        importance=importance,
        support_count=support_count,
        contradiction_count=contradiction_count,
        observation_evidence=candidate.observation_evidence,
        entities=candidate.entities,
        metadata=candidate.metadata,
        derivations=derivations,
        knowledge_at=knowledge_at,
        correction=correction,
        withdrawal=withdrawal,
        closed_by=closed_by,
    )


def snapshot_is_present(snapshot: Mapping[str, Any]) -> bool:
    return snapshot.get("version") == SNAPSHOT_VERSION


def memory_from_revision_snapshot(
    memory: StoredSemanticMemory,
    revision: StoredSemanticRevision,
) -> StoredSemanticMemory | None:
    """Rebuild a historical memory from a revision snapshot when one was stored."""
    snapshot = revision.snapshot
    if not snapshot_is_present(snapshot):
        return None
    try:
        polarity = SemanticPolarity(str(snapshot["polarity"]))
        cardinality = SemanticCardinality(str(snapshot["cardinality"]))
        status = SemanticMemoryStatus(str(snapshot["status"]))
    except (KeyError, ValueError):
        return None
    evidence = _evidence_from_snapshot(snapshot.get("observation_evidence", ()))
    entities = _entities_from_snapshot(snapshot.get("entities", ()))
    derivations = _derivations_from_snapshot(snapshot.get("derivations", ()))
    metadata = snapshot.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    qualifiers = snapshot.get("qualifiers", {})
    if not isinstance(qualifiers, Mapping):
        qualifiers = {}
    return StoredSemanticMemory(
        id=memory.id,
        tenant_id=memory.tenant_id,
        subject_id=memory.subject_id,
        memory_key=revision.memory_key,
        slot_key=str(snapshot.get("slot_key") or memory.slot_key),
        revision_key=revision.revision_key,
        revision_number=revision.revision_number,
        statement=str(snapshot.get("statement") or memory.statement),
        subject_entity_id=_optional_str(snapshot.get("subject_entity_id")),
        predicate=str(snapshot.get("predicate") or memory.predicate),
        object_value=str(snapshot.get("object_value") or memory.object_value),
        object_entity_id=_optional_str(snapshot.get("object_entity_id")),
        polarity=polarity,
        cardinality=cardinality,
        qualifiers=MappingProxyType(dict(qualifiers)),
        confidence=float(snapshot.get("confidence", revision.confidence)),
        importance=float(snapshot.get("importance", revision.importance)),
        status=status,
        support_count=int(snapshot.get("support_count", revision.support_count)),
        contradiction_count=int(snapshot.get("contradiction_count", revision.contradiction_count)),
        first_supported_at=revision.first_supported_at,
        last_supported_at=revision.last_supported_at,
        valid_from=_parse_snapshot_datetime(snapshot.get("valid_from")),
        valid_until=_parse_snapshot_datetime(snapshot.get("valid_until")),
        is_active=True,
        derivations=derivations or revision.derivations,
        observation_evidence=evidence,
        entities=entities,
        metadata=MappingProxyType(dict(metadata)),
        created_at=memory.created_at,
        updated_at=revision.updated_at,
    )


def project_known_semantics(
    current: Sequence[StoredSemanticMemory],
    superseded: Sequence[StoredSemanticMemory],
    revisions: Sequence[StoredSemanticRevision],
    *,
    as_of: datetime,
    valid_at: datetime | None,
    retracted: Sequence[StoredSemanticMemory] = (),
    projection_bases: Sequence[StoredSemanticMemory] = (),
) -> list[StoredSemanticMemory]:
    """Hide a correction or withdrawal that is not yet known or not yet effective."""
    by_key = {revision.revision_key: revision for revision in revisions}
    bases_by_memory: dict[str, StoredSemanticMemory] = {}
    for memory in (*current, *superseded, *retracted, *projection_bases):
        bases_by_memory.setdefault(memory.memory_key, memory)
    hidden = {
        revision.revision_key
        for revision in revisions
        if _transition_hidden(revision, as_of=as_of, valid_at=valid_at)
    }
    visible = [memory for memory in current if memory.revision_key not in hidden]
    if valid_at is None:
        visible = [memory for memory in visible if semantic_status_answers_currently(memory.status)]
    else:
        visible = [
            memory for memory in visible if memory.status is not SemanticMemoryStatus.RETRACTED
        ]
    known_keys = {memory.revision_key for memory in visible}
    for revision in revisions:
        if revision.revision_key not in hidden:
            continue
        predecessor_key = _predecessor_key(revision.snapshot)
        if predecessor_key is None or predecessor_key in known_keys or predecessor_key in hidden:
            continue
        predecessor = by_key.get(predecessor_key)
        if predecessor is None:
            continue
        if valid_at is not None and valid_at < (
            predecessor.valid_from or predecessor.first_supported_at
        ):
            continue
        base = bases_by_memory.get(predecessor.memory_key)
        if base is None:
            continue
        restored = memory_from_revision_snapshot(base, predecessor)
        if restored is None:
            continue
        if valid_at is None:
            restored = replace(restored, status=SemanticMemoryStatus.ACTIVE, valid_until=None)
        visible.append(restored)
        known_keys.add(predecessor_key)
    return visible


def _transition_hidden(
    revision: StoredSemanticRevision,
    *,
    as_of: datetime,
    valid_at: datetime | None,
) -> bool:
    if _predecessor_key(revision.snapshot) is None:
        return False
    knowledge = _knowledge_at(revision.snapshot)
    if knowledge is not None and knowledge > as_of:
        return True
    effective = _boundary_effective_at(revision.snapshot)
    boundary = valid_at if valid_at is not None else as_of
    return effective is not None and effective > boundary


def _boundary_effective_at(snapshot: Mapping[str, Any]) -> datetime | None:
    for key in ("correction", "withdrawal"):
        block = snapshot.get(key)
        if isinstance(block, Mapping):
            parsed = _parse_snapshot_datetime(block.get("effective_at"))
            if parsed is not None:
                return parsed
    return None


def unknown_correction_episode_ids(
    revisions: Sequence[StoredSemanticRevision],
    *,
    as_of: datetime,
) -> set[str]:
    """Hide only the episode that carries a correction or withdrawal not yet known."""
    hidden: set[str] = set()
    for revision in revisions:
        knowledge = _knowledge_at(revision.snapshot)
        if knowledge is None or knowledge <= as_of:
            continue
        withdrawal = revision.snapshot.get("withdrawal")
        if isinstance(withdrawal, Mapping):
            episode_id = withdrawal.get("episode_id")
            if isinstance(episode_id, str) and episode_id:
                hidden.add(episode_id)
            continue
        if isinstance(revision.snapshot.get("correction"), Mapping):
            for derivation in revision.derivations:
                hidden.add(derivation.episode_id)
    return hidden


def hidden_current_episode_ids(
    revisions: Sequence[StoredSemanticRevision],
    *,
    as_of: datetime,
) -> set[str]:
    """Return no whole-episode exclusions.

    Current answers come from the semantic assertion. Hiding an entire supporting
    episode would also hide unrelated facts encoded in that episode.
    """
    del revisions, as_of
    return set()


def selectively_filter_stale_episode_content(
    episodes: Sequence[StoredEpisode],
    revisions: Sequence[StoredSemanticRevision],
    *,
    as_of: datetime,
    valid_at: datetime | None,
) -> list[StoredEpisode]:
    """Remove known stale assertion fragments from current-state episode reads.

    Semantic revisions identify the supported episode and the assertion's object
    value plus predicate. We remove only complete sentence-like fragments that
    contain both, so another assertion that shares a generic value is preserved.
    If the source cannot be separated safely, the episode is omitted rather than
    leaking stale context.
    """
    stale_assertions = _stale_episode_assertions(
        revisions,
        as_of=as_of,
        valid_at=valid_at,
    )
    if not stale_assertions:
        return list(episodes)
    filtered: list[StoredEpisode] = []
    for episode in episodes:
        assertions = stale_assertions.get(episode.id)
        if not assertions:
            filtered.append(episode)
            continue
        fragments = [
            fragment.strip()
            for fragment in _EPISODE_FRAGMENT_BOUNDARY.split(episode.statement)
            if fragment.strip()
        ]
        if not fragments:
            continue
        safe_fragments = [
            fragment
            for fragment in fragments
            if not any(
                _fragment_carries_assertion(canonical_content_features(fragment), assertion)
                for assertion in assertions
            )
        ]
        if not safe_fragments:
            continue
        if len(safe_fragments) == len(fragments):
            # The assertion cannot be located in source text, so retaining the
            # whole episode would be unsafe for a current-state read.
            continue
        filtered.append(replace(episode, statement="\n".join(safe_fragments)))
    return filtered


def _stale_episode_assertions(
    revisions: Sequence[StoredSemanticRevision],
    *,
    as_of: datetime,
    valid_at: datetime | None,
) -> dict[str, set[_StaleEpisodeAssertion]]:
    by_key = {revision.revision_key: revision for revision in revisions}
    stale: dict[str, set[_StaleEpisodeAssertion]] = {}
    for revision in revisions:
        predecessor_key = _predecessor_key(revision.snapshot)
        if predecessor_key is None or _transition_hidden(
            revision,
            as_of=as_of,
            valid_at=valid_at,
        ):
            continue
        predecessor = by_key.get(predecessor_key)
        if predecessor is None:
            continue
        object_value = predecessor.snapshot.get("object_value")
        assertion = _StaleEpisodeAssertion(
            object_terms=canonical_content_features(str(object_value or "")),
            predicate_terms=predicate_content_features(
                _optional_str(predecessor.snapshot.get("predicate"))
            ),
        )
        for derivation in predecessor.derivations:
            if derivation.relation is SemanticDerivationRelation.SUPPORTS:
                stale.setdefault(derivation.episode_id, set()).add(assertion)
    return stale


def _fragment_carries_assertion(
    fragment_terms: frozenset[str],
    assertion: _StaleEpisodeAssertion,
) -> bool:
    """Require object identity and a predicate anchor before redacting prose."""
    if not assertion.object_terms or not assertion.predicate_terms:
        return False
    if not assertion.object_terms.issubset(fragment_terms):
        return False
    return any(
        _features_share_lexeme(predicate_term, fragment_term)
        for predicate_term in assertion.predicate_terms
        for fragment_term in fragment_terms
    )


def _features_share_lexeme(left: str, right: str) -> bool:
    """Match a predicate feature to normal singular/plural source prose."""
    return left == right or left == right.rstrip("s") or right == left.rstrip("s")


def _collect_fact_corrections(
    *,
    observation: StoredObservation,
    diagnostics: SemanticDecisionCollector | None = None,
    revision: StoredObservationRevision | None,
    evidence_revision: int,
    authority_enabled: bool,
    candidates: Sequence[SemanticRevisionCandidate],
    memories_by_key: Mapping[str, StoredSemanticMemory],
    revisions_by_key: Mapping[str, StoredSemanticRevision],
    admitted_by_key: Mapping[str, SemanticRevisionCandidate],
    existing_memories: Sequence[StoredSemanticMemory],
    admitted_candidates: Sequence[SemanticRevisionCandidate],
    raw_corrections: list[ExplicitSemanticCorrection],
    withhold: set[str],
) -> tuple[int, int]:
    invalid = 0
    deferred = 0
    source_facts = _raw_facts_from_revision(revision) if revision is not None else ()
    for raw_fact in source_facts:
        if not isinstance(raw_fact, Mapping) or "correction" not in raw_fact:
            continue
        parsed, malformed = _parse_correction(raw_fact)
        successor = _matching_candidate(candidates, observation.id, raw_fact)
        if successor is not None:
            withhold.add(successor.memory_key)
        observer = EditDecisionObserver(
            diagnostics,
            operation=SemanticDecisionOperation.REPLACE,
            observation_id=observation.id,
            observation_revision=evidence_revision,
            target_key=parsed["replaces_revision_key"] if parsed else None,
            effective_at=parsed["effective_at"] if parsed else None,
            slot_key=successor.slot_key if successor else None,
            claim_key=successor.memory_key if successor else None,
            resulting_revision_key=generate_revision_key(
                successor.memory_key, valid_from=parsed["effective_at"], valid_until=None
            )
            if successor and parsed
            else None,
        )
        if malformed or parsed is None:
            observer.decide("invalid", SemanticDecisionReason.MALFORMED_DECLARATION)
            invalid += 1
            continue
        if not authority_enabled or not _fact_has_authority(raw_fact):
            observer.decide("invalid", SemanticDecisionReason.AUTHORITY_REQUIRED)
            invalid += 1
            continue
        if (
            revision is None
            or revision.is_deleted
            or revision.current_revision != evidence_revision
        ):
            observer.decide("invalid", SemanticDecisionReason.STALE_OR_DELETED_AUTHORIZING_SOURCE)
            invalid += 1
            continue
        if successor is None or successor.cardinality is not SemanticCardinality.ONE:
            observer.decide("invalid", SemanticDecisionReason.UNSUPPORTED_CARDINALITY)
            invalid += 1
            continue
        evidence_time = revision.observed_at
        if parsed["effective_at"] < evidence_time:
            observer.decide("invalid", SemanticDecisionReason.INVALID_EFFECTIVE_TIME)
            invalid += 1
            continue
        outcome = _evaluate_predecessor(
            predecessor_key=parsed["replaces_revision_key"],
            successor=successor,
            effective_at=parsed["effective_at"],
            memories_by_key=memories_by_key,
            revisions_by_key=revisions_by_key,
            admitted_by_key=admitted_by_key,
            observer=observer,
            existing_memories=existing_memories,
            admitted_candidates=admitted_candidates,
        )
        if outcome == "invalid":
            invalid += 1
            continue
        if outcome == "deferred":
            deferred += 1
            continue
        if outcome == "applied":
            continue
        raw_corrections.append(
            ExplicitSemanticCorrection(
                successor=successor,
                predecessor_revision_key=parsed["replaces_revision_key"],
                effective_at=parsed["effective_at"],
                knowledge_at=evidence_time,
                observation_id=observation.id,
                observation_revision=evidence_revision,
            )
        )
    return invalid, deferred


def _collect_withdrawals(
    *,
    observation: StoredObservation,
    diagnostics: SemanticDecisionCollector | None = None,
    revision: StoredObservationRevision | None,
    evidence_revision: int,
    episode_id: str,
    authority_enabled: bool,
    memories_by_key: Mapping[str, StoredSemanticMemory],
    revisions_by_key: Mapping[str, StoredSemanticRevision],
    admitted_by_key: Mapping[str, SemanticRevisionCandidate],
    existing_memories: Sequence[StoredSemanticMemory],
    admitted_candidates: Sequence[SemanticRevisionCandidate],
    raw_withdrawals: list[ExplicitSemanticWithdrawal],
) -> tuple[int, int]:
    invalid = 0
    deferred = 0
    source = revision.metadata if revision is not None else {}
    raw_items = source.get("semantic_withdrawals", ()) if isinstance(source, Mapping) else ()
    if raw_items is None:
        return 0, 0
    if isinstance(raw_items, Mapping):
        raw_items = (raw_items,)
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        if diagnostics is not None:
            EditDecisionObserver(
                diagnostics,
                operation=SemanticDecisionOperation.WITHDRAW,
                observation_id=observation.id,
                observation_revision=evidence_revision,
                target_key=None,
                effective_at=None,
            ).decide("invalid", SemanticDecisionReason.MALFORMED_DECLARATION)
        return 1, 0
    for raw_item in raw_items:
        parsed, malformed = _parse_withdrawal(raw_item)
        target = revisions_by_key.get(parsed["target_revision_key"]) if parsed else None
        target_memory = memories_by_key.get(target.memory_key) if target else None
        observer = EditDecisionObserver(
            diagnostics,
            operation=SemanticDecisionOperation.WITHDRAW,
            observation_id=observation.id,
            observation_revision=evidence_revision,
            target_key=parsed["target_revision_key"] if parsed else None,
            effective_at=parsed["effective_at"] if parsed else None,
            slot_key=target_memory.slot_key if target_memory else None,
            claim_key=target.memory_key if target else None,
            resulting_revision_key=generate_revision_key(
                parsed["target_revision_key"], valid_from=parsed["effective_at"], valid_until=None
            )
            if parsed
            else None,
        )
        if malformed or parsed is None:
            observer.decide("invalid", SemanticDecisionReason.MALFORMED_DECLARATION)
            invalid += 1
            continue
        if not authority_enabled:
            observer.decide("invalid", SemanticDecisionReason.AUTHORITY_REQUIRED)
            invalid += 1
            continue
        if (
            revision is None
            or revision.is_deleted
            or revision.current_revision != evidence_revision
        ):
            observer.decide("invalid", SemanticDecisionReason.STALE_OR_DELETED_AUTHORIZING_SOURCE)
            invalid += 1
            continue
        evidence_time = revision.observed_at
        if parsed["effective_at"] < evidence_time:
            observer.decide("invalid", SemanticDecisionReason.INVALID_EFFECTIVE_TIME)
            invalid += 1
            continue
        outcome = _evaluate_withdrawal_target(
            target_key=parsed["target_revision_key"],
            effective_at=parsed["effective_at"],
            tenant_id=observation.tenant_id,
            memories_by_key=memories_by_key,
            revisions_by_key=revisions_by_key,
            admitted_by_key=admitted_by_key,
            observer=observer,
            existing_memories=existing_memories,
            admitted_candidates=admitted_candidates,
        )
        if outcome == "invalid":
            invalid += 1
            continue
        if outcome == "deferred":
            deferred += 1
            continue
        if outcome == "applied":
            continue
        raw_withdrawals.append(
            ExplicitSemanticWithdrawal(
                tenant_id=observation.tenant_id,
                target_revision_key=parsed["target_revision_key"],
                effective_at=parsed["effective_at"],
                knowledge_at=evidence_time,
                observation_id=observation.id,
                observation_revision=evidence_revision,
                episode_id=episode_id,
            )
        )
    return invalid, deferred


def _evaluate_predecessor(
    *,
    predecessor_key: str,
    successor: SemanticRevisionCandidate,
    effective_at: datetime,
    memories_by_key: Mapping[str, StoredSemanticMemory],
    revisions_by_key: Mapping[str, StoredSemanticRevision],
    admitted_by_key: Mapping[str, SemanticRevisionCandidate],
    existing_memories: Sequence[StoredSemanticMemory],
    observer: EditDecisionObserver | None = None,
    admitted_candidates: Sequence[SemanticRevisionCandidate],
) -> str:
    stored = revisions_by_key.get(predecessor_key)
    batch = admitted_by_key.get(predecessor_key)
    if stored is None and batch is None:
        return _observed_edit_outcome(
            "deferred", SemanticDecisionReason.TARGET_UNAVAILABLE, observer
        )
    if stored is not None and stored.status in (
        SemanticMemoryStatus.SUPERSEDED,
        SemanticMemoryStatus.RETRACTED,
    ):
        if _correction_already_applied(stored, successor, effective_at, revisions_by_key):
            return _observed_edit_outcome(
                "applied", SemanticDecisionReason.ALREADY_APPLIED, observer
            )
        return _observed_edit_outcome(
            "invalid", SemanticDecisionReason.TARGET_ALREADY_CLOSED, observer
        )
    memory_key = stored.memory_key if stored is not None else batch.memory_key if batch else ""
    slot_key = _slot_key(stored, batch, memories_by_key)
    cardinality = _cardinality(stored, batch, memories_by_key)
    if slot_key is None or cardinality is not SemanticCardinality.ONE:
        return _observed_edit_outcome(
            "invalid", SemanticDecisionReason.UNSUPPORTED_CARDINALITY, observer
        )
    if slot_key != successor.slot_key or memory_key == successor.memory_key:
        return _observed_edit_outcome(
            "invalid",
            SemanticDecisionReason.WRONG_SLOT
            if slot_key != successor.slot_key
            else SemanticDecisionReason.SAME_CLAIM_REPLACEMENT,
            observer,
        )
    start = _effective_start(stored, batch)
    if start is None or effective_at <= start:
        return _observed_edit_outcome(
            "invalid", SemanticDecisionReason.INVALID_EFFECTIVE_TIME, observer
        )
    if not _unique_current_slot(
        slot_key=slot_key,
        predecessor_key=predecessor_key,
        ignore_revision_keys={successor.revision_key},
        existing_memories=existing_memories,
        admitted_candidates=admitted_candidates,
        revisions_by_key=revisions_by_key,
    ):
        return _observed_edit_outcome(
            "deferred", SemanticDecisionReason.TARGET_NOT_UNIQUE_CURRENT, observer
        )
    if stored is not None and stored.status is SemanticMemoryStatus.CONTESTED:
        return _observed_edit_outcome("deferred", SemanticDecisionReason.TARGET_CONTESTED, observer)
    return "ok"


def _evaluate_withdrawal_target(
    *,
    target_key: str,
    effective_at: datetime,
    tenant_id: str,
    memories_by_key: Mapping[str, StoredSemanticMemory],
    revisions_by_key: Mapping[str, StoredSemanticRevision],
    admitted_by_key: Mapping[str, SemanticRevisionCandidate],
    existing_memories: Sequence[StoredSemanticMemory],
    observer: EditDecisionObserver | None = None,
    admitted_candidates: Sequence[SemanticRevisionCandidate],
) -> str:
    stored = revisions_by_key.get(target_key)
    batch = admitted_by_key.get(target_key)
    if stored is None and batch is None:
        return _observed_edit_outcome(
            "deferred", SemanticDecisionReason.TARGET_UNAVAILABLE, observer
        )
    if stored is not None and stored.tenant_id != tenant_id:
        return _observed_edit_outcome("invalid", SemanticDecisionReason.WRONG_SLOT, observer)
    if stored is not None and stored.status is SemanticMemoryStatus.RETRACTED:
        return _observed_edit_outcome(
            "applied" if _withdrawal_already_applied(stored, effective_at) else "invalid",
            SemanticDecisionReason.ALREADY_APPLIED,
            observer,
        )
    if stored is not None and stored.status is SemanticMemoryStatus.SUPERSEDED:
        return _observed_edit_outcome(
            "invalid", SemanticDecisionReason.TARGET_ALREADY_CLOSED, observer
        )
    slot_key = _slot_key(stored, batch, memories_by_key)
    cardinality = _cardinality(stored, batch, memories_by_key)
    if slot_key is None or cardinality is not SemanticCardinality.ONE:
        return _observed_edit_outcome(
            "invalid", SemanticDecisionReason.UNSUPPORTED_CARDINALITY, observer
        )
    start = _effective_start(stored, batch)
    if start is None or effective_at <= start:
        return _observed_edit_outcome(
            "invalid", SemanticDecisionReason.INVALID_EFFECTIVE_TIME, observer
        )
    if not _unique_current_slot(
        slot_key=slot_key,
        predecessor_key=target_key,
        existing_memories=existing_memories,
        admitted_candidates=admitted_candidates,
        revisions_by_key=revisions_by_key,
    ):
        return _observed_edit_outcome(
            "deferred", SemanticDecisionReason.TARGET_NOT_UNIQUE_CURRENT, observer
        )
    return "ok"


def _correction_already_applied(
    predecessor: StoredSemanticRevision,
    successor: SemanticRevisionCandidate,
    effective_at: datetime,
    revisions_by_key: Mapping[str, StoredSemanticRevision],
) -> bool:
    if predecessor.status is not SemanticMemoryStatus.SUPERSEDED:
        return False
    if predecessor.valid_until != effective_at:
        return False
    successor_key = generate_revision_key(
        successor.memory_key,
        valid_from=effective_at,
        valid_until=None,
    )
    return successor_key in revisions_by_key


def _withdrawal_already_applied(
    target: StoredSemanticRevision,
    effective_at: datetime,
) -> bool:
    closed = target.snapshot.get("closed_by")
    if not isinstance(closed, Mapping) or closed.get("kind") != "withdrawal":
        return target.status is SemanticMemoryStatus.RETRACTED
    parsed = _parse_snapshot_datetime(closed.get("effective_at"))
    return parsed == effective_at


def _unique_current_slot(
    *,
    slot_key: str,
    predecessor_key: str,
    existing_memories: Sequence[StoredSemanticMemory],
    admitted_candidates: Sequence[SemanticRevisionCandidate],
    revisions_by_key: Mapping[str, StoredSemanticRevision],
    ignore_revision_keys: set[str] | None = None,
) -> bool:
    ignored = ignore_revision_keys or set()
    occupants: set[str] = set()
    for memory in existing_memories:
        if memory.slot_key != slot_key:
            continue
        if memory.status in (SemanticMemoryStatus.ACTIVE, SemanticMemoryStatus.CONTESTED):
            occupants.add(memory.revision_key)
    if predecessor_key in {candidate.revision_key for candidate in admitted_candidates}:
        occupants.add(predecessor_key)
    for candidate in admitted_candidates:
        if candidate.slot_key != slot_key or candidate.revision_key in ignored:
            continue
        if promotion_basis_value(candidate.metadata) != SemanticPromotionBasis.RECURRENCE.value:
            continue
        occupants.add(candidate.revision_key)
    for revision in revisions_by_key.values():
        if revision.revision_key == predecessor_key and revision.status in (
            SemanticMemoryStatus.SUPERSEDED,
            SemanticMemoryStatus.RETRACTED,
        ):
            occupants.discard(revision.revision_key)
    return occupants == {predecessor_key}


def _drop_cross_operation_conflicts(
    corrections: list[ExplicitSemanticCorrection],
    withdrawals: list[ExplicitSemanticWithdrawal],
    *,
    diagnostics: SemanticDecisionCollector | None = None,
) -> tuple[list[ExplicitSemanticCorrection], list[ExplicitSemanticWithdrawal], int]:
    """One predecessor can receive only one explicit operation in a pass."""
    correction_keys = {item.predecessor_revision_key for item in corrections}
    withdrawal_keys = {item.target_revision_key for item in withdrawals}
    conflict = correction_keys & withdrawal_keys
    if not conflict:
        return corrections, withdrawals, 0
    kept_corrections = [
        item for item in corrections if item.predecessor_revision_key not in conflict
    ]
    kept_withdrawals = [item for item in withdrawals if item.target_revision_key not in conflict]
    for correction in corrections:
        if correction.predecessor_revision_key in conflict:
            _edit_observer(diagnostics, correction).decide(
                "deferred", SemanticDecisionReason.REPLACE_WITHDRAW_CONFLICT
            )
    for withdrawal in withdrawals:
        if withdrawal.target_revision_key in conflict:
            _edit_observer(diagnostics, withdrawal).decide(
                "deferred", SemanticDecisionReason.REPLACE_WITHDRAW_CONFLICT
            )
    deferred = (len(corrections) - len(kept_corrections)) + (
        len(withdrawals) - len(kept_withdrawals)
    )
    return kept_corrections, kept_withdrawals, deferred


def _drop_competing_corrections(
    corrections: Sequence[ExplicitSemanticCorrection],
    *,
    diagnostics: SemanticDecisionCollector | None = None,
) -> tuple[list[ExplicitSemanticCorrection], set[str], int]:
    grouped: dict[str, list[ExplicitSemanticCorrection]] = {}
    for correction in corrections:
        grouped.setdefault(correction.predecessor_revision_key, []).append(correction)
    kept: list[ExplicitSemanticCorrection] = []
    dropped: set[str] = set()
    deferred = 0
    for group in grouped.values():
        identities = {
            (
                item.successor.memory_key,
                item.effective_at.isoformat(),
                item.observation_id,
                item.observation_revision,
            )
            for item in group
        }
        if len(identities) == 1:
            kept.append(group[0])
            continue
        for item in group:
            _edit_observer(diagnostics, item).decide(
                "deferred", SemanticDecisionReason.COMPETING_CORRECTIONS
            )
        deferred += len(group)
        dropped.update(item.successor.memory_key for item in group)
    return kept, dropped, deferred


def _drop_competing_withdrawals(
    withdrawals: Sequence[ExplicitSemanticWithdrawal],
    *,
    diagnostics: SemanticDecisionCollector | None = None,
) -> tuple[list[ExplicitSemanticWithdrawal], int, int]:
    grouped: dict[str, list[ExplicitSemanticWithdrawal]] = {}
    for withdrawal in withdrawals:
        grouped.setdefault(withdrawal.target_revision_key, []).append(withdrawal)
    kept: list[ExplicitSemanticWithdrawal] = []
    invalid = 0
    deferred = 0
    for group in grouped.values():
        identities = {
            (item.effective_at.isoformat(), item.observation_id, item.observation_revision)
            for item in group
        }
        if len(identities) == 1:
            kept.append(group[0])
            continue
        for item in group:
            _edit_observer(diagnostics, item).decide(
                "deferred", SemanticDecisionReason.COMPETING_WITHDRAWALS
            )
        deferred += len(group)
        invalid += 0
    return kept, invalid, deferred


def _apply_correction(
    correction: ExplicitSemanticCorrection,
    *,
    revisions: dict[str, SemanticRevisionInput],
    memories: dict[str, SemanticMemoryInput],
    relations: dict[tuple[str, str, str, SemanticUpdateRelation], SemanticRevisionRelation],
    created: int,
    updated: int,
    superseded: int,
) -> tuple[int, int, int, int]:
    predecessor = revisions.get(correction.predecessor_revision_key)
    if predecessor is None:
        return 0, created, updated, superseded
    successor_key = generate_revision_key(
        correction.successor.memory_key,
        valid_from=correction.effective_at,
        valid_until=None,
    )
    if (
        predecessor.status is SemanticMemoryStatus.SUPERSEDED
        and predecessor.valid_until == correction.effective_at
        and successor_key in revisions
    ):
        return 0, created, updated, superseded
    start = predecessor.valid_from or predecessor.first_supported_at
    closed_by = {
        "kind": "correction",
        "successor_revision_key": successor_key,
        "effective_at": correction.effective_at.isoformat(),
        "knowledge_at": correction.knowledge_at.isoformat(),
    }
    predecessor_memory = memories.get(predecessor.memory_key)
    closed_snapshot = _closed_snapshot(
        predecessor,
        predecessor_memory,
        valid_from=start,
        valid_until=correction.effective_at,
        closed_by=closed_by,
    )
    revisions[predecessor.revision_key] = replace(
        predecessor,
        status=SemanticMemoryStatus.SUPERSEDED,
        valid_from=start,
        valid_until=correction.effective_at,
        snapshot=MappingProxyType(closed_snapshot),
    )
    updated += 1
    superseded += 1
    if predecessor_memory is not None:
        memories[predecessor.memory_key] = replace(
            predecessor_memory,
            status=SemanticMemoryStatus.SUPERSEDED,
            valid_from=start,
            valid_until=correction.effective_at,
            revision_key=predecessor.revision_key,
        )
    successor_snapshot = build_snapshot_from_candidate(
        correction.successor,
        status=SemanticMemoryStatus.ACTIVE,
        valid_from=correction.effective_at,
        valid_until=None,
        confidence=correction.successor.support_confidence,
        importance=correction.successor.importance,
        support_count=correction.successor.support_count,
        contradiction_count=0,
        derivations=correction.successor.derivations,
        knowledge_at=correction.knowledge_at,
        correction={
            "replaces_revision_key": correction.predecessor_revision_key,
            "effective_at": correction.effective_at.isoformat(),
        },
    )
    number = _next_revision_number(revisions, correction.successor.memory_key)
    revisions[successor_key] = SemanticRevisionInput(
        tenant_id=correction.successor.tenant_id,
        memory_key=correction.successor.memory_key,
        revision_key=successor_key,
        revision_number=number,
        status=SemanticMemoryStatus.ACTIVE,
        valid_from=correction.effective_at,
        valid_until=None,
        confidence=correction.successor.support_confidence,
        importance=correction.successor.importance,
        support_count=correction.successor.support_count,
        contradiction_count=0,
        first_supported_at=correction.successor.first_supported_at,
        last_supported_at=correction.successor.last_supported_at,
        derivations=correction.successor.derivations,
        metadata=correction.successor.metadata,
        snapshot=MappingProxyType(successor_snapshot),
    )
    created += 1
    memories[correction.successor.memory_key] = _memory_input_from_successor(
        correction.successor,
        revision_key=successor_key,
        revision_number=number,
        valid_from=correction.effective_at,
    )
    left, right = _sorted_keys(predecessor.revision_key, successor_key)
    relations[(predecessor.tenant_id, left, right, SemanticUpdateRelation.SUPERSEDES)] = (
        SemanticRevisionRelation(
            tenant_id=predecessor.tenant_id,
            left_revision_key=left,
            right_revision_key=right,
            relation=SemanticUpdateRelation.SUPERSEDES,
            effective_at=correction.effective_at,
        )
    )
    return 1, created, updated, superseded


def _pin_retracted_projection(
    withdrawal: ExplicitSemanticWithdrawal,
    *,
    revisions: Mapping[str, SemanticRevisionInput],
    memories: dict[str, SemanticMemoryInput],
    retracted_key: str,
) -> None:
    """Keep a replayed withdrawal as the current projection."""
    retracted = revisions.get(retracted_key)
    current = memories.get(retracted.memory_key) if retracted is not None else None
    if retracted is None or current is None:
        return
    if current.status is SemanticMemoryStatus.RETRACTED and current.revision_key == retracted_key:
        return
    memories[retracted.memory_key] = replace(
        current,
        status=SemanticMemoryStatus.RETRACTED,
        revision_key=retracted_key,
        revision_number=retracted.revision_number,
        valid_from=withdrawal.effective_at,
        valid_until=None,
    )


def _apply_withdrawal(
    withdrawal: ExplicitSemanticWithdrawal,
    *,
    revisions: dict[str, SemanticRevisionInput],
    memories: dict[str, SemanticMemoryInput],
    relations: dict[tuple[str, str, str, SemanticUpdateRelation], SemanticRevisionRelation],
    created: int,
    updated: int,
) -> tuple[int, int, int]:
    target = revisions.get(withdrawal.target_revision_key)
    if target is None or target.tenant_id != withdrawal.tenant_id:
        return 0, created, updated
    retracted_key = generate_revision_key(
        withdrawal.target_revision_key,
        valid_from=withdrawal.effective_at,
        valid_until=None,
    )
    closed_by = target.snapshot.get("closed_by")
    already_applied = (
        isinstance(closed_by, Mapping)
        and closed_by.get("kind") == "withdrawal"
        and (target.valid_until == withdrawal.effective_at or retracted_key in revisions)
    ) or (retracted_key in revisions and target.valid_until == withdrawal.effective_at)
    if already_applied:
        _pin_retracted_projection(
            withdrawal,
            revisions=revisions,
            memories=memories,
            retracted_key=retracted_key,
        )
        return 0, created, updated
    start = target.valid_from or target.first_supported_at
    closed_by = {
        "kind": "withdrawal",
        "retracted_revision_key": retracted_key,
        "effective_at": withdrawal.effective_at.isoformat(),
        "knowledge_at": withdrawal.knowledge_at.isoformat(),
    }
    target_memory = memories.get(target.memory_key)
    revisions[target.revision_key] = replace(
        target,
        valid_from=start,
        valid_until=withdrawal.effective_at,
        snapshot=MappingProxyType(
            _closed_snapshot(
                target,
                target_memory,
                valid_from=start,
                valid_until=withdrawal.effective_at,
                closed_by=closed_by,
            )
        ),
    )
    updated += 1
    number = _next_revision_number(revisions, target.memory_key)
    retracted_snapshot = _retracted_snapshot(
        target,
        target_memory,
        retracted_key=retracted_key,
        effective_at=withdrawal.effective_at,
        knowledge_at=withdrawal.knowledge_at,
        target_revision_key=withdrawal.target_revision_key,
        observation_id=withdrawal.observation_id,
        observation_revision=withdrawal.observation_revision,
        episode_id=withdrawal.episode_id,
    )
    revisions[retracted_key] = SemanticRevisionInput(
        tenant_id=target.tenant_id,
        memory_key=target.memory_key,
        revision_key=retracted_key,
        revision_number=number,
        status=SemanticMemoryStatus.RETRACTED,
        valid_from=withdrawal.effective_at,
        valid_until=None,
        confidence=target.confidence,
        importance=target.importance,
        support_count=target.support_count,
        contradiction_count=target.contradiction_count,
        first_supported_at=target.first_supported_at,
        last_supported_at=withdrawal.effective_at,
        derivations=target.derivations,
        metadata=target.metadata,
        snapshot=MappingProxyType(retracted_snapshot),
    )
    created += 1
    if target_memory is not None:
        memories[target.memory_key] = replace(
            target_memory,
            status=SemanticMemoryStatus.RETRACTED,
            revision_key=retracted_key,
            revision_number=number,
            valid_from=withdrawal.effective_at,
            valid_until=None,
        )
    left, right = _sorted_keys(target.revision_key, retracted_key)
    relations[(target.tenant_id, left, right, SemanticUpdateRelation.RETRACTS)] = (
        SemanticRevisionRelation(
            tenant_id=target.tenant_id,
            left_revision_key=left,
            right_revision_key=right,
            relation=SemanticUpdateRelation.RETRACTS,
            effective_at=withdrawal.effective_at,
        )
    )
    return 1, created, updated


def _closed_snapshot(
    revision: SemanticRevisionInput,
    memory: SemanticMemoryInput | None,
    *,
    valid_from: datetime,
    valid_until: datetime,
    closed_by: Mapping[str, Any],
) -> dict[str, Any]:
    if snapshot_is_present(revision.snapshot) and memory is None:
        updated = dict(revision.snapshot)
        updated["status"] = SemanticMemoryStatus.ACTIVE.value
        updated["valid_from"] = valid_from.isoformat()
        updated["valid_until"] = valid_until.isoformat()
        updated["closed_by"] = dict(closed_by)
        return updated
    source = memory
    return _snapshot(
        statement=source.statement if source is not None else revision.revision_key,
        slot_key=source.slot_key if source is not None else "",
        subject_entity_id=source.subject_entity_id if source is not None else None,
        predicate=source.predicate if source is not None else "",
        object_value=source.object_value if source is not None else "",
        object_entity_id=source.object_entity_id if source is not None else None,
        polarity=source.polarity if source is not None else SemanticPolarity.AFFIRM,
        cardinality=source.cardinality if source is not None else SemanticCardinality.ONE,
        qualifiers=source.qualifiers if source is not None else {},
        status=SemanticMemoryStatus.ACTIVE,
        valid_from=valid_from,
        valid_until=valid_until,
        confidence=revision.confidence,
        importance=revision.importance,
        support_count=revision.support_count,
        contradiction_count=revision.contradiction_count,
        observation_evidence=source.observation_evidence if source is not None else (),
        entities=source.entities if source is not None else (),
        metadata=source.metadata if source is not None else revision.metadata,
        derivations=revision.derivations,
        closed_by=closed_by,
    )


def _retracted_snapshot(
    target: SemanticRevisionInput,
    memory: SemanticMemoryInput | None,
    *,
    retracted_key: str,
    effective_at: datetime,
    knowledge_at: datetime,
    target_revision_key: str,
    observation_id: str,
    observation_revision: int,
    episode_id: str,
) -> dict[str, Any]:
    statement = memory.statement if memory is not None else target.revision_key
    return _snapshot(
        statement=statement,
        slot_key=memory.slot_key if memory is not None else "",
        subject_entity_id=memory.subject_entity_id if memory is not None else None,
        predicate=memory.predicate if memory is not None else "",
        object_value=memory.object_value if memory is not None else "",
        object_entity_id=memory.object_entity_id if memory is not None else None,
        polarity=memory.polarity if memory is not None else SemanticPolarity.AFFIRM,
        cardinality=memory.cardinality if memory is not None else SemanticCardinality.ONE,
        qualifiers=memory.qualifiers if memory is not None else {},
        status=SemanticMemoryStatus.RETRACTED,
        valid_from=effective_at,
        valid_until=None,
        confidence=target.confidence,
        importance=target.importance,
        support_count=target.support_count,
        contradiction_count=target.contradiction_count,
        observation_evidence=memory.observation_evidence if memory is not None else (),
        entities=memory.entities if memory is not None else (),
        metadata=memory.metadata if memory is not None else target.metadata,
        derivations=target.derivations,
        knowledge_at=knowledge_at,
        withdrawal={
            "target_revision_key": target_revision_key,
            "effective_at": effective_at.isoformat(),
            "retracted_revision_key": retracted_key,
            "observation_id": observation_id,
            "observation_revision": observation_revision,
            "episode_id": episode_id,
        },
    )


def _memory_input_from_successor(
    candidate: SemanticRevisionCandidate,
    *,
    revision_key: str,
    revision_number: int,
    valid_from: datetime,
) -> SemanticMemoryInput:
    return SemanticMemoryInput(
        tenant_id=candidate.tenant_id,
        subject_id=candidate.subject_id,
        memory_key=candidate.memory_key,
        slot_key=candidate.slot_key,
        revision_key=revision_key,
        revision_number=revision_number,
        statement=candidate.statement,
        subject_entity_id=candidate.subject_entity_id,
        predicate=candidate.predicate,
        object_value=candidate.object_value,
        object_entity_id=candidate.object_entity_id,
        polarity=candidate.polarity,
        cardinality=candidate.cardinality,
        qualifiers=candidate.qualifiers,
        confidence=candidate.support_confidence,
        importance=candidate.importance,
        status=SemanticMemoryStatus.ACTIVE,
        support_count=candidate.support_count,
        contradiction_count=0,
        first_supported_at=candidate.first_supported_at,
        last_supported_at=candidate.last_supported_at,
        derivations=candidate.derivations,
        observation_evidence=candidate.observation_evidence,
        valid_from=valid_from,
        valid_until=None,
        entities=candidate.entities,
        metadata=candidate.metadata,
    )


def _snapshot(
    *,
    statement: str,
    slot_key: str,
    subject_entity_id: str | None,
    predicate: str,
    object_value: str,
    object_entity_id: str | None,
    polarity: SemanticPolarity,
    cardinality: SemanticCardinality,
    qualifiers: Mapping[str, Any],
    status: SemanticMemoryStatus,
    valid_from: datetime | None,
    valid_until: datetime | None,
    confidence: float,
    importance: float,
    support_count: int,
    contradiction_count: int,
    observation_evidence: Sequence[EpisodeEvidenceInput],
    entities: Sequence[EpisodeEntity],
    metadata: Mapping[str, Any],
    derivations: Sequence[SemanticDerivationInput],
    knowledge_at: datetime | None = None,
    correction: Mapping[str, Any] | None = None,
    withdrawal: Mapping[str, Any] | None = None,
    closed_by: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": SNAPSHOT_VERSION,
        "statement": statement,
        "slot_key": slot_key,
        "subject_entity_id": subject_entity_id,
        "predicate": predicate,
        "object_value": object_value,
        "object_entity_id": object_entity_id,
        "polarity": polarity.value,
        "cardinality": cardinality.value,
        "qualifiers": _jsonable(qualifiers),
        "status": status.value,
        "valid_from": valid_from.isoformat() if valid_from is not None else None,
        "valid_until": valid_until.isoformat() if valid_until is not None else None,
        "confidence": confidence,
        "importance": importance,
        "support_count": support_count,
        "contradiction_count": contradiction_count,
        "observation_evidence": [
            {
                "observation_id": item.observation_id,
                "observation_revision": item.observation_revision,
                "sequence_number": item.sequence_number,
                "contribution_score": item.contribution_score,
            }
            for item in observation_evidence
        ],
        "entities": [{"entity_id": entity.entity_id, "role": entity.role} for entity in entities],
        "metadata": _jsonable(metadata),
        "derivations": [
            {
                "episode_id": item.episode_id,
                "relation": item.relation.value,
                "contribution_score": item.contribution_score,
            }
            for item in derivations
        ],
    }
    if knowledge_at is not None:
        payload["knowledge_at"] = knowledge_at.isoformat()
    if correction is not None:
        payload["correction"] = _jsonable(correction)
    if withdrawal is not None:
        payload["withdrawal"] = _jsonable(withdrawal)
    if closed_by is not None:
        payload["closed_by"] = _jsonable(closed_by)
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    value_attr = getattr(value, "value", None)
    if isinstance(value_attr, str):
        return value_attr
    return str(value)


def _parse_correction(raw_fact: Mapping[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    if "correction" not in raw_fact:
        return None, False
    raw = raw_fact.get("correction")
    if not isinstance(raw, Mapping) or set(raw) != {"replaces_revision_key", "effective_at"}:
        return None, True
    key = raw.get("replaces_revision_key")
    if not isinstance(key, str) or not key.strip():
        return None, True
    try:
        effective = _parse_optional_datetime(raw.get("effective_at"))
    except ValidationError:
        return None, True
    if effective is None:
        return None, True
    return {"replaces_revision_key": key.strip(), "effective_at": effective}, False


def _parse_withdrawal(raw_item: object) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(raw_item, Mapping) or set(raw_item) != {
        "target_revision_key",
        "effective_at",
        "authority",
    }:
        return None, True
    key = raw_item.get("target_revision_key")
    authority = raw_item.get("authority")
    if not isinstance(key, str) or not key.strip():
        return None, True
    if not isinstance(authority, Mapping) or set(authority) != {"basis"}:
        return None, True
    if authority.get("basis") != "application_asserted":
        return None, True
    try:
        effective = _parse_optional_datetime(raw_item.get("effective_at"))
    except ValidationError:
        return None, True
    if effective is None:
        return None, True
    return {"target_revision_key": key.strip(), "effective_at": effective}, False


def _fact_has_authority(raw_fact: Mapping[str, Any]) -> bool:
    authority = raw_fact.get("authority")
    return (
        isinstance(authority, Mapping)
        and set(authority) == {"basis"}
        and authority.get("basis") == "application_asserted"
    )


def _matching_candidate(
    candidates: Sequence[SemanticRevisionCandidate],
    observation_id: str,
    raw_fact: Mapping[str, Any],
) -> SemanticRevisionCandidate | None:
    predicate = str(raw_fact.get("predicate", "")).strip().casefold()
    object_value = str(raw_fact.get("object_value", "")).strip().casefold()
    for candidate in candidates:
        if candidate.predicate.casefold() != predicate:
            continue
        if candidate.object_value.casefold() != object_value:
            continue
        if any(item.observation_id == observation_id for item in candidate.observation_evidence):
            return candidate
    return None


def _raw_facts_from_revision(revision: StoredObservationRevision | None) -> tuple[Any, ...]:
    if revision is None:
        return ()
    return _sequence_of_facts(revision.metadata.get("semantic_facts", ()))


def _raw_facts(
    observation: StoredObservation | None,
    revision: StoredObservationRevision | None,
) -> tuple[Any, ...]:
    if revision is not None:
        return _raw_facts_from_revision(revision)
    if observation is None:
        return ()
    return _sequence_of_facts(observation.metadata.get("semantic_facts", ()))


def _sequence_of_facts(raw_facts: object) -> tuple[Any, ...]:
    if raw_facts is None:
        return ()
    if isinstance(raw_facts, Mapping):
        return (raw_facts,)
    if isinstance(raw_facts, Sequence) and not isinstance(raw_facts, (str, bytes)):
        return tuple(raw_facts)
    return ()


def _slot_key(
    stored: StoredSemanticRevision | None,
    batch: SemanticRevisionCandidate | None,
    memories_by_key: Mapping[str, StoredSemanticMemory],
) -> str | None:
    if batch is not None:
        return batch.slot_key
    if stored is None:
        return None
    memory = memories_by_key.get(stored.memory_key)
    if memory is not None:
        return memory.slot_key
    slot = stored.snapshot.get("slot_key")
    return str(slot) if isinstance(slot, str) and slot else None


def _cardinality(
    stored: StoredSemanticRevision | None,
    batch: SemanticRevisionCandidate | None,
    memories_by_key: Mapping[str, StoredSemanticMemory],
) -> SemanticCardinality | None:
    if batch is not None:
        return batch.cardinality
    if stored is None:
        return None
    memory = memories_by_key.get(stored.memory_key)
    if memory is not None:
        return memory.cardinality
    raw = stored.snapshot.get("cardinality")
    try:
        return SemanticCardinality(str(raw))
    except ValueError:
        return None


def _effective_start(
    stored: StoredSemanticRevision | None,
    batch: SemanticRevisionCandidate | None,
) -> datetime | None:
    if stored is not None:
        return stored.valid_from or stored.first_supported_at
    if batch is not None:
        return batch.valid_from or batch.first_supported_at
    return None


def _next_revision_number(revisions: Mapping[str, SemanticRevisionInput], memory_key: str) -> int:
    numbers = [
        revision.revision_number
        for revision in revisions.values()
        if revision.memory_key == memory_key
    ]
    return max(numbers, default=0) + 1


def _sorted_keys(left: str, right: str) -> tuple[str, str]:
    if left <= right:
        return left, right
    return right, left


def _knowledge_at(snapshot: Mapping[str, Any]) -> datetime | None:
    return _parse_snapshot_datetime(snapshot.get("knowledge_at"))


def _predecessor_key(snapshot: Mapping[str, Any]) -> str | None:
    correction = snapshot.get("correction")
    if isinstance(correction, Mapping):
        key = correction.get("replaces_revision_key")
        if isinstance(key, str) and key:
            return key
    withdrawal = snapshot.get("withdrawal")
    if isinstance(withdrawal, Mapping):
        key = withdrawal.get("target_revision_key")
        if isinstance(key, str) and key:
            return key
    return None


def _parse_snapshot_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if not isinstance(value, str) or not value:
        return None
    try:
        return _parse_optional_datetime(value)
    except ValidationError:
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _evidence_from_snapshot(raw_items: object) -> tuple[EpisodeEvidenceInput, ...]:
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return ()
    evidence: list[EpisodeEvidenceInput] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        try:
            evidence.append(
                EpisodeEvidenceInput(
                    observation_id=str(item["observation_id"]),
                    observation_revision=int(item["observation_revision"]),
                    sequence_number=int(item["sequence_number"]),
                    contribution_score=float(item.get("contribution_score", 1.0)),
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            continue
    return tuple(evidence)


def _entities_from_snapshot(raw_items: object) -> tuple[EpisodeEntity, ...]:
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return ()
    entities: list[EpisodeEntity] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        try:
            entities.append(EpisodeEntity(entity_id=str(item["entity_id"]), role=str(item["role"])))
        except (KeyError, TypeError, ValidationError):
            continue
    return tuple(entities)


def _derivations_from_snapshot(raw_items: object) -> tuple[SemanticDerivationInput, ...]:
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return ()
    derivations: list[SemanticDerivationInput] = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        try:
            derivations.append(
                SemanticDerivationInput(
                    episode_id=str(item["episode_id"]),
                    relation=SemanticDerivationRelation(str(item["relation"])),
                    contribution_score=float(item["contribution_score"]),
                )
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            continue
    return tuple(derivations)


def _observed_edit_outcome(
    outcome: str,
    reason: SemanticDecisionReason,
    observer: EditDecisionObserver | None,
) -> str:
    if outcome == "invalid" and reason is SemanticDecisionReason.ALREADY_APPLIED:
        reason = SemanticDecisionReason.TARGET_ALREADY_CLOSED
    return observer.decide(outcome, reason) if observer is not None else outcome


def _edit_observer(
    diagnostics: SemanticDecisionCollector | None,
    edit: ExplicitSemanticCorrection | ExplicitSemanticWithdrawal,
) -> EditDecisionObserver:
    if isinstance(edit, ExplicitSemanticCorrection):
        target = edit.predecessor_revision_key
        key = generate_revision_key(
            edit.successor.memory_key, valid_from=edit.effective_at, valid_until=None
        )
        return EditDecisionObserver(
            diagnostics,
            operation=SemanticDecisionOperation.REPLACE,
            observation_id=edit.observation_id,
            observation_revision=edit.observation_revision,
            target_key=target,
            effective_at=edit.effective_at,
            slot_key=edit.successor.slot_key,
            claim_key=edit.successor.memory_key,
            resulting_revision_key=key,
        )
    return EditDecisionObserver(
        diagnostics,
        operation=SemanticDecisionOperation.WITHDRAW,
        observation_id=edit.observation_id,
        observation_revision=edit.observation_revision,
        target_key=edit.target_revision_key,
        effective_at=edit.effective_at,
        resulting_revision_key=generate_revision_key(
            edit.target_revision_key, valid_from=edit.effective_at, valid_until=None
        ),
    )
