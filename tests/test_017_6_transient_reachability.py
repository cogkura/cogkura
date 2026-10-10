"""Reachability tests for 0.17.6 transient interference."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.competition import _noisy_or_pressure
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    ActivationConfig,
    BehavioralStructuralAnchor,
    CompetitionConfig,
    CompetitionDirection,
    InhibitionConfig,
    InhibitionScopeSignature,
    MemoryAssessmentFlag,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionResult,
    RecallResult,
    RetrievalInterferenceObservability,
    RetrievalInterferenceState,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_dynamics import InMemoryMemoryDynamicsStore
from cogkura.storage.in_memory_entity_relationship import InMemoryEntityRelationshipStore
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore
from cogkura.storage.in_memory_inhibition import InMemoryInhibitionStore
from cogkura.storage.in_memory_learning import InMemoryLearningStore
from cogkura.storage.in_memory_observation import InMemoryCheckpointStore, InMemoryObservationStore
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore

_TENANT = "reach-tenant"
_SUBJECT = "operator-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_QUERY = "How is payments-api deployed?"


class TransientReachabilityFailure(StrEnum):
    """First stage at which a controlled transient effect stops."""

    NO_DIAGNOSTIC_COMPETITION = "no_diagnostic_competition"
    SCOPE_INELIGIBLE = "scope_ineligible"
    TEMPORALLY_INELIGIBLE = "temporally_ineligible"
    ZERO_COMPETITOR_ACCESSIBILITY = "zero_competitor_accessibility"
    ZERO_PAIR_PRESSURE = "zero_pair_pressure"
    ZERO_AGGREGATE_PRESSURE = "zero_aggregate_pressure"
    ZERO_WEIGHT = "zero_weight"
    NO_NEGATIVE_PENALTY = "no_negative_penalty"


def _memory(**kwargs: object) -> Memory:
    defaults: dict[str, object] = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
        "competition_config": CompetitionConfig(enabled=True, apply_interference=True),
        "inhibition_config": InhibitionConfig(enabled=False, apply_to_recall=False),
    }
    defaults.update(kwargs)
    return Memory(**defaults)  # type: ignore[arg-type]


def _fact(
    *,
    source_record_id: str,
    predicate: str,
    object_value: str,
    subject_entity_id: str,
    observed_at: datetime,
    content: str | None = None,
    cardinality: str = "many",
) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        source_namespace="facts",
        source_record_id=source_record_id,
        source_type="fact",
        content=content or f"{subject_entity_id} {predicate} {object_value}",
        observed_at=observed_at,
        metadata={
            "entity_ids": [subject_entity_id],
            "semantic_facts": [
                {
                    "predicate": predicate,
                    "object_value": object_value,
                    "subject_entity_id": subject_entity_id,
                    "cardinality": cardinality,
                    "polarity": "affirm",
                    "qualifiers": {},
                }
            ],
        },
    )


async def _seed_deployment_episodes(memory: Memory, *, same_time: bool = False) -> None:
    jenkins_at = _T - timedelta(days=10 if same_time else 120)
    await memory.observe(
        _fact(
            source_record_id="jenkins",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
            observed_at=jenkins_at,
            content="In 2025 payments-api deployments used Jenkins.",
        )
    )
    await memory.observe(
        _fact(
            source_record_id="gha",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
            content="In 2026 payments-api deployments use GitHub Actions.",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)


def _episodes(results: list[RecallResult], token: str) -> list[RecallResult]:
    return [
        result
        for result in results
        if result.memory_kind is MemoryKind.EPISODE and token in result.memory.statement
    ]


def _inspect_episodes(
    inspection: RecallInspectionResult,
    token: str,
) -> list[RecallInspectionCandidate]:
    return [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.memory_kind is MemoryKind.EPISODE and token in candidate.memory.statement
    ]


def _first_failure(
    candidate: RecallInspectionCandidate,
    *,
    competitor_token: str,
    direction: CompetitionDirection,
) -> TransientReachabilityFailure | None:
    competition = candidate.competition
    if competition is None or not competition.competitors:
        return TransientReachabilityFailure.NO_DIAGNOSTIC_COMPETITION
    matching = [
        evidence
        for evidence in competition.competitors
        if competitor_token in evidence.competitor_identity.memory_key
        or competitor_token in str(evidence.competitor_identity)
    ]
    # Competitor identity is a memory key, not the statement. Match any evidence
    # whose direction and later filters identify the pair; callers also check keys.
    if not matching:
        matching = list(competition.competitors)
        if not matching:
            return TransientReachabilityFailure.NO_DIAGNOSTIC_COMPETITION
    scope = [
        evidence
        for evidence in matching
        if evidence.behavioral_eligibility is not None
        and evidence.behavioral_eligibility.scope_eligible
    ]
    if not scope:
        return TransientReachabilityFailure.SCOPE_INELIGIBLE
    temporal = [
        evidence
        for evidence in scope
        if evidence.direction is direction
        and evidence.behavioral_eligibility is not None
        and evidence.behavioral_eligibility.eligible
    ]
    if not temporal:
        return TransientReachabilityFailure.TEMPORALLY_INELIGIBLE
    interference = competition.interference
    if interference is None:
        return TransientReachabilityFailure.ZERO_AGGREGATE_PRESSURE
    contributions = [item for item in interference.contributions if item.direction is direction]
    if not contributions:
        return TransientReachabilityFailure.TEMPORALLY_INELIGIBLE
    if all(item.competitor_accessibility <= 0.0 for item in contributions):
        return TransientReachabilityFailure.ZERO_COMPETITOR_ACCESSIBILITY
    if all(item.pressure <= 0.0 for item in contributions):
        return TransientReachabilityFailure.ZERO_PAIR_PRESSURE
    aggregate = (
        interference.proactive_pressure
        if direction is CompetitionDirection.PROACTIVE
        else interference.retroactive_pressure
    )
    if aggregate <= 0.0:
        return TransientReachabilityFailure.ZERO_AGGREGATE_PRESSURE
    penalty = (
        interference.proactive_penalty
        if direction is CompetitionDirection.PROACTIVE
        else interference.retroactive_penalty
    )
    if penalty == 0.0:
        return TransientReachabilityFailure.ZERO_WEIGHT
    if interference.total_penalty >= 0.0:
        return TransientReachabilityFailure.NO_NEGATIVE_PENALTY
    return None


def _require_direction(
    inspection: RecallInspectionResult,
    *,
    candidate_token: str,
    direction: CompetitionDirection,
) -> RecallInspectionCandidate:
    episodes = _inspect_episodes(inspection, candidate_token)
    assert episodes, f"episode containing {candidate_token!r} did not enter inspection"
    candidate = episodes[0]
    failure = _first_failure(candidate, competitor_token="", direction=direction)
    detail = _episode_detail(candidate)
    summary = inspection.interference
    assert failure is None, (
        f"{failure.value if failure is not None else 'ok'} "
        f"summary={_summary(summary)} detail={detail}"
    )
    return candidate


def _episode_detail(candidate: RecallInspectionCandidate) -> str:
    diagnostics = candidate.diagnostics
    provenance = ()
    slot = None
    cue_fit = None
    if diagnostics is not None:
        provenance = tuple(
            (item.semantic_status.value, item.semantic_slot_key)
            for item in diagnostics.support_provenance
        )
        slot = diagnostics.semantic_slot_key
        cue_fit = (diagnostics.semantic_relevance, diagnostics.text_cue_fit, diagnostics.slot_fit)
    competition = candidate.competition
    reasons = []
    if competition is not None:
        for evidence in competition.competitors:
            eligibility = evidence.behavioral_eligibility
            reasons.append(
                (
                    evidence.direction.value,
                    evidence.strength,
                    evidence.candidate_cue_fit,
                    evidence.competitor_cue_fit,
                    None if eligibility is None else eligibility.scope_eligible,
                    None if eligibility is None else eligibility.reason.value,
                    evidence.same_semantic_slot,
                    evidence.same_predicate,
                )
            )
    return f"slot={slot} provenance={provenance} cue={cue_fit} reasons={reasons}"


def _summary(interference: RetrievalInterferenceObservability | None) -> str:
    if interference is None:
        return "none"
    return (
        f"state={interference.state.value} "
        f"diagnostic={interference.diagnostic_relationship_count} "
        f"scope={interference.scope_eligible_relationship_count} "
        f"transient={interference.transient_eligible_relationship_count} "
        f"pressure={interference.max_transient_pressure} "
        f"penalty={interference.max_transient_penalty_magnitude}"
    )


@pytest.mark.asyncio
async def test_proactive_episode_interference_is_reachable() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    candidate = _require_direction(
        inspection,
        candidate_token="GitHub Actions",
        direction=CompetitionDirection.PROACTIVE,
    )
    assert candidate.competition is not None
    assert candidate.competition.interference is not None
    assert candidate.competition.interference.proactive_penalty < 0.0
    recalled_episode = _episodes(recalled, "GitHub Actions")
    assert recalled_episode
    assert recalled_episode[0].components.interference < 0.0
    assert recalled_episode[0].components.interference == pytest.approx(
        candidate.components.interference
    )


@pytest.mark.asyncio
async def test_retroactive_episode_interference_is_reachable() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    candidate = _require_direction(
        inspection,
        candidate_token="Jenkins",
        direction=CompetitionDirection.RETROACTIVE,
    )
    assert candidate.competition is not None
    assert candidate.competition.interference is not None
    assert candidate.competition.interference.retroactive_penalty < 0.0
    recalled_episode = _episodes(recalled, "Jenkins")
    assert recalled_episode
    assert recalled_episode[0].components.interference < 0.0


def _signature(results: list[RecallResult]) -> list[tuple[str, str, float]]:
    return [
        (result.memory_kind.value, result.memory.statement, result.activation) for result in results
    ]


def _semantic_contribution_pressure(
    inspection: RecallInspectionResult,
    *,
    candidate_token: str,
    competitor_token: str,
) -> float:
    candidates = [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.memory_kind is MemoryKind.SEMANTIC
        and candidate_token in candidate.memory.statement
    ]
    assert candidates
    competition = candidates[0].competition
    assert competition is not None and competition.interference is not None
    competitor_keys = {
        item.memory.memory_key
        for item in (*inspection.returned, *inspection.rejected)
        if item.memory_kind is MemoryKind.SEMANTIC and competitor_token in item.memory.statement
    }
    pressures = [
        item.pressure
        for item in competition.interference.contributions
        if item.competitor_identity.memory_key in competitor_keys
    ]
    assert pressures
    return max(pressures)


def _stores() -> dict[str, object]:
    return {
        "observation_store": InMemoryObservationStore(),
        "checkpoint_store": InMemoryCheckpointStore(),
        "episode_store": InMemoryEpisodeStore(),
        "semantic_store": InMemorySemanticMemoryStore(),
        "activation_store": InMemoryActivationStore(),
        "dynamics_store": InMemoryMemoryDynamicsStore(),
        "learning_store": InMemoryLearningStore(),
        "entity_relationship_store": InMemoryEntityRelationshipStore(),
        "inhibition_store": InMemoryInhibitionStore(),
    }


@pytest.mark.asyncio
async def test_semantic_deployment_pair_is_also_reachable() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    by_statement = {
        candidate.memory.statement: candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.memory_kind is MemoryKind.SEMANTIC
    }
    newer = next(
        candidate for statement, candidate in by_statement.items() if "github_actions" in statement
    )
    older = next(
        candidate for statement, candidate in by_statement.items() if "jenkins" in statement
    )
    assert newer.competition and newer.competition.interference
    assert older.competition and older.competition.interference
    assert newer.competition.interference.proactive_penalty < 0.0
    assert older.competition.interference.retroactive_penalty < 0.0


@pytest.mark.asyncio
async def test_more_accessible_competitor_creates_more_pressure() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    await memory.observe(
        _fact(
            source_record_id="circleci",
            predicate="deployment_system",
            object_value="circleci",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=120),
            content="In 2025 payments-api deployments also used CircleCI.",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    before = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    circle = [
        result
        for result in recalled
        if result.memory_kind is MemoryKind.SEMANTIC and "circleci" in result.memory.statement
    ]
    assert circle
    for index in range(3):
        await memory.record_access(
            circle,
            tenant_id=_TENANT,
            referenced_at=_T,
            request_id=f"boost-circle-{index}",
        )
    after = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)

    def _pair_pressures(inspection: RecallInspectionResult) -> tuple[float, float]:
        episode = _inspect_episodes(inspection, "GitHub Actions")[0]
        assert episode.competition is not None and episode.competition.interference is not None
        keys = {
            candidate.memory.memory_key
            for candidate in (*inspection.returned, *inspection.rejected)
        }
        jenkins_keys = {
            candidate.memory.memory_key
            for candidate in (*inspection.returned, *inspection.rejected)
            if "Jenkins" in candidate.memory.statement
        }
        circle_keys = {
            candidate.memory.memory_key
            for candidate in (*inspection.returned, *inspection.rejected)
            if "circleci" in candidate.memory.statement.lower()
        }
        assert jenkins_keys and circle_keys
        assert jenkins_keys <= keys
        jenkins_pressure = 0.0
        circle_pressure = 0.0
        for item in episode.competition.interference.contributions:
            if item.competitor_identity.memory_key in jenkins_keys:
                jenkins_pressure = max(jenkins_pressure, item.pressure)
            if item.competitor_identity.memory_key in circle_keys:
                circle_pressure = max(circle_pressure, item.pressure)
        return jenkins_pressure, circle_pressure

    _before_jenkins, before_circle = _pair_pressures(before)
    after_jenkins, after_circle = _pair_pressures(after)
    assert after_circle > before_circle
    assert after_circle > after_jenkins


@pytest.mark.asyncio
async def test_two_competitors_aggregate_with_bounded_noisy_or() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    await memory.observe(
        _fact(
            source_record_id="circleci",
            predicate="deployment_system",
            object_value="circleci",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=90),
            content="In 2025 payments-api deployments also used CircleCI.",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    episode = _inspect_episodes(inspection, "GitHub Actions")[0]
    assert episode.competition is not None and episode.competition.interference is not None
    interference = episode.competition.interference
    pressures = [
        item.pressure
        for item in interference.contributions
        if item.direction is CompetitionDirection.PROACTIVE
    ]
    assert len(pressures) >= 2
    assert interference.proactive_pressure > max(pressures)
    assert interference.proactive_pressure <= 1.0
    assert interference.proactive_pressure == pytest.approx(_noisy_or_pressure(pressures))


def test_noisy_or_is_order_independent_and_bounded() -> None:
    forward = _noisy_or_pressure((0.2, 0.4))
    reverse = _noisy_or_pressure((0.4, 0.2))
    assert forward == pytest.approx(reverse)
    assert forward > 0.4
    assert forward <= 1.0
    assert _noisy_or_pressure((0.0,)) == 0.0


@pytest.mark.asyncio
async def test_transient_penalty_can_move_rank() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=True,
            proactive_weight=1.0,
            retroactive_weight=1.0,
        ),
    )
    await _seed_deployment_episodes(memory)
    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="notes",
            source_record_id="runbook",
            source_type="note",
            content="How payments-api is deployed is recorded in the deployment runbook.",
            observed_at=_T - timedelta(days=400),
            metadata={"entity_ids": ["payments-api"]},
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    moved = [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.rank_before_interference is not None
        and candidate.rank_after_interference is not None
        and candidate.rank_after_interference > candidate.rank_before_interference
    ]
    assert moved


@pytest.mark.asyncio
async def test_controlled_threshold_crossing() -> None:
    stores = _stores()
    probe = _memory(
        **stores,
        competition_config=CompetitionConfig(enabled=True, apply_interference=False),
    )
    await _seed_deployment_episodes(probe)
    recalled = await probe.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    target = _episodes(recalled, "GitHub Actions")[0]
    memory = _memory(
        **stores,
        activation_config=ActivationConfig(
            retrieval_threshold=target.activation - 0.05,
            enable_semantic_slot_admission=False,
            enable_entity_slot_admission=False,
        ),
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=True,
            proactive_weight=1.0,
            retroactive_weight=1.0,
        ),
    )
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    episode = _inspect_episodes(inspection, "GitHub Actions")[0]
    diagnostics = episode.diagnostics
    threshold = memory._activation_config.retrieval_threshold
    assert diagnostics is not None
    assert diagnostics.activation_before_interference is not None
    assert diagnostics.activation_before_interference >= threshold
    assert episode.activation < threshold
    assert diagnostics.crossed_activation_threshold_due_to_interference


@pytest.mark.asyncio
async def test_negative_controls_have_zero_transient_penalty() -> None:
    memory = _memory()
    observations = [
        _fact(
            source_record_id="deploy",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
            content="payments-api deployments use GitHub Actions.",
        ),
        _fact(
            source_record_id="db",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=20),
            content="payments-api production database is PostgreSQL.",
        ),
        _fact(
            source_record_id="identity",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="identity-service",
            observed_at=_T - timedelta(days=30),
            content="identity-service deployments use Jenkins.",
        ),
        _fact(
            source_record_id="backup",
            predicate="backup_schedule",
            object_value="six_hours",
            subject_entity_id="postgresql",
            observed_at=_T - timedelta(days=5),
            content="PostgreSQL backups run every six hours.",
        ),
        _fact(
            source_record_id="broad-payments",
            predicate="oncall_team",
            object_value="payments",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=15),
            content="payments-api oncall team is payments.",
        ),
        _fact(
            source_record_id="broad-identity",
            predicate="oncall_team",
            object_value="identity",
            subject_entity_id="identity-service",
            observed_at=_T - timedelta(days=16),
            content="identity-service oncall team is identity.",
        ),
    ]
    for observation in observations:
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, as_of=_T)
    queries = (
        _QUERY,
        "What is the production database?",
        "How is identity-service deployed?",
    )
    for query in queries:
        recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
        assert all(result.components.interference == 0.0 for result in recalled)
        assessment = await memory.assess_memory(query, tenant_id=_TENANT, as_of=_T)
        assert MemoryAssessmentFlag.HIGH_INTERFERENCE not in assessment.flags


@pytest.mark.asyncio
async def test_co_temporal_competition_has_no_transient_penalty() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory, same_time=True)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    episode = _inspect_episodes(inspection, "GitHub Actions")[0]
    assert episode.competition is not None
    scope = [
        evidence
        for evidence in episode.competition.competitors
        if evidence.behavioral_eligibility is not None
        and evidence.behavioral_eligibility.scope_eligible
        and evidence.direction is CompetitionDirection.CO_TEMPORAL
    ]
    assert scope
    assert all(
        evidence.behavioral_eligibility is not None and not evidence.behavioral_eligibility.eligible
        for evidence in scope
    )
    interference = episode.competition.interference
    assert interference is None or interference.total_penalty == 0.0
    recalled = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert all(result.components.interference == 0.0 for result in recalled)


@pytest.mark.asyncio
async def test_same_lineage_does_not_interfere() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    for candidate in (*inspection.returned, *inspection.rejected):
        if candidate.competition is None or candidate.competition.interference is None:
            continue
        diagnostics = candidate.diagnostics
        supported: set[str] = set()
        if diagnostics is not None:
            supported = {item.semantic_memory_key for item in diagnostics.support_provenance}
        for contribution in candidate.competition.interference.contributions:
            assert contribution.competitor_identity.memory_key not in supported
        assert candidate.competition.interference.total_penalty <= 0.0


@pytest.mark.asyncio
async def test_superseded_semantic_is_not_a_live_transient_competitor() -> None:
    memory = _memory()
    await memory.observe(
        _fact(
            source_record_id="mysql",
            predicate="production_database",
            object_value="mysql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=200),
            content="payments-api production database was MySQL.",
            cardinality="one",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T - timedelta(days=100))
    await memory.observe(
        _fact(
            source_record_id="postgres",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
            content="payments-api production database is PostgreSQL.",
            cardinality="one",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    recalled = await memory.recall(
        "What is the production database?",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    assert recalled
    assert all("MySQL" not in result.memory.statement for result in recalled)
    assert all(result.components.interference == 0.0 for result in recalled)


@pytest.mark.asyncio
async def test_historical_valid_at_excludes_future_competitor() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    recalled = await memory.recall(
        _QUERY,
        tenant_id=_TENANT,
        as_of=_T,
        valid_at=_T - timedelta(days=30),
        limit=10,
    )
    assert all("GitHub Actions" not in result.memory.statement for result in recalled)
    assert all(result.components.interference == 0.0 for result in recalled)


@pytest.mark.asyncio
async def test_unrelated_high_accessibility_cannot_interfere() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    await memory.observe(
        _fact(
            source_record_id="db",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=3),
            content="payments-api production database is PostgreSQL.",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    database = await memory.recall(
        "What is the production database?",
        tenant_id=_TENANT,
        as_of=_T,
        limit=10,
    )
    assert database
    for index in range(4):
        await memory.record_access(
            database[:1],
            tenant_id=_TENANT,
            referenced_at=_T,
            request_id=f"boost-db-{index}",
        )
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    episode = _inspect_episodes(inspection, "GitHub Actions")[0]
    if episode.competition is not None and episode.competition.interference is not None:
        for contribution in episode.competition.interference.contributions:
            key = contribution.competitor_identity.memory_key
            assert "postgresql" not in key


@pytest.mark.asyncio
async def test_persistent_inhibition_lowers_transient_pressure() -> None:
    from cogkura.models import InhibitoryTrace, MemoryIdentity

    store = InMemoryInhibitionStore()
    memory = _memory(
        inhibition_store=store,
        inhibition_config=InhibitionConfig(enabled=False, apply_to_recall=True),
    )
    await _seed_deployment_episodes(memory)
    query = "payments-api deployment_system"
    before = await memory.inspect_recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    recalled = await memory.recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    jenkins = next(
        result
        for result in recalled
        if result.memory_kind is MemoryKind.SEMANTIC and "jenkins" in result.memory.statement
    )
    before_pressure = _semantic_contribution_pressure(
        before,
        candidate_token="github_actions",
        competitor_token="jenkins",
    )
    await store.append_traces(
        [
            InhibitoryTrace(
                id="00000000-0000-0000-0000-000000000176",
                tenant_id=_TENANT,
                inhibited_identity=MemoryIdentity(
                    memory_kind=MemoryKind.SEMANTIC,
                    memory_key=jenkins.memory.memory_key,
                ),
                selected_identity=MemoryIdentity(
                    memory_kind=MemoryKind.SEMANTIC,
                    memory_key="winner",
                ),
                direction=CompetitionDirection.PROACTIVE,
                scope=InhibitionScopeSignature(
                    structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
                    subject_id="payments-api",
                    predicate="deployment_system",
                    semantic_slot_key=jenkins.memory.slot_key,
                ),
                competition_strength=0.8,
                competitor_accessibility=1.0,
                induction_pressure=0.8,
                retrieval_evaluated_at=_T,
                induced_at=_T,
            )
        ]
    )
    after = await memory.inspect_recall(query, tenant_id=_TENANT, as_of=_T, limit=10)
    after_pressure = _semantic_contribution_pressure(
        after,
        candidate_token="github_actions",
        competitor_token="jenkins",
    )
    assert after_pressure < before_pressure
    assert after.interference is not None
    assert after.interference.state is RetrievalInterferenceState.COMBINED


@pytest.mark.asyncio
async def test_metamemory_reports_transient_interference() -> None:
    from cogkura import MetamemoryConfig

    memory = _memory(metamemory_config=MetamemoryConfig(high_interference_pressure_threshold=0.01))
    await _seed_deployment_episodes(memory)
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT, as_of=_T)
    assert assessment.interference is not None
    assert assessment.interference.state is RetrievalInterferenceState.TRANSIENT_INTERFERENCE
    assert assessment.interference.max_transient_penalty_magnitude is not None
    assert assessment.interference.max_transient_penalty_magnitude > 0.0
    assert MemoryAssessmentFlag.HIGH_INTERFERENCE in assessment.flags
    assert MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY not in assessment.flags


@pytest.mark.asyncio
async def test_interference_off_matches_on_for_negative_control() -> None:
    off = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=False))
    on = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    for memory in (off, on):
        await memory.observe(
            _fact(
                source_record_id="deploy",
                predicate="deployment_system",
                object_value="github_actions",
                subject_entity_id="payments-api",
                observed_at=_T - timedelta(days=10),
                content="payments-api deployments use GitHub Actions.",
            )
        )
        await memory.observe(
            _fact(
                source_record_id="db",
                predicate="production_database",
                object_value="postgresql",
                subject_entity_id="payments-api",
                observed_at=_T - timedelta(days=20),
                content="payments-api production database is PostgreSQL.",
            )
        )
        await memory.process(tenant_id=_TENANT, as_of=_T)
    off_results = await off.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    on_results = await on.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    assert _signature(off_results) == _signature(on_results)


@pytest.mark.asyncio
async def test_repeated_and_reordered_retrieval_is_deterministic() -> None:
    async def _episode_penalties(reverse: bool) -> list[tuple[str, float]]:
        memory = _memory()
        facts = [
            _fact(
                source_record_id="jenkins",
                predicate="deployment_system",
                object_value="jenkins",
                subject_entity_id="payments-api",
                observed_at=_T - timedelta(days=120),
                content="In 2025 payments-api deployments used Jenkins.",
            ),
            _fact(
                source_record_id="gha",
                predicate="deployment_system",
                object_value="github_actions",
                subject_entity_id="payments-api",
                observed_at=_T - timedelta(days=10),
                content="In 2026 payments-api deployments use GitHub Actions.",
            ),
        ]
        if reverse:
            facts.reverse()
        for fact in facts:
            await memory.observe(fact)
        await memory.process(tenant_id=_TENANT, as_of=_T)
        first = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
        second = await memory.recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
        assert _signature(first) == _signature(second)
        return [
            (result.memory.statement, result.components.interference)
            for result in first
            if result.memory_kind is MemoryKind.EPISODE
        ]

    forward = await _episode_penalties(False)
    backward = await _episode_penalties(True)
    assert forward == backward
    assert any(penalty < 0.0 for _statement, penalty in forward)


@pytest.mark.asyncio
async def test_penalties_are_negative_only() -> None:
    memory = _memory()
    await _seed_deployment_episodes(memory)
    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, as_of=_T, limit=10)
    affected = 0
    for candidate in (*inspection.returned, *inspection.rejected):
        before = (
            None
            if candidate.diagnostics is None
            else candidate.diagnostics.activation_before_interference
        )
        if before is not None:
            assert candidate.activation <= before + 1e-9
        if candidate.competition is None or candidate.competition.interference is None:
            continue
        penalty = candidate.competition.interference.total_penalty
        assert penalty <= 0.0
        if penalty < 0.0:
            affected += 1
            assert candidate.components.interference <= 0.0
    assert affected > 0
