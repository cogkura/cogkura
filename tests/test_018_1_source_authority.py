"""Source-authority admission for semantic consolidation (0.18.1)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import (
    ComplementaryLearningSemanticConsolidator,
    Memory,
    MemoryKind,
    ObservationInput,
    SemanticPromotionBasis,
    semantic_promotion_from_metadata,
)
from cogkura.storage.in_memory_observation import InMemoryObservationStore

_T = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
_TENANT = "repository"


def _memory(*, allow: bool) -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=allow,
        ),
    )


def _fact(
    predicate: str,
    object_value: str,
    *,
    authority: bool = False,
    cardinality: str = "one",
    confidence: float | None = None,
) -> dict[str, object]:
    fact: dict[str, object] = {
        "predicate": predicate,
        "object_value": object_value,
        "cardinality": cardinality,
    }
    if confidence is not None:
        fact["confidence"] = confidence
    if authority:
        fact["authority"] = {"basis": "application_asserted"}
    return fact


def _observation(
    *,
    record_id: str,
    content: str,
    facts: list[dict[str, object]] | None = None,
    tenant_id: str = _TENANT,
    subject_id: str = "payments-api",
    conversation_id: str = "decision-1",
    version: str = "1",
    observed_at: datetime | None = None,
    source_ref: str | None = None,
    status: str | None = None,
    is_deleted: bool = False,
) -> ObservationInput:
    metadata: dict[str, object] = {"conversation_id": conversation_id}
    if facts is not None:
        metadata["semantic_facts"] = facts
    if source_ref is not None:
        metadata["source_ref"] = source_ref
    if status is not None:
        metadata["status"] = status
    return ObservationInput(
        tenant_id=tenant_id,
        subject_id=subject_id,
        source_namespace="repository",
        source_record_id=record_id,
        source_version=version,
        content=content,
        observed_at=observed_at or _T,
        metadata=metadata,
        is_deleted=is_deleted,
    )


async def _semantic(memory: Memory, *, tenant_id: str = _TENANT, subject_id: str | None = None):
    return await memory.list_semantic_memories(
        tenant_id=tenant_id,
        subject_id=subject_id,
    )


@pytest.mark.asyncio
async def test_one_ordinary_fact_is_not_promoted() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="note-1",
            content="Migrations start with the application.",
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    assert result.semantics.authority_promoted == 0
    assert result.semantics.recurrence_promoted == 0
    assert await _semantic(memory) == []


@pytest.mark.asyncio
async def test_two_ordinary_episodes_promote_by_recurrence() -> None:
    memory = _memory(allow=True)
    for record_id, conversation_id in (("note-1", "a"), ("note-2", "b")):
        await memory.observe(
            _observation(
                record_id=record_id,
                conversation_id=conversation_id,
                content="Migrations start with the application.",
                facts=[_fact("migration_execution_strategy", "application_startup")],
            )
        )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    memories = await _semantic(memory)
    assert result.semantics.recurrence_promoted == 1
    assert result.semantics.authority_promoted == 0
    assert len(memories) == 1
    assert memories[0].support_count == 2
    promotion = semantic_promotion_from_metadata(memories[0].metadata)
    assert promotion is not None
    assert promotion.basis is SemanticPromotionBasis.RECURRENCE


@pytest.mark.asyncio
async def test_explicit_authority_promotes_one_episode_when_enabled() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Database migrations run through the deployment pipeline.",
            source_ref="docs/adr/0001.md",
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    authority=True,
                    confidence=0.8,
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    memories = await _semantic(memory)
    assert result.semantics.authority_promoted == 1
    assert result.semantics.created == 1
    assert len(memories) == 1
    assert memories[0].support_count == 1
    assert memories[0].confidence < 1.0
    promotion = semantic_promotion_from_metadata(memories[0].metadata)
    assert promotion is not None
    assert promotion.basis is SemanticPromotionBasis.EXPLICIT_SOURCE_AUTHORITY
    assert promotion.observation_revision == 1
    assert promotion.support_count == 1
    assert promotion.claim_key == memories[0].memory_key


@pytest.mark.asyncio
async def test_authority_is_ignored_when_the_feature_is_disabled() -> None:
    memory = _memory(allow=False)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Database migrations run through the deployment pipeline.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    assert result.semantics.authority_promoted == 0
    assert result.semantics.authority_invalid == 0
    assert await _semantic(memory) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "facts,metadata_status,source_ref",
    [
        (
            [_fact("migration_execution_strategy", "deployment_pipeline", confidence=1.0)],
            None,
            None,
        ),
        ([_fact("migration_execution_strategy", "deployment_pipeline")], "confirmed", None),
        ([_fact("migration_execution_strategy", "deployment_pipeline")], None, "docs/adr/0001.md"),
    ],
)
async def test_confidence_status_and_source_ref_do_not_bypass_recurrence(
    facts: list[dict[str, object]],
    metadata_status: str | None,
    source_ref: str | None,
) -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Database migrations run through the deployment pipeline.",
            facts=facts,
            status=metadata_status,
            source_ref=source_ref,
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    assert result.semantics.authority_promoted == 0
    assert await _semantic(memory) == []


@pytest.mark.asyncio
async def test_authority_does_not_leak_to_other_facts_in_the_episode() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Pipeline migrations and a PostgreSQL database.",
            facts=[
                _fact("migration_execution_strategy", "deployment_pipeline", authority=True),
                _fact("database_engine", "postgresql"),
            ],
        )
    )
    await memory.observe(
        _observation(
            record_id="note-2",
            conversation_id="decision-1",
            content="The service also records its owner.",
            facts=[_fact("owning_team", "payments")],
            observed_at=_T.replace(minute=5),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    memories = await _semantic(memory)
    assert [memory.predicate for memory in memories] == ["migration_execution_strategy"]


@pytest.mark.asyncio
async def test_same_claim_keeps_authority_evidence_despite_higher_confidence() -> None:
    store = InMemoryObservationStore()
    memory = Memory(
        observation_store=store,
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True,
        ),
    )
    await memory.observe(
        _observation(
            record_id="high",
            content="Pipeline migrations.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", confidence=1.0)],
        )
    )
    await memory.observe(
        _observation(
            record_id="low",
            content="Pipeline migrations, confirmed.",
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    authority=True,
                    confidence=0.8,
                )
            ],
            observed_at=_T.replace(minute=1),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    authorised = await store.get_by_source(
        tenant_id=_TENANT,
        source_namespace="repository",
        source_record_id="low",
    )
    assert authorised is not None
    memories = await _semantic(memory)
    promotion = semantic_promotion_from_metadata(memories[0].metadata)
    assert promotion is not None
    assert promotion.observation_id == authorised.id
    assert memories[0].support_count == 1


@pytest.mark.asyncio
async def test_authority_does_not_cross_tenants_or_subjects() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-payments",
            tenant_id="tenant-a",
            subject_id="payments-api",
            content="Payments migrations use the pipeline.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    await memory.observe(
        _observation(
            record_id="adr-billing",
            tenant_id="tenant-a",
            subject_id="billing-api",
            content="Billing migrations start with the application.",
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    await memory.observe(
        _observation(
            record_id="adr-other",
            tenant_id="tenant-b",
            subject_id="payments-api",
            content="Other tenant migrations.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline")],
        )
    )
    await memory.process(tenant_id="tenant-a", as_of=_T)
    await memory.process(tenant_id="tenant-b", as_of=_T)
    payments = await _semantic(memory, tenant_id="tenant-a", subject_id="payments-api")
    billing = await _semantic(memory, tenant_id="tenant-a", subject_id="billing-api")
    other = await _semantic(memory, tenant_id="tenant-b")
    assert len(payments) == 1
    assert billing == []
    assert other == []


@pytest.mark.asyncio
async def test_missing_revision_fails_closed() -> None:
    store = InMemoryObservationStore()
    memory = Memory(
        observation_store=store,
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True,
        ),
    )
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Pipeline migrations.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    await memory.encode_episodes(tenant_id=_TENANT, as_of=_T)
    stored = await store.get_by_source(
        tenant_id=_TENANT,
        source_namespace="repository",
        source_record_id="adr-0001",
    )
    assert stored is not None
    store.revisions[stored.id].clear()
    result = await memory.consolidate_semantics(tenant_id=_TENANT, as_of=_T)
    assert result.authority_invalid == 1
    assert result.authority_promoted == 0
    assert await _semantic(memory) == []


@pytest.mark.asyncio
async def test_later_revision_does_not_authorise_earlier_evidence() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            version="1",
            content="Migrations start with the application.",
            facts=[_fact("migration_execution_strategy", "application_startup")],
        )
    )
    await memory.encode_episodes(tenant_id=_TENANT, as_of=_T)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            version="2",
            content="Migrations run through the deployment pipeline.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
            observed_at=_T.replace(hour=11),
        )
    )
    result = await memory.consolidate_semantics(tenant_id=_TENANT, as_of=_T.replace(hour=11))
    assert result.authority_promoted == 0
    assert result.authority_invalid == 1
    assert await _semantic(memory) == []


@pytest.mark.asyncio
async def test_deleted_revision_is_not_admitted() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            version="1",
            content="Pipeline migrations.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    await memory.encode_episodes(tenant_id=_TENANT, as_of=_T)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            version="2",
            content="Pipeline migrations.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
            observed_at=_T.replace(hour=11),
            is_deleted=True,
        )
    )
    result = await memory.consolidate_semantics(tenant_id=_TENANT, as_of=_T.replace(hour=11))
    assert result.authority_invalid == 1
    assert await _semantic(memory) == []


@pytest.mark.asyncio
async def test_malformed_authority_is_not_a_bypass() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Pipeline migrations.",
            facts=[
                {
                    "predicate": "migration_execution_strategy",
                    "object_value": "deployment_pipeline",
                    "cardinality": "one",
                    "authority": {"basis": "confirmed"},
                }
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    assert result.semantics.extracted_failures == 0
    assert result.semantics.authority_invalid == 1
    assert result.semantics.authority_promoted == 0
    assert await _semantic(memory) == []


@pytest.mark.asyncio
async def test_repeated_processing_is_idempotent_and_provenance_survives() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Database migrations run through the deployment pipeline.",
            source_ref="docs/adr/0001.md",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    first = await memory.process(tenant_id=_TENANT, as_of=_T)
    second = await memory.process(tenant_id=_TENANT, as_of=_T)
    third = await memory.process(tenant_id=_TENANT, as_of=_T)
    assert first.semantics.created == 1
    assert second.semantics.unchanged == 1
    assert third.semantics.unchanged == 1
    assert second.semantics.authority_promoted == 1
    memories = await _semantic(memory)
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    assert len(memories) == 1
    assert memories[0].support_count == 1
    assert len(revisions) == 1
    promotion = semantic_promotion_from_metadata(memories[0].metadata)
    assert promotion is not None
    assert promotion.observation_revision == 1

    context = await memory.prepare_context(
        "payments-api migration execution strategy deployment pipeline",
        tenant_id=_TENANT,
        as_of=_T,
    )
    provenance = await memory.resolve_provenance(context)
    semantic = next(
        item for item in provenance.items if item.memory_identity.memory_kind is MemoryKind.SEMANTIC
    )
    assert semantic.promotion == promotion
    assert semantic.sources
    assert "explicit_source_authority" not in context.render()
    assert "application_asserted" not in context.render()


@pytest.mark.asyncio
async def test_later_support_preserves_original_authority_basis() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            conversation_id="decision-1",
            content="Pipeline migrations.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    await memory.observe(
        _observation(
            record_id="note-2",
            conversation_id="decision-2",
            content="Pipeline migrations again.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline")],
            observed_at=_T.replace(day=2),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T.replace(day=2))
    memories = await _semantic(memory)
    revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    assert len(memories) == 1
    assert memories[0].support_count == 2
    assert len(revisions) == 1
    promotion = semantic_promotion_from_metadata(memories[0].metadata)
    assert promotion is not None
    assert promotion.basis is SemanticPromotionBasis.EXPLICIT_SOURCE_AUTHORITY
    assert promotion.support_count == 1


@pytest.mark.asyncio
async def test_competing_authority_does_not_replace_a_current_claim() -> None:
    memory = _memory(allow=True)
    for record_id, conversation_id in (("old-1", "a"), ("old-2", "b")):
        await memory.observe(
            _observation(
                record_id=record_id,
                conversation_id=conversation_id,
                content="Migrations start with the application.",
                facts=[_fact("migration_execution_strategy", "application_startup")],
            )
        )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    await memory.observe(
        _observation(
            record_id="adr-new",
            conversation_id="c",
            content="Migrations run through the deployment pipeline.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
            observed_at=_T.replace(day=3),
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T.replace(day=3))
    assert result.semantics.authority_deferred_conflict == 1
    assert result.semantics.authority_promoted == 0
    memories = await _semantic(memory)
    assert [item.object_value for item in memories] == ["application_startup"]


@pytest.mark.asyncio
async def test_incompatible_authority_claims_in_one_batch_have_no_winner() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-a",
            conversation_id="a",
            content="Application startup.",
            facts=[_fact("migration_execution_strategy", "application_startup", authority=True)],
        )
    )
    await memory.observe(
        _observation(
            record_id="adr-b",
            conversation_id="b",
            content="Deployment pipeline.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    assert result.semantics.authority_deferred_conflict == 2
    assert result.semantics.authority_promoted == 0
    assert await _semantic(memory) == []


@pytest.mark.asyncio
async def test_cardinality_many_authority_claims_coexist() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            content="Supported databases.",
            facts=[
                _fact("database_engine", "postgresql", authority=True, cardinality="many"),
                _fact("database_engine", "mysql", authority=True, cardinality="many"),
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_T)
    memories = await _semantic(memory)
    assert result.semantics.authority_promoted == 2
    assert result.semantics.authority_deferred_conflict == 0
    assert {item.object_value for item in memories} == {"postgresql", "mysql"}
    assert all(item.support_count == 1 for item in memories)


@pytest.mark.asyncio
async def test_recurrence_qualified_claims_still_reconcile_when_authority_is_enabled() -> None:
    async def _objects(*, allow: bool) -> list[str]:
        memory = _memory(allow=allow)
        for record_id, conversation_id, value in (
            ("old-1", "a", "application_startup"),
            ("old-2", "b", "application_startup"),
            ("new-1", "c", "deployment_pipeline"),
            ("new-2", "d", "deployment_pipeline"),
        ):
            await memory.observe(
                _observation(
                    record_id=record_id,
                    conversation_id=conversation_id,
                    content=f"Strategy is {value}.",
                    facts=[_fact("migration_execution_strategy", value)],
                    observed_at=_T if value == "application_startup" else _T.replace(day=4),
                )
            )
        await memory.process(tenant_id=_TENANT, as_of=_T.replace(day=4))
        memories = await memory.list_semantic_memories(tenant_id=_TENANT, include_inactive=True)
        return sorted(f"{item.object_value}:{item.status.value}" for item in memories)

    assert await _objects(allow=False) == await _objects(allow=True)


@pytest.mark.asyncio
async def test_authority_confidence_and_recall_match_single_episode_consolidation() -> None:
    fact = _fact(
        "migration_execution_strategy",
        "deployment_pipeline",
        authority=True,
        confidence=0.8,
    )
    ordinary = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )
    authorised = _memory(allow=True)
    observation = _observation(
        record_id="adr-0001",
        content="Database migrations run through the deployment pipeline.",
        facts=[fact],
    )
    plain = _observation(
        record_id="adr-0001",
        tenant_id="ordinary",
        content="Database migrations run through the deployment pipeline.",
        facts=[{key: value for key, value in fact.items() if key != "authority"}],
    )
    await ordinary.observe(plain)
    await authorised.observe(observation)
    await ordinary.process(tenant_id="ordinary", as_of=_T)
    await authorised.process(tenant_id=_TENANT, as_of=_T)
    ordinary_memory = (await _semantic(ordinary, tenant_id="ordinary"))[0]
    authority_memory = (await _semantic(authorised))[0]
    assert authority_memory.confidence == ordinary_memory.confidence
    assert authority_memory.importance == ordinary_memory.importance
    query = "payments-api migration execution strategy deployment pipeline"
    ordinary_recall = await ordinary.recall(query, tenant_id="ordinary", as_of=_T)
    authority_recall = await authorised.recall(query, tenant_id=_TENANT, as_of=_T)
    ordinary_semantic = next(
        item for item in ordinary_recall if item.memory_kind is MemoryKind.SEMANTIC
    )
    authority_semantic = next(
        item for item in authority_recall if item.memory_kind is MemoryKind.SEMANTIC
    )
    assert authority_semantic.activation == ordinary_semantic.activation
    assert authority_semantic.score == ordinary_semantic.score


@pytest.mark.asyncio
async def test_withdrawn_authority_does_not_retract_the_semantic_memory() -> None:
    memory = _memory(allow=True)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            version="1",
            content="Pipeline migrations.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline", authority=True)],
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    await memory.observe(
        _observation(
            record_id="adr-0001",
            version="2",
            content="Pipeline migrations, authority removed.",
            facts=[_fact("migration_execution_strategy", "deployment_pipeline")],
            observed_at=_T.replace(day=2),
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T.replace(day=2))
    memories = await _semantic(memory)
    assert len(memories) == 1
    assert memories[0].support_count == 1
    promotion = semantic_promotion_from_metadata(memories[0].metadata)
    assert promotion is not None
    assert promotion.basis is SemanticPromotionBasis.EXPLICIT_SOURCE_AUTHORITY
    assert promotion.observation_revision == 1
