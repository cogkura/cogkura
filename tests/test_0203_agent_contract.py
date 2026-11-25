"""Agent-facing contracts which must not alter Core retrieval semantics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    MemoryAssessmentFlag,
    ObservationInput,
    RetrievalCue,
    WorkingMemoryConfig,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


async def _missing_knowledge(memory: Memory) -> tuple[list[str], object]:
    await memory.observe(
        ObservationInput(
            tenant_id="agent-contract",
            subject_id="project-a",
            source_namespace="notes",
            source_record_id="astronomy",
            content="Astronomers observed Jupiter through a telescope.",
            observed_at=_T0,
        )
    )
    await memory.process(tenant_id="agent-contract", subject_id="project-a", as_of=_T0)
    cue = RetrievalCue(
        text="what is the production database engine?",
        entity_ids=("production-database",),
        predicate="database_engine",
    )
    kwargs = {
        "tenant_id": "agent-contract",
        "subject_id": "project-a",
        "as_of": _T0 + timedelta(hours=1),
    }
    context = await memory.prepare_context(cue, **kwargs)
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE in context.assessment.flags
    assert context.items
    assert "Jupiter" in context.render()
    assert context.assessment == await memory.assess_memory(cue, **kwargs)
    assert context == await memory.prepare_context(cue, **kwargs)
    return [item.memory.statement for item in context.items], context.assessment.flags


@pytest.mark.asyncio
async def test_missing_knowledge_can_coexist_with_unrelated_context(tmp_path: Path) -> None:
    options = {"activation_config": ActivationConfig(retrieval_threshold=-10)}
    expected = await _missing_knowledge(Memory(**options))
    async with await EmbeddedMemory.open(tmp_path / "answerability.db", **options) as memory:
        assert await _missing_knowledge(memory) == expected


@pytest.mark.asyncio
async def test_reprocessing_unchanged_subject_explains_sqlite_history(tmp_path: Path) -> None:
    import json
    import sqlite3

    path = tmp_path / "unchanged.db"

    def history() -> tuple[list[tuple], ...]:
        with sqlite3.connect(path) as connection:
            return tuple(
                connection.execute(f"SELECT * FROM {table} ORDER BY sequence").fetchall()
                for table in ("episode_history", "semantic_history", "semantic_revision_history")
            )

    async with await EmbeddedMemory.open(
        path,
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    ) as memory:
        for subject in ("project-a", "project-b"):
            await memory.observe(
                ObservationInput(
                    tenant_id="agent-contract",
                    subject_id=subject,
                    source_namespace="notes",
                    source_record_id=subject,
                    content=f"{subject} deployment uses SQLite.",
                    observed_at=_T0,
                    metadata={
                        "semantic_facts": [
                            {
                                "subject_entity_id": subject,
                                "predicate": "database_engine",
                                "object_value": "sqlite",
                                "cardinality": "one",
                                "authority": {"basis": "application_asserted"},
                            }
                        ]
                    },
                )
            )
            await memory.process(tenant_id="agent-contract", subject_id=subject, as_of=_T0)
        before = history()
        assert all(before)
        await memory.process(
            tenant_id="agent-contract", subject_id="project-a", as_of=_T0 + timedelta(days=1)
        )
        after = history()
        assert after[0] == before[0]
        for old_rows, new_rows, field in zip(
            before[1:], after[1:], ("memory", "revision"), strict=True
        ):
            assert new_rows[:-1] == old_rows
            new_row = new_rows[-1]
            old_row = next(row for row in old_rows if row[2] == new_row[2])
            old_snapshot = json.loads(old_row[-1])[field]
            new_snapshot = json.loads(new_row[-1])[field]
            assert old_snapshot.pop("updated_at") != new_snapshot.pop("updated_at")
            assert old_snapshot == new_snapshot
        other_episodes = await memory.list_episodes(
            tenant_id="agent-contract", subject_id="project-b"
        )

    async with await EmbeddedMemory.open(path) as reopened:
        assert history() == after
        assert (
            await reopened.list_episodes(tenant_id="agent-contract", subject_id="project-b")
            == other_episodes
        )


@pytest.mark.asyncio
async def test_public_embedded_lifecycle_reopens_in_new_process(tmp_path: Path) -> None:
    import subprocess
    import sys

    from cogkura import (
        LearningFeedback,
        LearningOutcome,
        MemoryFeedback,
        MemoryIdentity,
        SemanticCorrectionConfig,
    )

    path = tmp_path / "lifecycle.db"
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
        "working_memory_config": WorkingMemoryConfig(max_items=1),
    }
    fact = {
        "subject_entity_id": "project-a",
        "predicate": "storage_engine",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    async with await EmbeddedMemory.open(path, **options) as memory:
        for subject in ("project-a", "project-b"):
            await memory.observe(
                ObservationInput(
                    tenant_id="agent-contract",
                    subject_id=subject,
                    source_namespace="notes",
                    source_record_id=subject,
                    content=f"{subject} storage engine uses PostgreSQL.",
                    observed_at=_T0,
                    metadata={
                        "source_ref": f"docs/{subject}.md",
                        "semantic_facts": [{**fact, "subject_entity_id": subject}],
                    },
                )
            )
            await memory.process(tenant_id="agent-contract", subject_id=subject, as_of=_T0)
        other_before = await memory.list_episodes(
            tenant_id="agent-contract", subject_id="project-b"
        )
        context = await memory.prepare_context(
            "storage engine",
            tenant_id="agent-contract",
            subject_id="project-a",
            as_of=_T0 + timedelta(hours=1),
            prompt_budget_tokens=256,
        )
        assert context.items and context.estimated_tokens <= 256
        inspection = await memory.inspect_recall(
            "storage engine",
            tenant_id="agent-contract",
            subject_id="project-a",
            as_of=_T0 + timedelta(hours=1),
        )
        assert inspection.returned
        provenance = await memory.resolve_provenance(context)
        assert any(
            source.source_ref == "docs/project-a.md"
            for item in provenance.items
            for source in item.sources
        )
        consumed = context.recall_results[0]
        assert len(context.recall_results) == 1
        await memory.record_context_use(context, referenced_at=_T0 + timedelta(hours=1))
        identity = MemoryIdentity(
            memory_kind=consumed.memory_kind, memory_key=consumed.memory.memory_key
        )
        await memory.learn(
            LearningFeedback(
                tenant_id="agent-contract",
                feedback_id="agent-feedback",
                occurred_at=_T0 + timedelta(hours=1),
                items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),),
            )
        )
        predecessor = (
            await memory.list_semantic_revisions(tenant_id="agent-contract", subject_id="project-a")
        )[0]
        await memory.observe(
            ObservationInput(
                tenant_id="agent-contract",
                subject_id="project-a",
                source_namespace="notes",
                source_record_id="correction",
                content="project-a storage engine is SQLite.",
                observed_at=_T0 + timedelta(days=1),
                metadata={
                    "source_ref": "docs/correction.md",
                    "semantic_facts": [
                        {
                            **fact,
                            "object_value": "sqlite",
                            "correction": {
                                "replaces_revision_key": predecessor.revision_key,
                                "effective_at": (_T0 + timedelta(days=1)).isoformat(),
                            },
                        }
                    ],
                },
            )
        )
        result = await memory.process(
            tenant_id="agent-contract", subject_id="project-a", as_of=_T0 + timedelta(days=1)
        )
        assert result.semantics.authoritative_corrections_applied == 1
        assert (
            await memory.list_episodes(tenant_id="agent-contract", subject_id="project-b")
            == other_before
        )
        states = await memory.list_learning_state(tenant_id="agent-contract", identities=[identity])
        assert states[0].helpful_count == 1

    code = """
import asyncio, sys
from datetime import UTC, datetime, timedelta
from cogkura import ActivationConfig, EmbeddedMemory, MemoryIdentity, MemoryKind
async def main():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    owner = await EmbeddedMemory.open(
        sys.argv[1], activation_config=ActivationConfig(retrieval_threshold=-10)
    )
    async with owner as memory:
        for day, expected in [(0, "postgresql"), (2, "sqlite")]:
            at = t + timedelta(days=day, hours=12)
            results = await memory.recall(
                "storage engine", tenant_id="agent-contract", subject_id="project-a",
                as_of=at, valid_at=at,
            )
            values = [r.memory.object_value for r in results
                      if r.memory_kind is MemoryKind.SEMANTIC]
            assert values == [expected], values
        assert await memory.list_episodes(tenant_id="agent-contract", subject_id="project-b")
        identity = MemoryIdentity(memory_kind=MemoryKind(sys.argv[2]), memory_key=sys.argv[3])
        states = await memory.list_learning_state(tenant_id="agent-contract", identities=[identity])
        assert any(s.helpful_count == 1 for s in states)
asyncio.run(main())
"""
    subprocess.run(
        [sys.executable, "-c", code, str(path), identity.memory_kind.value, identity.memory_key],
        check=True,
        timeout=30,
    )


async def _shared_claim_requires_broader_scope(memory: Memory) -> None:
    for subject in ("project-a", "project-b"):
        await memory.observe(
            ObservationInput(
                tenant_id="shared-claim",
                subject_id=subject,
                source_namespace=f"notes-{subject}",
                source_record_id=subject,
                content="The shared platform uses PostgreSQL.",
                observed_at=_T0,
                metadata={
                    "semantic_facts": [
                        {
                            "subject_entity_id": "shared-platform",
                            "predicate": "storage_engine",
                            "object_value": "postgresql",
                        }
                    ]
                },
            )
        )
        await memory.process(tenant_id="shared-claim", subject_id=subject, as_of=_T0)
    assert not await memory.list_semantic_memories(tenant_id="shared-claim")
    await memory.process(tenant_id="shared-claim", as_of=_T0 + timedelta(hours=1))
    claims = await memory.list_semantic_memories(tenant_id="shared-claim")
    assert len(claims) == 1
    assert claims[0].subject_entity_id == "shared-platform"
    assert claims[0].support_count == 2


@pytest.mark.asyncio
async def test_shared_evidence_requires_broader_processing(tmp_path: Path) -> None:
    await _shared_claim_requires_broader_scope(Memory())
    async with await EmbeddedMemory.open(tmp_path / "shared-claim.db") as memory:
        await _shared_claim_requires_broader_scope(memory)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_shared_evidence_requires_broader_processing_postgres(
    postgres_memory_url: str | None,
) -> None:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    from sqlalchemy.ext.asyncio import create_async_engine

    from cogkura.migrations import apply_migrations
    from cogkura.storage.postgres import (
        PostgresActivationStore,
        PostgresCheckpointStore,
        PostgresEntityRelationshipStore,
        PostgresEpisodeStore,
        PostgresInhibitionStore,
        PostgresLearningStore,
        PostgresMemoryDynamicsStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    semantic = PostgresSemanticMemoryStore(engine)
    memory = Memory(
        observation_store=PostgresObservationStore(engine),
        checkpoint_store=PostgresCheckpointStore(engine),
        episode_store=PostgresEpisodeStore(engine),
        semantic_store=semantic,
        semantic_inspection_store=semantic,
        activation_store=PostgresActivationStore(engine),
        dynamics_store=PostgresMemoryDynamicsStore(engine),
        learning_store=PostgresLearningStore(engine),
        inhibition_store=PostgresInhibitionStore(engine),
        entity_relationship_store=PostgresEntityRelationshipStore(engine),
    )
    try:
        await memory.clear(tenant_id="shared-claim")
        await _shared_claim_requires_broader_scope(memory)
        for tenant in ("tenant-a", "tenant-b"):
            await memory.clear(tenant_id=tenant)
        await _scoped_processing_does_not_select_other_tenants(memory)
    finally:
        await memory.clear(tenant_id="shared-claim")
        for tenant in ("tenant-a", "tenant-b"):
            await memory.clear(tenant_id=tenant)
        await engine.dispose()


async def _scoped_processing_does_not_select_other_tenants(memory: Memory) -> None:
    for tenant in ("tenant-a", "tenant-b"):
        await memory.observe(
            ObservationInput(
                tenant_id=tenant,
                subject_id="project-a",
                source_namespace="notes",
                source_record_id="incident",
                content=f"{tenant} incident response.",
                observed_at=_T0,
            )
        )
    result = await memory.process(tenant_id="tenant-a", subject_id="project-a", as_of=_T0)
    assert result.episodes.observations == 1
    assert len(await memory.list_episodes(tenant_id="tenant-a", subject_id="project-a")) == 1
    assert not await memory.list_episodes(tenant_id="tenant-b", subject_id="project-a")
    await memory.process(tenant_id="tenant-b", subject_id="project-a", as_of=_T0)
    assert len(await memory.list_episodes(tenant_id="tenant-b", subject_id="project-a")) == 1


@pytest.mark.asyncio
async def test_scoped_processing_is_tenant_isolated(tmp_path: Path) -> None:
    await _scoped_processing_does_not_select_other_tenants(Memory())
    async with await EmbeddedMemory.open(tmp_path / "tenant-isolation.db") as memory:
        await _scoped_processing_does_not_select_other_tenants(memory)
