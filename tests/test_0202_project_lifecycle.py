"""Project decisions across sessions, corrections, validity and withdrawal."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    MemoryKind,
    ObservationInput,
    SemanticCorrectionConfig,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = "project-lifecycle"
_SUBJECT = "payments-api"


def _event(record: str, day: int, content: str, **metadata: Any) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        source_namespace="project-docs",
        source_record_id=record,
        content=content,
        observed_at=_T0 + timedelta(days=day),
        metadata={"conversation_id": record, "source_ref": f"docs/{record}.md", **metadata},
    )


async def _project_lifecycle(memory: Memory) -> dict[str, Any]:
    fact = {
        "subject_entity_id": _SUBJECT,
        "predicate": "storage_engine",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await memory.observe(
        _event("adr-original", 0, "Storage engine is PostgreSQL.", semantic_facts=[fact])
    )
    await memory.process(tenant_id=_TENANT, as_of=_T0)
    predecessor = (await memory.list_semantic_revisions(tenant_id=_TENANT))[0]
    await memory.observe(
        _event(
            "incident",
            1,
            "Deployment incident: restart workers after database connection failures.",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T0 + timedelta(days=1))
    correction = {
        **fact,
        "object_value": "sqlite",
        "correction": {
            "replaces_revision_key": predecessor.revision_key,
            "effective_at": (_T0 + timedelta(days=3)).isoformat(),
        },
    }
    await memory.observe(
        _event("adr-correction", 2, "Storage will use SQLite.", semantic_facts=[correction])
    )
    applied = await memory.process(tenant_id=_TENANT, as_of=_T0 + timedelta(days=2))
    assert applied.semantics.authoritative_corrections_applied == 1
    successor = next(
        revision
        for revision in await memory.list_semantic_revisions(tenant_id=_TENANT)
        if revision.revision_key != predecessor.revision_key
    )
    await memory.observe(
        _event(
            "adr-withdrawal",
            4,
            "SQLite guidance is withdrawn.",
            semantic_withdrawals=[
                {
                    "target_revision_key": successor.revision_key,
                    "effective_at": (_T0 + timedelta(days=4)).isoformat(),
                    "authority": {"basis": "application_asserted"},
                }
            ],
        )
    )
    withdrawn = await memory.process(tenant_id=_TENANT, as_of=_T0 + timedelta(days=4))
    assert withdrawn.semantics.authoritative_withdrawals_applied == 1
    results = {}
    for name, at, valid_at in (
        ("before_known", _T0 + timedelta(days=1, hours=12), _T0 + timedelta(days=1, hours=12)),
        (
            "known_not_effective",
            _T0 + timedelta(days=2, hours=12),
            _T0 + timedelta(days=2, hours=12),
        ),
        ("effective", _T0 + timedelta(days=3, hours=12), _T0 + timedelta(days=3, hours=12)),
        ("withdrawn", _T0 + timedelta(days=4, hours=12), _T0 + timedelta(days=4, hours=12)),
        (
            "known_future_validity",
            _T0 + timedelta(days=2, hours=12),
            _T0 + timedelta(days=3, hours=12),
        ),
        (
            "known_past_validity",
            _T0 + timedelta(days=3, hours=12),
            _T0 + timedelta(days=2, hours=12),
        ),
    ):
        recalled = await memory.recall(
            "storage engine",
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            as_of=at,
            valid_at=valid_at,
            limit=20,
        )
        values = [
            item.memory.object_value for item in recalled if item.memory_kind is MemoryKind.SEMANTIC
        ]
        context = await memory.prepare_context(
            "storage engine", tenant_id=_TENANT, subject_id=_SUBJECT, as_of=at, valid_at=valid_at
        )
        provenance = await memory.resolve_provenance(context)
        refs = sorted(
            {
                source.source_ref
                for item in provenance.items
                for source in item.sources
                if source.source_ref
            }
        )
        results[name] = {
            "values": values,
            "source_refs": refs,
            "statements": [item.memory.statement for item in recalled],
        }
    assert results["before_known"]["values"] == ["postgresql"]
    assert results["known_not_effective"]["values"] == ["postgresql"]
    assert results["effective"]["values"] == ["sqlite"]
    assert results["withdrawn"]["values"] == []
    assert results["known_future_validity"]["values"] == ["sqlite"]
    assert results["known_past_validity"]["values"] == ["postgresql"]
    assert "docs/adr-original.md" in results["before_known"]["source_refs"]
    assert "docs/adr-correction.md" not in results["before_known"]["source_refs"]
    assert "docs/adr-withdrawal.md" not in results["effective"]["source_refs"]
    assert all("SQLite" not in item for item in results["before_known"]["statements"])
    return results


@pytest.mark.asyncio
async def test_project_lifecycle_history_and_provenance_match_sqlite(tmp_path: Path) -> None:
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    expected = await _project_lifecycle(Memory(**options))
    async with await EmbeddedMemory.open(tmp_path / "project-lifecycle.db", **options) as memory:
        assert await _project_lifecycle(memory) == expected
    async with await EmbeddedMemory.open(tmp_path / "project-lifecycle.db", **options) as memory:
        at = _T0 + timedelta(days=3, hours=12)
        result = await memory.recall(
            "storage engine",
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            as_of=at,
            valid_at=at,
            limit=20,
        )
        assert [
            item.memory.object_value for item in result if item.memory_kind is MemoryKind.SEMANTIC
        ] == ["sqlite"]


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_project_lifecycle_history_and_provenance_match_postgres(
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

    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    expected = await _project_lifecycle(Memory(**options))
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
        **options,
    )
    try:
        await memory.clear(tenant_id=_TENANT)
        assert await _project_lifecycle(memory) == expected
    finally:
        await memory.clear(tenant_id=_TENANT)
        await engine.dispose()
