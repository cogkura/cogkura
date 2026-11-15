"""SQLite semantic correction, persistence and transactional guarantees."""

import inspect
from datetime import UTC, datetime
from pathlib import Path

import pytest
import test_018_2_authoritative_corrections as correction_contract
from test_018_2_authoritative_corrections import (
    _CORRECTION,
    _LATER,
    _T,
    _TENANT,
    _adopt,
    _fact,
    _observation,
    _values,
)

from cogkura import (
    ComplementaryLearningSemanticConsolidator,
    Memory,
    SemanticCorrectionConfig,
)
from cogkura.exceptions import StorageError
from cogkura.storage.sqlite import SQLiteObservationStore, SQLiteStorage
from cogkura.storage.sqlite_episode import SQLiteEpisodeStore
from cogkura.storage.sqlite_semantic import SQLiteSemanticMemoryStore


def _memory(storage: SQLiteStorage) -> Memory:
    return Memory(
        observation_store=SQLiteObservationStore(storage),
        episode_store=SQLiteEpisodeStore(storage),
        semantic_store=SQLiteSemanticMemoryStore(storage),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
    )


@pytest.mark.asyncio
async def test_correction_and_provenance_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    memory = _memory(storage)
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations now run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    assert result.semantics.authoritative_corrections_applied == 1
    before = await memory.list_semantic_revisions(tenant_id=_TENANT)
    current = await memory.list_semantic_memories(tenant_id=_TENANT)
    assert len(before) == 2 and len(current) == 1
    await storage.close()
    storage = await SQLiteStorage.open(path)
    memory = _memory(storage)
    assert await memory.list_semantic_revisions(tenant_id=_TENANT) == before
    assert await memory.list_semantic_memories(tenant_id=_TENANT) == current
    assert await _values(memory, valid_at=datetime(2026, 10, 15, tzinfo=UTC)) == [
        "application_startup"
    ]
    assert await _values(memory, valid_at=_CORRECTION) == ["deployment_pipeline"]
    context = await memory.prepare_context(
        "payments-api migration execution strategy", tenant_id=_TENANT, as_of=_LATER
    )
    provenance = await memory.resolve_provenance(context)
    assert provenance.items
    sources = [source for item in provenance.items for source in item.sources]
    assert any(
        source.source_namespace == "repository"
        and source.source_record_id == "adr-0002"
        and source.source_version == "1"
        for source in sources
    )
    await memory.process(tenant_id=_TENANT, as_of=_LATER)
    assert len(await memory.list_semantic_revisions(tenant_id=_TENANT)) == 2
    await storage.close()


@pytest.mark.asyncio
async def test_withdrawal_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    memory = _memory(storage)
    predecessor = await _adopt(memory)
    await memory.observe(
        _observation(
            record_id="withdraw",
            content="The migration strategy is withdrawn.",
            observed_at=_CORRECTION,
            withdrawals=[
                {
                    "target_revision_key": predecessor,
                    "effective_at": _CORRECTION.isoformat(),
                    "authority": {"basis": "application_asserted"},
                }
            ],
        )
    )
    result = await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    assert result.semantics.authoritative_withdrawals_applied == 1
    await storage.close()
    storage = await SQLiteStorage.open(path)
    memory = _memory(storage)
    assert await _values(memory) == []
    assert await _values(memory, valid_at=_T) == ["application_startup"]
    await storage.close()


@pytest.mark.asyncio
async def test_reconciliation_failure_rolls_back(tmp_path: Path) -> None:
    storage = await SQLiteStorage.open(tmp_path / "db")
    memory = _memory(storage)
    await _adopt(memory)
    before = await memory.list_semantic_revisions(tenant_id=_TENANT)
    await storage._run(
        lambda: storage._conn().execute("""CREATE TRIGGER fail_revision_update
        BEFORE UPDATE ON semantic_revisions BEGIN SELECT RAISE(ABORT, 'injected'); END""")
    )
    predecessor = before[0].revision_key
    await memory.observe(
        _observation(
            record_id="adr-0002",
            content="Migrations run during deployment.",
            observed_at=_CORRECTION,
            facts=[
                _fact(
                    "migration_execution_strategy",
                    "deployment_pipeline",
                    correction={
                        "replaces_revision_key": predecessor,
                        "effective_at": _CORRECTION.isoformat(),
                    },
                )
            ],
        )
    )
    with pytest.raises(StorageError, match="injected"):
        await memory.process(tenant_id=_TENANT, as_of=_CORRECTION)
    assert await memory.list_semantic_revisions(tenant_id=_TENANT) == before
    assert await _values(memory) == ["application_startup"]
    await storage.close()


# Run the existing authoritative-correction invariants against the SQLite stores.
_CORRECTION_CASES = [
    name
    for name, value in vars(correction_contract).items()
    if name.startswith("test_")
    and inspect.iscoroutinefunction(value)
    and not inspect.signature(value).parameters
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CORRECTION_CASES)
async def test_correction_contract_on_sqlite(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cogkura import ActivationConfig

    storage = await SQLiteStorage.open(tmp_path / "contract.db")

    def create_memory(
        *, correction: bool = True, retrieval_threshold: float | None = None
    ) -> Memory:
        return Memory(
            observation_store=SQLiteObservationStore(storage),
            episode_store=SQLiteEpisodeStore(storage),
            semantic_store=SQLiteSemanticMemoryStore(storage),
            semantic_consolidator=ComplementaryLearningSemanticConsolidator(
                allow_authoritative_single_episode=True
            ),
            semantic_correction_config=SemanticCorrectionConfig(enabled=correction),
            activation_config=(
                ActivationConfig(retrieval_threshold=retrieval_threshold)
                if retrieval_threshold is not None
                else None
            ),
        )

    monkeypatch.setattr(correction_contract, "_memory", create_memory)
    try:
        await getattr(correction_contract, case)()
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_semantics_isolate_tenants_and_clear(tmp_path: Path) -> None:
    from dataclasses import replace

    storage = await SQLiteStorage.open(tmp_path / "db")
    memory = _memory(storage)
    first = _observation(
        record_id="same",
        content="The strategy is startup.",
        facts=[_fact("migration_execution_strategy", "startup")],
    )
    await memory.observe(first)
    await memory.observe(replace(first, tenant_id="other"))
    await memory.process(tenant_id=_TENANT, as_of=_T)
    await memory.process(tenant_id="other", as_of=_T)
    source = await memory.list_semantic_memories(tenant_id=_TENANT)
    other = await memory.list_semantic_memories(tenant_id="other")
    assert len(source) == len(other) == 1
    assert source[0].id != other[0].id
    assert source[0].observation_evidence != other[0].observation_evidence
    await memory.clear(tenant_id=_TENANT)
    assert await memory.list_semantic_memories(tenant_id=_TENANT) == []
    assert await memory.list_semantic_memories(tenant_id="other") == other
    await storage.close()
