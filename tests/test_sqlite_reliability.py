"""Process recovery and SQLite-supported backup verification."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_018_2_authoritative_corrections import _T, _TENANT, _adopt

from cogkura import (
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    ObservationInput,
    SemanticCorrectionConfig,
)
from cogkura.algorithms.learning import DeterministicLearningProcessor
from cogkura.models import (
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
)
from cogkura.storage.sqlite import SQLiteCheckpointStore
from cogkura.storage.sqlite_cognitive import SQLiteLearningStore


async def _child(program: str, *args: str) -> tuple[int | None, str, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        program,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


def _observation(record_id: str) -> ObservationInput:
    return ObservationInput(
        tenant_id="tenant",
        source_namespace="src",
        source_record_id=record_id,
        source_version="1",
        content="A durable source observation used in recovery testing.",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _integrity(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.asyncio
async def test_abrupt_exit_rolls_back_observation_revision_pair(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    async with await EmbeddedMemory.open(path) as memory:
        await memory.observe(_observation("existing"))
    program = """import asyncio, os, sys
from datetime import UTC, datetime
from cogkura import EmbeddedMemory, ObservationInput
async def main():
    owner = await EmbeddedMemory.open(sys.argv[1])
    def setup():
        conn = owner._storage._conn()
        conn.create_function("crash_now", 0, lambda: os._exit(73))
        conn.execute("CREATE TRIGGER crash_revision BEFORE INSERT ON observation_revisions "
                     "BEGIN SELECT crash_now(); END")
    await owner._storage._run(setup)
    await owner.memory.observe(ObservationInput(
        tenant_id="tenant", source_namespace="src", source_record_id="new", source_version="1",
        content="This observation must not be partially stored.",
        observed_at=datetime(2026,1,1,tzinfo=UTC)))
asyncio.run(main())
"""
    code, _, err = await _child(program, os.fspath(path))
    assert code == 73, err
    _integrity(path)
    async with await EmbeddedMemory.open(path) as memory:
        rows = await memory._observation_store.list(tenant_id="tenant")
        assert [item.source_record_id for item in rows] == ["existing"]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM observation_revisions").fetchone()[0] == 1
        conn.execute("DROP TRIGGER crash_revision")


@pytest.mark.asyncio
async def test_abrupt_exit_rolls_back_learning_plan_and_allows_retry(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    async with await EmbeddedMemory.open(path):
        pass
    program = """import asyncio, os, sys
from datetime import UTC, datetime
from cogkura import EmbeddedMemory
from cogkura.algorithms.learning import DeterministicLearningProcessor
from cogkura.models import (LearningConfig, LearningFeedback, LearningOutcome,
                            MemoryFeedback, MemoryIdentity, MemoryKind)
from cogkura.storage.sqlite_cognitive import SQLiteLearningStore
async def main():
    owner = await EmbeddedMemory.open(sys.argv[1])
    def setup():
        conn = owner._storage._conn()
        conn.create_function("crash_now", 0, lambda: os._exit(74))
        conn.execute("CREATE TRIGGER crash_learning BEFORE INSERT ON learning_state "
                     "BEGIN SELECT crash_now(); END")
    await owner._storage._run(setup)
    feedback = LearningFeedback(
        tenant_id="tenant",feedback_id="feedback",occurred_at=datetime(2026,1,1,tzinfo=UTC),
        items=(MemoryFeedback(identity=MemoryIdentity(MemoryKind.EPISODE,"episode"),
                              outcome=LearningOutcome.HELPFUL),))
    plan = DeterministicLearningProcessor().plan(feedback=feedback,config=LearningConfig())
    await SQLiteLearningStore(owner._storage).apply(plan)
asyncio.run(main())
"""
    code, _, err = await _child(program, os.fspath(path))
    assert code == 74, err
    _integrity(path)
    with sqlite3.connect(path) as conn:
        for table in (
            "learning_events",
            "learning_feedback",
            "learning_state",
            "learned_associations",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        conn.execute("DROP TRIGGER crash_learning")
    owner = await EmbeddedMemory.open(path)
    identity = MemoryIdentity(MemoryKind.EPISODE, "episode")
    feedback = LearningFeedback(
        tenant_id="tenant",
        feedback_id="feedback",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),),
    )
    plan = DeterministicLearningProcessor().plan(feedback=feedback, config=LearningConfig())
    store = SQLiteLearningStore(owner._storage)
    assert (await store.apply(plan)).created
    assert (await store.apply(plan)).unchanged
    await owner.close()


@pytest.mark.asyncio
async def test_commits_survive_abrupt_exit_without_close(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    program = """import asyncio, os, sys
from datetime import UTC, datetime
from cogkura import EmbeddedMemory, ObservationInput
async def main():
    owner = await EmbeddedMemory.open(sys.argv[1])
    await owner.memory.observe(ObservationInput(
        tenant_id="tenant",source_namespace="src",source_record_id="1",source_version="1",
        content="This committed observation survives abrupt exit.",
        observed_at=datetime(2026,1,1,tzinfo=UTC)))
    await owner.memory.process(tenant_id="tenant", as_of=datetime(2026,1,1,tzinfo=UTC))
    os._exit(0)
asyncio.run(main())
"""
    code, _, err = await _child(program, os.fspath(path))
    assert code == 0, err
    _integrity(path)
    async with await EmbeddedMemory.open(path) as memory:
        assert len(await memory.list_episodes(tenant_id="tenant")) == 1
        assert len(await memory._observation_store.list(tenant_id="tenant")) == 1


@pytest.mark.asyncio
async def test_online_backup_restores_a_consistent_snapshot(tmp_path: Path) -> None:
    path, backup = tmp_path / "live.db", tmp_path / "backup.db"
    owner = await EmbeddedMemory.open(
        path,
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
    )
    await _adopt(owner.memory)
    semantic = (await owner.memory.list_semantic_memories(tenant_id=_TENANT))[0]
    identity = MemoryIdentity(MemoryKind.SEMANTIC, semantic.memory_key)
    feedback = LearningFeedback(
        tenant_id=_TENANT,
        feedback_id="backup-feedback",
        occurred_at=_T,
        items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),),
    )
    assert (await owner.memory.learn(feedback)).created
    await SQLiteCheckpointStore(owner._storage).set(
        tenant_id=_TENANT, connector_id="source", checkpoint={"cursor": 7}
    )
    semantic_before = await owner.memory.list_semantic_memories(tenant_id=_TENANT)
    revisions_before = await owner.memory.list_semantic_revisions(tenant_id=_TENANT)
    learning_before = await owner.memory.list_learning_state(
        tenant_id=_TENANT, identities=[identity]
    )
    for i in range(5):
        await owner.memory.observe(_observation(f"before-{i}"))
    await owner.memory.process(tenant_id="tenant", as_of=datetime(2026, 1, 1, tzinfo=UTC))
    episodes = await owner.memory.list_episodes(tenant_id="tenant")

    async def writer() -> None:
        for i in range(20):
            await owner.memory.observe(_observation(f"during-{i}"))
            await asyncio.sleep(0)

    def backup_snapshot() -> None:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            backup.touch(mode=0o600, exist_ok=False)
            with sqlite3.connect(backup) as destination:
                source.backup(destination)

    await asyncio.gather(writer(), asyncio.to_thread(backup_snapshot))
    await owner.close()
    _integrity(path)
    _integrity(backup)
    async with await EmbeddedMemory.open(backup) as restored:
        observations = await restored._observation_store.list(tenant_id="tenant")
        assert 5 <= len(observations) <= 25
        assert await restored.list_episodes(tenant_id="tenant") == episodes
        assert await restored.list_semantic_memories(tenant_id=_TENANT) == semantic_before
        assert await restored.list_semantic_revisions(tenant_id=_TENANT) == revisions_before
        assert (
            await restored.list_learning_state(tenant_id=_TENANT, identities=[identity])
            == learning_before
        )
        assert await restored._checkpoint_store.get(tenant_id=_TENANT, connector_id="source") == {
            "cursor": 7
        }
        assert {f"before-{i}" for i in range(5)} <= {item.source_record_id for item in observations}


@pytest.mark.asyncio
async def test_abrupt_exit_rolls_back_semantic_reconciliation(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    options = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    async with await EmbeddedMemory.open(path, **options) as memory:
        predecessor = await _adopt(memory)
        before = await memory.list_semantic_memories(tenant_id=_TENANT)
        revisions = await memory.list_semantic_revisions(tenant_id=_TENANT)
    program = """import asyncio, os, sys
from datetime import UTC, datetime
from cogkura import (EmbeddedMemory, ObservationInput, ComplementaryLearningSemanticConsolidator,
                     SemanticCorrectionConfig)
async def main():
    owner = await EmbeddedMemory.open(sys.argv[1],
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True))
    def setup():
        conn = owner._storage._conn()
        conn.create_function("crash_now", 0, lambda: os._exit(75))
        conn.execute("CREATE TRIGGER crash_semantic BEFORE UPDATE ON semantic_revisions "
                     "BEGIN SELECT crash_now(); END")
    await owner._storage._run(setup)
    at = datetime(2026,10,20,10,0,tzinfo=UTC)
    fact = {"predicate":"migration_execution_strategy", "object_value":"deployment_pipeline",
        "cardinality":"one", "authority":{"basis":"application_asserted"},
        "correction":{"replaces_revision_key":sys.argv[2],"effective_at":at.isoformat()}}
    await owner.memory.observe(ObservationInput(tenant_id="repository", subject_id="payments-api",
        source_namespace="repository",source_record_id="correction",source_version="1",
        content="Migrations now run during deployment.",observed_at=at,
        metadata={"conversation_id":"correction","semantic_facts":[fact]}))
    await owner.memory.process(tenant_id="repository",as_of=at)
asyncio.run(main())
"""
    code, _, err = await _child(program, os.fspath(path), predecessor)
    assert code == 75, err
    _integrity(path)
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER crash_semantic")
    async with await EmbeddedMemory.open(path, **options) as memory:
        assert await memory.list_semantic_memories(tenant_id=_TENANT) == before
        assert await memory.list_semantic_revisions(tenant_id=_TENANT) == revisions
        # Earlier observation/episode stages are independent commits, so retry is required.
        result = await memory.process(
            tenant_id=_TENANT, as_of=datetime(2026, 10, 20, 10, tzinfo=UTC)
        )
        assert result.semantics.authoritative_corrections_applied == 1
        assert len(await memory.list_semantic_revisions(tenant_id=_TENANT)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_count", [1, 2, 3, 4])
async def test_upgrade_each_schema_preserves_existing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema_count: int
) -> None:
    from cogkura.models import EpisodeEvidenceInput, EpisodeInput
    from cogkura.storage import sqlite as sqlite_module
    from cogkura.storage.sqlite import SQLiteObservationStore
    from cogkura.storage.sqlite_episode import SQLiteEpisodeStore

    paths = sqlite_module._migrations()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths[:schema_count])
    path = tmp_path / f"old-{schema_count}.db"
    owner = await EmbeddedMemory.open(
        path,
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    observation = _observation("upgrade")
    await owner.memory.observe(observation)
    current = (await SQLiteObservationStore(owner._storage).list(tenant_id="tenant"))[0]
    episode = None
    if schema_count >= 2:
        episode = EpisodeInput(
            tenant_id="tenant",
            subject_id=None,
            memory_key="episode-key",
            statement="Historical upgrade episode.",
            started_at=observation.observed_at,
            ended_at=observation.observed_at,
            confidence=0.8,
            importance=0.5,
            evidence=(EpisodeEvidenceInput(current.id, 1, 0),),
            metadata={"episode": {"content_fingerprint": "upgrade"}},
        )
        await SQLiteEpisodeStore(owner._storage).upsert(episode, as_of=observation.observed_at)
    await SQLiteCheckpointStore(owner._storage).set(
        tenant_id="tenant", connector_id="c", checkpoint={"cursor": 3}
    )
    semantic_before = []
    revision_before = []
    if schema_count >= 3:
        await _adopt(owner.memory)
        semantic_before = await owner.memory.list_semantic_memories(tenant_id=_TENANT)
        revision_before = await owner.memory.list_semantic_revisions(tenant_id=_TENANT)
        assert semantic_before and revision_before
    state_before = ()
    identity = MemoryIdentity(MemoryKind.EPISODE, "episode-key")
    if schema_count >= 4:
        feedback = LearningFeedback(
            tenant_id="tenant",
            feedback_id="upgrade-feedback",
            occurred_at=_T,
            items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),),
        )
        plan = DeterministicLearningProcessor().plan(feedback=feedback, config=LearningConfig())
        await SQLiteLearningStore(owner._storage).apply(plan)
        state_before = await SQLiteLearningStore(owner._storage).list_states(
            tenant_id="tenant", identities=[identity], context_keys=["global"]
        )
    await owner.close()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths)
    owner = await EmbeddedMemory.open(path)
    assert (await SQLiteObservationStore(owner._storage).list(tenant_id="tenant"))[0] == current
    assert await SQLiteCheckpointStore(owner._storage).get(
        tenant_id="tenant", connector_id="c"
    ) == {"cursor": 3}
    if episode is not None:
        stored = await SQLiteEpisodeStore(owner._storage).list(tenant_id="tenant")
        assert stored[0].evidence == episode.evidence
    if schema_count >= 3:
        assert await owner.memory.list_semantic_memories(tenant_id=_TENANT) == semantic_before
        assert await owner.memory.list_semantic_revisions(tenant_id=_TENANT) == revision_before
    if schema_count >= 4:
        assert (
            await SQLiteLearningStore(owner._storage).list_states(
                tenant_id="tenant", identities=[identity], context_keys=["global"]
            )
            == state_before
        )
    await owner.close()
    _integrity(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode and umask semantics")
@pytest.mark.asyncio
async def test_host_umask_controls_new_database_and_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "permissions.db"
    program = """import asyncio, os, stat, sys
from cogkura import EmbeddedMemory
async def main():
    os.umask(0o077)
    owner = await EmbeddedMemory.open(sys.argv[1])
    for suffix in ("", "-wal", "-shm"):
        file = sys.argv[1] + suffix
        if os.path.exists(file):
            assert stat.S_IMODE(os.stat(file).st_mode) & 0o077 == 0, file
    await owner.close()
asyncio.run(main())
"""
    code, _, err = await _child(program, os.fspath(path))
    assert code == 0, err
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_semantic_inventory_batches_child_reads_without_changing_rows(tmp_path: Path) -> None:
    from cogkura.storage.sqlite_semantic import _hydrate, _load_memory, _load_revision

    owner = await EmbeddedMemory.open(
        tmp_path / "batch.db",
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    for index in range(8):
        await owner.memory.observe(
            ObservationInput(
                tenant_id="tenant",
                subject_id=f"user-{index}",
                source_namespace="src",
                source_record_id=str(index),
                source_version="1",
                content=f"User {index} uses database {index}.",
                observed_at=_T,
                metadata={
                    "conversation_id": str(index),
                    "semantic_facts": [
                        {
                            "predicate": "uses",
                            "object_value": f"db-{index}",
                            "authority": {"basis": "application_asserted"},
                        }
                    ],
                },
            )
        )
    await owner.memory.process(tenant_id="tenant", as_of=_T)

    def check() -> None:
        conn = owner._storage._conn()
        conn.execute("BEGIN")
        try:
            statements: list[str] = []
            conn.set_trace_callback(statements.append)
            state = _hydrate(conn, {"tenant"})
            conn.set_trace_callback(None)
            assert len(statements) == 6
            assert len(state._memories) == len(state._revisions) == 8
            for row in conn.execute(
                "SELECT * FROM semantic_memories WHERE tenant_id=?", ("tenant",)
            ):
                assert state._memories[("tenant", row["memory_key"])] == _load_memory(conn, row)
            for row in conn.execute(
                "SELECT * FROM semantic_revisions WHERE tenant_id=?", ("tenant",)
            ):
                assert state._revisions[("tenant", row["revision_key"])] == _load_revision(
                    conn, row
                )
            conn.commit()
        except BaseException:
            conn.set_trace_callback(None)
            conn.rollback()
            raise

    await owner._storage._run(check)
    await owner.close()


@pytest.mark.asyncio
async def test_contending_writer_succeeds_after_lock_release(tmp_path: Path) -> None:
    path = tmp_path / "contention.db"
    first, second = await asyncio.gather(
        EmbeddedMemory.open(path, busy_timeout=2), EmbeddedMemory.open(path, busy_timeout=2)
    )
    await first._storage._run(lambda: first._storage._conn().execute("BEGIN IMMEDIATE"))
    pending = asyncio.create_task(second.memory.observe(_observation("waiting")))
    await asyncio.sleep(0.05)
    assert not pending.done()
    assert (
        await first.memory._observation_store.get_by_source(
            tenant_id="tenant", source_namespace="src", source_record_id="waiting"
        )
        is None
    )
    await first._storage._run(lambda: first._storage._conn().rollback())
    assert await pending is not None
    await asyncio.gather(first.close(), second.close())
    _integrity(path)


@pytest.mark.asyncio
async def test_reproducible_workload_emits_ranked_comparisons(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_sqlite_workload.py"
    output = tmp_path / "workload.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        os.fspath(script),
        "--observations",
        "3",
        "--revisions",
        "2",
        "--recalls",
        "1",
        "--output",
        os.fspath(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()
    import json

    report = json.loads(output.read_text(encoding="utf-8"))
    assert all(report["logical_parity"].values())
    assert report["sqlite"]["counts"]["observation_revisions"] == 6
    assert report["sqlite"]["recall_sql_statements_total"] > 0
    assert (
        report["logical_comparisons"]["sqlite"]["ranked_recall_outputs"]
        == (report["logical_comparisons"]["in_memory"]["ranked_recall_outputs"])
    )
