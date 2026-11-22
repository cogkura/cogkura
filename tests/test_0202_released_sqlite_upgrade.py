"""Exercise SQLite upgrades from prior tagged sources and optional release wheels.

The wheel checks run only when COGKURA_RELEASE_WHEEL_DIRECTORY names a local
directory containing the two published universal wheels. They make no network
requests. Tagged sources and installed wheels use separate fresh interpreters.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_WHEEL_SHA256 = {
    "0.19.5": "29ec696acba22b9752948d8f394551066ad1038989e0a30244162a6ed7460fe0",
    "0.19.6": "862d1a08b3d939bc1eea81d883c03315a30509c6779e12abec6f5fb85d8252d4",
}
_FIXTURE = """
import asyncio, json, sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
import cogkura
from cogkura import (ActivationConfig, BehavioralStructuralAnchor, CompetitionDirection,
    EmbeddedMemory, InhibitionScopeSignature, InhibitoryTrace, LearningFeedback,
    LearningOutcome, MemoryFeedback, MemoryIdentity, ObservationInput, ObservationRevisionIdentity,
    ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig)
from cogkura.models import MemoryRetentionState, StoredMemoryDynamics
from uuid import uuid4

async def main():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    learned_identities = {}
    async with await EmbeddedMemory.open(Path(sys.argv[1]),
            activation_config=ActivationConfig(retrieval_threshold=-10),
            semantic_consolidator=ComplementaryLearningSemanticConsolidator(
                allow_authoritative_single_episode=True),
            semantic_correction_config=SemanticCorrectionConfig(enabled=True)) as memory:
        for tenant in ("alpha", "beta"):
            for n in range(2):
                await memory.observe(ObservationInput(
                    tenant_id=tenant, subject_id="service", source_namespace="upgrade",
                    source_record_id=f"{tenant}-{n}", source_version="1",
                    observed_at=now, content="Deployment policy is pipeline.",
                    metadata={"conversation_id": f"session-{n}", "relationships": [{
                        "source_entity_id": "service", "relation_type": "depends_on",
                        "target_entity_id": "database"}], "semantic_facts": [{
                        "predicate": "deployment_policy", "object_value": "pipeline",
                        "cardinality": "one"}]},
                ))
                if n == 0:
                    await memory.observe(ObservationInput(
                        tenant_id=tenant, subject_id="service", source_namespace="upgrade",
                        source_record_id=f"{tenant}-{n}", source_version="2",
                        observed_at=now + timedelta(seconds=1),
                        content="Deployment policy is pipeline, revised.",
                        metadata={"conversation_id": f"session-{n}", "relationships": [{
                            "source_entity_id": "service", "relation_type": "depends_on",
                            "target_entity_id": "database"}], "semantic_facts": [{
                            "predicate": "deployment_policy", "object_value": "pipeline",
                            "cardinality": "one"}]},
                    ))
            await memory.process(tenant_id=tenant, as_of=now + timedelta(days=1))
            results = await memory.recall("deployment policy pipeline", tenant_id=tenant,
                                          as_of=now + timedelta(days=1), limit=10)
            assert len(results) >= 2
            await memory.record_access(results, tenant_id=tenant,
                                       referenced_at=now + timedelta(days=1), request_id="use")
            identities = [MemoryIdentity(r.memory_kind, r.memory.memory_key) for r in results]
            learned_identities[tenant] = [
                (i.memory_kind.value, i.memory_key) for i in identities[:2]]
            await memory.learn(LearningFeedback(
                tenant_id=tenant, feedback_id="helpful", occurred_at=now + timedelta(days=1),
                items=(MemoryFeedback(identity=identities[0], outcome=LearningOutcome.HELPFUL),
                       MemoryFeedback(identity=identities[1], outcome=LearningOutcome.HELPFUL)),
            ))
            await memory._dynamics_store.upsert_many([StoredMemoryDynamics(
                tenant_id=tenant, memory_kind=identities[0].memory_kind,
                memory_key=identities[0].memory_key, retention_state=MemoryRetentionState.FORGOTTEN,
                last_base_level=-2, last_retention_score=0.1, below_threshold_since=now,
                forgotten_at=now, evaluated_at=now, updated_at=now,
            )])
            await memory._inhibition_store.append_traces([InhibitoryTrace(
                id=str(uuid4()), tenant_id=tenant, inhibited_identity=identities[0],
                selected_identity=identities[1], direction=CompetitionDirection.CO_TEMPORAL,
                scope=InhibitionScopeSignature(
                    structural_anchor=BehavioralStructuralAnchor.SUBJECT_PREDICATE,
                    subject_id="service", predicate="deployment_policy"),
                competition_strength=0.8, competitor_accessibility=0.7,
                induction_pressure=0.56, retrieval_evaluated_at=now + timedelta(days=1),
                induced_at=now + timedelta(days=1), request_id="inhibition",
            )])
        result = {}
        for tenant in ("alpha", "beta"):
            episodes = await memory.list_episodes(tenant_id=tenant)
            semantics = await memory.list_semantic_memories(tenant_id=tenant)
            observations = await memory._observation_store.list(tenant_id=tenant)
            references = [ObservationRevisionIdentity(item.id, n) for item in observations
                          for n in (1, 2)]
            revisions = await memory._observation_store.get_revisions_many(
                tenant_id=tenant, references=references)
            result[tenant] = {"episodes": sorted(e.id for e in episodes),
                              "semantics": sorted((s.id, s.object_value) for s in semantics),
                              "learned_identities": learned_identities[tenant],
                              "revisions": sorted((v.identity.observation_id,
                                                   v.identity.observation_revision,
                                                   v.source_version) for v in revisions.values())}
        fact = {"predicate": "storage_engine", "object_value": "postgresql",
                "cardinality": "one", "authority": {"basis": "application_asserted"}}
        async def event(record, day, metadata):
            await memory.observe(ObservationInput(
                tenant_id="gamma", subject_id="service", source_namespace="upgrade-transition",
                source_record_id=record, observed_at=now + timedelta(days=day), content=record,
                metadata={"conversation_id": record, **metadata}))
            return await memory.process(tenant_id="gamma", as_of=now + timedelta(days=day))
        await event("original", 0, {"semantic_facts": [fact]})
        original = (await memory.list_semantic_revisions(tenant_id="gamma"))[0]
        correction = {**fact, "object_value": "sqlite", "correction": {
            "replaces_revision_key": original.revision_key,
            "effective_at": (now + timedelta(days=1)).isoformat()}}
        changed = await event("correction", 1, {"semantic_facts": [correction]})
        assert changed.semantics.authoritative_corrections_applied == 1
        successor = next(r for r in await memory.list_semantic_revisions(tenant_id="gamma")
                         if r.revision_key != original.revision_key)
        withdrawn = await event("withdrawal", 2, {"semantic_withdrawals": [{
            "target_revision_key": successor.revision_key,
            "effective_at": (now + timedelta(days=2)).isoformat(),
            "authority": {"basis": "application_asserted"}}]})
        assert withdrawn.semantics.authoritative_withdrawals_applied == 1
        transitions = sorted((r.revision_key, r.status.value,
                              r.valid_from.isoformat() if r.valid_from else None,
                              r.valid_until.isoformat() if r.valid_until else None,
                              dict(r.snapshot))
                             for r in await memory.list_semantic_revisions(tenant_id="gamma"))
        assert len(transitions) == 3
    print(json.dumps({"version": cogkura.__version__, "module": cogkura.__file__,
                      "tenants": result, "transitions": transitions}, sort_keys=True))
asyncio.run(main())
"""
_INSTALLED_UPGRADE = """
import asyncio, json, sys
from pathlib import Path
import cogkura
from cogkura import EmbeddedMemory, MemoryIdentity, MemoryKind, ObservationRevisionIdentity

async def main():
    expected = json.loads(Path(sys.argv[2]).read_text())
    assert cogkura.__version__ == sys.argv[3]
    async with await EmbeddedMemory.open(Path(sys.argv[1])) as memory:
        for tenant, state in expected["tenants"].items():
            episodes = await memory.list_episodes(tenant_id=tenant)
            semantics = await memory.list_semantic_memories(tenant_id=tenant)
            assert sorted(e.id for e in episodes) == state["episodes"]
            assert sorted([s.id, s.object_value] for s in semantics) == state["semantics"]
            observations = await memory._observation_store.list(tenant_id=tenant)
            refs = [ObservationRevisionIdentity(item.id, n)
                    for item in observations for n in (1, 2)]
            revisions = await memory._observation_store.get_revisions_many(
                tenant_id=tenant, references=refs)
            assert sorted([v.identity.observation_id, v.identity.observation_revision,
                           v.source_version] for v in revisions.values()) == state["revisions"]
            identities = [MemoryIdentity(MemoryKind(kind), key)
                          for kind, key in state["learned_identities"]]
            learning = await memory.list_learning_state(tenant_id=tenant, identities=identities)
            assert len(learning) == 2 and all(item.helpful_count == 1 for item in learning)
            traces = await memory.list_inhibition_traces(
                tenant_id=tenant, memory_kind=identities[0].memory_kind,
                memory_key=identities[0].memory_key)
            assert len(traces) == 1
            status = await memory._episode_store.historical_order_status(tenant_id=tenant)
            assert not status.complete
        transitions = await memory.list_semantic_revisions(tenant_id="gamma")
        assert sorted([r.revision_key, r.status.value,
                       r.valid_from.isoformat() if r.valid_from else None,
                       r.valid_until.isoformat() if r.valid_until else None,
                       dict(r.snapshot)]
                      for r in transitions) == expected["transitions"]
asyncio.run(main())
"""


@pytest.mark.parametrize("version", ["0.19.5", "0.19.6"])
@pytest.mark.parametrize("fixture_kind", ["tagged-source", "published-wheel"])
async def test_upgrade_sqlite_from_released_source(
    version: str, fixture_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cogkura import (
        EmbeddedMemory,
        MemoryIdentity,
        MemoryKind,
        ObservationInput,
        ObservationRevisionIdentity,
    )
    from cogkura.exceptions import StorageError
    from cogkura.storage import sqlite as sqlite_module
    from cogkura.storage.sqlite import SQLiteStorage

    old_root = tmp_path / "old"
    old_root.mkdir()
    path = tmp_path / "upgrade.db"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    interpreter = sys.executable
    if fixture_kind == "published-wheel":
        wheel_dir = os.environ.get("COGKURA_RELEASE_WHEEL_DIRECTORY")
        if not wheel_dir:
            pytest.skip("COGKURA_RELEASE_WHEEL_DIRECTORY is not set")
        wheel = Path(wheel_dir) / f"cogkura-{version}-py3-none-any.whl"
        if not wheel.is_file():
            pytest.skip(f"published wheel for {version} is unavailable")
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == _RELEASE_WHEEL_SHA256[version]
        old_env = old_root / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(old_env)], check=True, capture_output=True
        )
        interpreter = str(old_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        subprocess.run(
            [interpreter, "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        if shutil.which("git") is None:
            pytest.skip("released-source upgrade fixture requires Git and the release tags")
        tag = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/v{version}"],
            cwd=_ROOT,
            capture_output=True,
            check=False,
        )
        if tag.returncode:
            pytest.skip(f"release tag v{version} is unavailable in this checkout")
        archive = subprocess.run(
            ["git", "archive", f"v{version}", "src/cogkura"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(old_root, filter="data")
        env["PYTHONPATH"] = str(old_root / "src")
    completed = subprocess.run(
        [interpreter, "-c", _FIXTURE, str(path)],
        cwd=old_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    before = json.loads(completed.stdout)
    assert before["version"] == version
    assert old_root.resolve() in Path(before["module"]).resolve().parents
    for state in before["tenants"].values():
        assert len(state["episodes"]) == 2
        assert len(state["semantics"]) == 1
        assert len(state["revisions"]) == 3

    candidate_wheel = os.environ.get("COGKURA_CANDIDATE_WHEEL")
    if fixture_kind == "published-wheel" and candidate_wheel:
        import cogkura

        copied_database = tmp_path / "installed-upgrade.db"
        with sqlite3.connect(path) as source, sqlite3.connect(copied_database) as destination:
            source.backup(destination)
        expected_file = tmp_path / "expected.json"
        expected_file.write_text(
            json.dumps({"tenants": before["tenants"], "transitions": before["transitions"]}),
            encoding="utf-8",
        )
        subprocess.run(
            [interpreter, "-m", "pip", "install", "--no-deps", "--no-index", candidate_wheel],
            cwd=old_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        for _ in range(2):
            subprocess.run(
                [
                    interpreter,
                    "-I",
                    "-c",
                    _INSTALLED_UPGRADE,
                    str(copied_database),
                    str(expected_file),
                    cogkura.__version__,
                ],
                cwd=old_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

    tables = (
        "activation_references",
        "learning_events",
        "learning_feedback",
        "learning_state",
        "learned_associations",
        "memory_dynamics",
        "inhibition_traces",
        "entity_relationships",
    )
    with sqlite3.connect(path) as connection:
        old_versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        cognitive_state = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in tables
        }
    assert all(cognitive_state.values())
    migrations = sqlite_module._migrations()
    failing = tmp_path / "009_injected_failure.sql"
    failing.write_text("CREATE TABLE temporary_test (id INTEGER);\nINVALID SQL;\n")
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: [*migrations, failing])
    with pytest.raises(StorageError):
        await SQLiteStorage.open(path)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
            == old_versions
        )
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name IN ('temporary_test', 'episode_history')"
        ).fetchall()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: migrations)
    for _ in range(2):
        async with await EmbeddedMemory.open(path) as memory:
            transitions = await memory.list_semantic_revisions(tenant_id="gamma")
            assert (
                sorted(
                    [
                        r.revision_key,
                        r.status.value,
                        r.valid_from.isoformat() if r.valid_from else None,
                        r.valid_until.isoformat() if r.valid_until else None,
                        dict(r.snapshot),
                    ]
                    for r in transitions
                )
                == before["transitions"]
            )
            for tenant, expected in before["tenants"].items():
                episodes = await memory.list_episodes(tenant_id=tenant)
                semantics = await memory.list_semantic_memories(tenant_id=tenant)
                assert sorted(e.id for e in episodes) == expected["episodes"]
                assert sorted([s.id, s.object_value] for s in semantics) == expected["semantics"]
                observations = await memory._observation_store.list(tenant_id=tenant)
                references = [
                    ObservationRevisionIdentity(item.id, n) for item in observations for n in (1, 2)
                ]
                revisions = await memory._observation_store.get_revisions_many(
                    tenant_id=tenant, references=references
                )
                assert (
                    sorted(
                        [
                            v.identity.observation_id,
                            v.identity.observation_revision,
                            v.source_version,
                        ]
                        for v in revisions.values()
                    )
                    == expected["revisions"]
                )
                identities = [
                    MemoryIdentity(MemoryKind(kind), key)
                    for kind, key in expected["learned_identities"]
                ]
                states = await memory.list_learning_state(tenant_id=tenant, identities=identities)
                assert {state.memory_key for state in states} == {
                    item.memory_key for item in identities
                }
                assert all(state.helpful_count == 1 for state in states)
                inhibition = await memory.list_inhibition_traces(
                    tenant_id=tenant,
                    memory_kind=identities[0].memory_kind,
                    memory_key=identities[0].memory_key,
                )
                assert len(inhibition) == 1
                status = await memory._episode_store.historical_order_status(tenant_id=tenant)
                assert not status.complete
        with sqlite3.connect(path) as connection:
            for table, expected in cognitive_state.items():
                assert (
                    connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                    == expected
                )

    now = datetime(2026, 1, 3, tzinfo=UTC)
    async with await EmbeddedMemory.open(path) as memory:
        await memory.observe(
            ObservationInput(
                tenant_id="alpha",
                subject_id="service",
                source_namespace="upgrade",
                source_record_id="post-upgrade",
                observed_at=now,
                content="A new post-upgrade deployment was recorded.",
                metadata={"conversation_id": "post-upgrade"},
            )
        )
        await memory.process(tenant_id="alpha", as_of=now + timedelta(days=1))
        beta = await memory.list_episodes(tenant_id="beta")
        assert sorted(e.id for e in beta) == before["tenants"]["beta"]["episodes"]
    with sqlite3.connect(path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM episode_history WHERE tenant_id=? AND snapshot IS NOT NULL",
            ("alpha",),
        ).fetchone()[0]
    assert count > 0
