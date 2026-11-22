"""Upgrade prior released PostgreSQL schemas in newly created disposable databases.

COGKURA_POSTGRES_UPGRADE_ADMIN_URL must point to a PostgreSQL admin database
whose role may create and drop databases. Only randomly named databases created
by this test are dropped. Published wheel checks require a local wheel directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from test_0202_released_sqlite_upgrade import (
    _FIXTURE,
    _INSTALLED_UPGRADE,
    _RELEASE_WHEEL_SHA256,
    _ROOT,
)

pytestmark = pytest.mark.postgres

_BOOTSTRAP = """
import os
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine
import cogkura
from cogkura.migrations import apply_migrations
from cogkura.storage.postgres import (PostgresObservationStore, PostgresCheckpointStore,
    PostgresEpisodeStore, PostgresSemanticMemoryStore, PostgresActivationStore,
    PostgresMemoryDynamicsStore, PostgresLearningStore, PostgresInhibitionStore,
    PostgresEntityRelationshipStore)

class PostgresOwner:
    @classmethod
    async def open(cls, path, **options):
        engine = create_async_engine(os.environ["COGKURA_TEST_UPGRADE_URL"])
        await apply_migrations(engine)
        semantic = PostgresSemanticMemoryStore(engine)
        memory = cogkura.Memory(
            observation_store=PostgresObservationStore(engine),
            checkpoint_store=PostgresCheckpointStore(engine),
            episode_store=PostgresEpisodeStore(engine), semantic_store=semantic,
            semantic_inspection_store=semantic, activation_store=PostgresActivationStore(engine),
            dynamics_store=PostgresMemoryDynamicsStore(engine),
            learning_store=PostgresLearningStore(engine),
            inhibition_store=PostgresInhibitionStore(engine),
            entity_relationship_store=PostgresEntityRelationshipStore(engine), **options)
        @asynccontextmanager
        async def owner():
            try:
                yield memory
            finally:
                await engine.dispose()
        return owner()

cogkura.EmbeddedMemory = PostgresOwner
"""


@pytest.mark.parametrize("version", ["0.19.5", "0.19.6"])
@pytest.mark.parametrize("fixture_kind", ["tagged-source", "published-wheel"])
async def test_upgrade_postgres_from_released_source(
    version: str, fixture_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cogkura import MemoryIdentity, MemoryKind, ObservationRevisionIdentity
    from cogkura import migrations as migrations_module
    from cogkura.migrations import apply_migrations
    from cogkura.models import EpisodeInput
    from cogkura.storage.postgres import (
        PostgresEpisodeStore,
        PostgresInhibitionStore,
        PostgresLearningStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    admin_url = os.environ.get("COGKURA_POSTGRES_UPGRADE_ADMIN_URL")
    if not admin_url:
        pytest.skip("COGKURA_POSTGRES_UPGRADE_ADMIN_URL for disposable databases is not set")
    old_root = tmp_path / "old"
    old_root.mkdir()
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
        package_dir = old_env
    else:
        if shutil.which("git") is None:
            pytest.skip("released-source upgrade fixture requires Git and release tags")
        tag = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/v{version}"],
            cwd=_ROOT,
            capture_output=True,
            check=False,
        )
        if tag.returncode:
            pytest.skip(f"release tag v{version} is unavailable")
        archive = subprocess.run(
            ["git", "archive", f"v{version}", "src/cogkura"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(old_root, filter="data")
        package_dir = old_root / "src"
    database = "ck_upgrade_" + uuid4().hex
    candidate_database = "ck_upgrade_" + uuid4().hex
    admin = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    created = False
    candidate_created = False
    engine = None
    try:
        async with admin.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{database}"'))
        created = True
        url = make_url(admin_url).set(database=database).render_as_string(hide_password=False)
        pythonpath = (
            sysconfig.get_paths()["purelib"]
            if fixture_kind == "published-wheel"
            else str(package_dir)
        )
        env = dict(os.environ, PYTHONPATH=pythonpath, COGKURA_TEST_UPGRADE_URL=url)
        completed = subprocess.run(
            [interpreter, "-c", _BOOTSTRAP + _FIXTURE, "unused"],
            cwd=old_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        before = json.loads(completed.stdout)
        assert before["version"] == version
        assert package_dir.resolve() in Path(before["module"]).resolve().parents
        candidate_wheel = os.environ.get("COGKURA_CANDIDATE_WHEEL")
        if fixture_kind == "published-wheel" and candidate_wheel:
            import cogkura

            async with admin.connect() as connection:
                await connection.execute(
                    text(f'CREATE DATABASE "{candidate_database}" TEMPLATE "{database}"')
                )
            candidate_created = True
            candidate_url = (
                make_url(admin_url)
                .set(database=candidate_database)
                .render_as_string(hide_password=False)
            )
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
            candidate_env = dict(env, COGKURA_TEST_UPGRADE_URL=candidate_url)
            for _ in range(2):
                subprocess.run(
                    [
                        interpreter,
                        "-c",
                        _BOOTSTRAP + "\nimport sys\nfrom pathlib import Path\n"
                        "assert Path(cogkura.__file__).resolve().is_relative_to(\n"
                        "    Path(sys.prefix).resolve())\n" + _INSTALLED_UPGRADE,
                        "unused",
                        str(expected_file),
                        cogkura.__version__,
                    ],
                    cwd=old_root,
                    env=candidate_env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
        engine = create_async_engine(url)
        tables = (
            "memory_activation_references",
            "memory_learning_events",
            "memory_learning_feedback",
            "memory_learning_state",
            "memory_learned_associations",
            "memory_dynamics",
            "memory_inhibition_traces",
            "entity_relationships",
        )
        async with engine.connect() as connection:
            before_rows = {
                table: (await connection.execute(text(f"SELECT * FROM cogkura.{table}"))).fetchall()
                for table in tables
            }
            versions = (
                (
                    await connection.execute(
                        text("SELECT version FROM cogkura.schema_migrations ORDER BY version")
                    )
                )
                .scalars()
                .all()
            )
        assert all(before_rows.values())
        assert "015_history_order" not in versions
        old_versions = versions
        migrations = migrations_module.migration_files()
        failing = tmp_path / "016_injected_failure.sql"
        failing.write_text("CREATE TABLE cogkura.temporary_test (id INTEGER);\nINVALID SQL;\n")
        monkeypatch.setattr(migrations_module, "migration_files", lambda: [*migrations, failing])
        from sqlalchemy.exc import SQLAlchemyError

        with pytest.raises(SQLAlchemyError):
            await apply_migrations(engine)
        async with engine.connect() as connection:
            assert (
                await connection.execute(
                    text("SELECT version FROM cogkura.schema_migrations ORDER BY version")
                )
            ).scalars().all() == old_versions
            assert (
                await connection.execute(text("SELECT to_regclass('cogkura.episode_history')"))
            ).scalar() is None
            assert (
                await connection.execute(text("SELECT to_regclass('cogkura.temporary_test')"))
            ).scalar() is None
        monkeypatch.setattr(migrations_module, "migration_files", lambda: migrations)
        await apply_migrations(engine)
        await apply_migrations(engine)
        async with engine.connect() as connection:
            for table, expected in before_rows.items():
                actual = (
                    await connection.execute(text(f"SELECT * FROM cogkura.{table}"))
                ).fetchall()
                assert sorted(map(repr, actual)) == sorted(map(repr, expected))
            versions = (
                (
                    await connection.execute(
                        text("SELECT version FROM cogkura.schema_migrations ORDER BY version")
                    )
                )
                .scalars()
                .all()
            )
        assert versions[-1] == "015_history_order"
        episodes = PostgresEpisodeStore(engine)
        observations = PostgresObservationStore(engine)
        semantics = PostgresSemanticMemoryStore(engine)
        transitions = await semantics.list_revisions(tenant_id="gamma")
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
        learning = PostgresLearningStore(engine)
        inhibition = PostgresInhibitionStore(engine)
        for tenant, expected in before["tenants"].items():
            stored_episodes = await episodes.list(tenant_id=tenant)
            assert sorted(e.id for e in stored_episodes) == expected["episodes"]
            stored_semantics = await semantics.list(tenant_id=tenant)
            assert sorted([s.id, s.object_value] for s in stored_semantics) == expected["semantics"]
            stored_observations = await observations.list(tenant_id=tenant)
            references = [
                ObservationRevisionIdentity(item.id, n)
                for item in stored_observations
                for n in (1, 2)
            ]
            revisions = await observations.get_revisions_many(
                tenant_id=tenant, references=references
            )
            assert (
                sorted(
                    [v.identity.observation_id, v.identity.observation_revision, v.source_version]
                    for v in revisions.values()
                )
                == expected["revisions"]
            )
            identities = [
                MemoryIdentity(MemoryKind(kind), key)
                for kind, key in expected["learned_identities"]
            ]
            states = await learning.list_states(
                tenant_id=tenant, identities=identities, context_keys=("global",)
            )
            assert len(states) == 2 and all(state.helpful_count == 1 for state in states)
            traces = await inhibition.list_for_memory(tenant_id=tenant, identity=identities[0])
            assert len(traces) == 1
            assert not (await episodes.historical_order_status(tenant_id=tenant)).complete
        original = (await episodes.list(tenant_id="alpha"))[0]
        at = datetime(2026, 1, 4, tzinfo=UTC)
        await episodes.upsert(
            EpisodeInput(
                tenant_id="alpha",
                subject_id=original.subject_id,
                memory_key=original.memory_key + "-post-upgrade",
                statement="A new post-upgrade episode.",
                started_at=original.started_at,
                ended_at=original.ended_at,
                confidence=original.confidence,
                importance=original.importance,
                evidence=original.evidence,
                entities=original.entities,
                metadata=original.metadata,
                encoding_context=original.encoding_context,
            ),
            as_of=at,
        )
        new_episodes = await episodes.list(tenant_id="alpha")
        inserted = next(e for e in new_episodes if e.memory_key.endswith("-post-upgrade"))
        snapshots = await episodes.historical_episodes(
            tenant_id="alpha", episode_ids={inserted.id}, as_of=at
        )
        assert snapshots[inserted.id] == inserted
        assert (
            sorted(e.id for e in await episodes.list(tenant_id="beta"))
            == before["tenants"]["beta"]["episodes"]
        )
    finally:
        if engine is not None:
            await engine.dispose()
        if candidate_created:
            async with admin.connect() as connection:
                await connection.execute(text(f'DROP DATABASE "{candidate_database}"'))
        if created:
            async with admin.connect() as connection:
                await connection.execute(text(f'DROP DATABASE "{database}"'))
        await admin.dispose()
