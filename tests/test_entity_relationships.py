"""Regression tests for 0.15.8 entity relationship models, store, and ingest."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura import Memory, ObservationInput
from cogkura.exceptions import ValidationError
from cogkura.migrations import apply_migrations
from cogkura.models import EntityRelationshipInput, StoredEntityRelationship
from cogkura.observations.models import IngestStatus
from cogkura.observations.relationships import (
    build_stored_entity_relationship,
    entity_relationship_id,
    parse_entity_relationship_inputs,
)
from cogkura.storage.in_memory_entity_relationship import InMemoryEntityRelationshipStore
from cogkura.storage.postgres import PostgresEntityRelationshipStore

_TENANT = "shop"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def test_entity_relationship_input_rejects_self_loop() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        EntityRelationshipInput(
            source_entity_id="a",
            relation_type="is_a",
            target_entity_id="a",
        )


def test_entity_relationship_id_is_case_insensitive_on_relation_type() -> None:
    lower = entity_relationship_id(
        tenant_id=_TENANT,
        source_entity_id="northpeak-alpine-shell",
        relation_type="is_a",
        target_entity_id="waterproof-shell",
    )
    upper = entity_relationship_id(
        tenant_id=_TENANT,
        source_entity_id="northpeak-alpine-shell",
        relation_type="IS_A",
        target_entity_id="waterproof-shell",
    )
    assert lower == upper


def test_parse_entity_relationship_inputs_rejects_malformed_entries() -> None:
    with pytest.raises(ValidationError, match="relationships\\[0\\]"):
        parse_entity_relationship_inputs({"relationships": ["not-a-mapping"]})


@pytest.mark.asyncio
async def test_in_memory_store_lists_incident_edges_in_both_directions() -> None:
    store = InMemoryEntityRelationshipStore()
    relationship = StoredEntityRelationship(
        relationship_id="rel-1",
        tenant_id=_TENANT,
        source_entity_id="northpeak-alpine-shell",
        relation_type="is_a",
        target_entity_id="waterproof-shell",
        provenance="catalog-import",
        source_namespace="catalog",
        source_record_id="row-1",
        created_at=_T,
    )
    await store.upsert_many([relationship])

    by_source = await store.list(tenant_id=_TENANT, entity_id="northpeak-alpine-shell")
    by_target = await store.list(tenant_id=_TENANT, entity_id="waterproof-shell")
    assert len(by_source) == 1
    assert len(by_target) == 1
    assert by_source[0].source_entity_id == "northpeak-alpine-shell"


@pytest.mark.asyncio
async def test_in_memory_store_idempotent_upsert() -> None:
    store = InMemoryEntityRelationshipStore()
    relationship = StoredEntityRelationship(
        relationship_id="rel-1",
        tenant_id=_TENANT,
        source_entity_id="a",
        relation_type="related_to",
        target_entity_id="b",
        provenance="first",
        source_namespace=None,
        source_record_id=None,
        created_at=_T,
    )
    updated = StoredEntityRelationship(
        relationship_id="rel-1",
        tenant_id=_TENANT,
        source_entity_id="a",
        relation_type="related_to",
        target_entity_id="b",
        provenance="second",
        source_namespace=None,
        source_record_id=None,
        created_at=_T,
    )
    await store.upsert_many([relationship])
    await store.upsert_many([updated])
    listed = await store.list(tenant_id=_TENANT)
    assert len(listed) == 1
    assert listed[0].provenance == "second"


@pytest.mark.asyncio
async def test_observe_upserts_relationships_even_when_unchanged() -> None:
    memory = Memory()
    observation = ObservationInput(
        tenant_id=_TENANT,
        subject_id="customer_42",
        actor_id="customer_42",
        source_namespace="catalog",
        source_record_id="taxonomy-1",
        event_type="import",
        content="Catalog taxonomy import",
        observed_at=_T,
        metadata={
            "relationships": [
                {
                    "source_entity_id": "northpeak-alpine-shell",
                    "relation_type": "is_a",
                    "target_entity_id": "waterproof-shell",
                    "provenance": "catalog-import",
                }
            ]
        },
    )
    first = await memory.observe(observation)
    second = await memory.observe(observation)
    assert first is IngestStatus.CREATED
    assert second is IngestStatus.UNCHANGED
    listed = await memory.list_entity_relationships(tenant_id=_TENANT)
    assert len(listed) == 1
    assert listed[0].target_entity_id == "waterproof-shell"


def test_build_stored_entity_relationship_requires_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        build_stored_entity_relationship(
            relationship=EntityRelationshipInput(
                source_entity_id="a",
                relation_type="depends_on",
                target_entity_id="b",
            ),
            tenant_id=_TENANT,
            source_namespace=None,
            source_record_id=None,
            observed_at=datetime(2026, 8, 1, 12, 0),
        )


@pytest.fixture
async def memory_engine(postgres_memory_url: str | None) -> AsyncIterator[AsyncEngine]:
    if postgres_memory_url is None:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    yield engine
    await engine.dispose()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_entity_relationship_store_parity(memory_engine: AsyncEngine) -> None:
    store = PostgresEntityRelationshipStore(memory_engine)
    relationship = StoredEntityRelationship(
        relationship_id=entity_relationship_id(
            tenant_id=_TENANT,
            source_entity_id="checkout-service",
            relation_type="depends_on",
            target_entity_id="payment-service",
        ),
        tenant_id=_TENANT,
        source_entity_id="checkout-service",
        relation_type="depends_on",
        target_entity_id="payment-service",
        provenance="service-map",
        source_namespace="topology",
        source_record_id="edge-1",
        created_at=_T,
    )
    await store.upsert_many([relationship])
    await store.upsert_many([relationship])
    listed = await store.list(tenant_id=_TENANT, entity_id="payment-service")
    assert len(listed) == 1
    assert listed[0].source_entity_id == "checkout-service"
    await store.clear(tenant_id=_TENANT)
    assert await store.list(tenant_id=_TENANT) == []
