"""In-memory entity relationship store for tests."""

from __future__ import annotations

from collections.abc import Sequence

from cogkura.models import StoredEntityRelationship
from cogkura.storage.base import EntityRelationshipStore


class InMemoryEntityRelationshipStore(EntityRelationshipStore):
    """In-memory store for directed entity relationships."""

    def __init__(self) -> None:
        self._relationships: dict[tuple[str, str], StoredEntityRelationship] = {}

    def _key(self, relationship: StoredEntityRelationship) -> tuple[str, str]:
        return (relationship.tenant_id, relationship.relationship_id)

    async def upsert_many(self, relationships: Sequence[StoredEntityRelationship]) -> None:
        for relationship in relationships:
            self._relationships[self._key(relationship)] = relationship

    async def list(
        self,
        *,
        tenant_id: str,
        entity_id: str | None = None,
    ) -> list[StoredEntityRelationship]:
        items = [
            relationship
            for relationship in self._relationships.values()
            if relationship.tenant_id == tenant_id
        ]
        if entity_id is not None:
            items = [
                relationship
                for relationship in items
                if relationship.source_entity_id == entity_id
                or relationship.target_entity_id == entity_id
            ]
        items.sort(
            key=lambda item: (
                item.source_entity_id,
                item.relation_type.casefold(),
                item.target_entity_id,
                item.relationship_id,
            )
        )
        return items

    async def clear(self, *, tenant_id: str) -> None:
        keys = [key for key in self._relationships if key[0] == tenant_id]
        for key in keys:
            del self._relationships[key]
