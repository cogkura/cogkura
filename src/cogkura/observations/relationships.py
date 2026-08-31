"""Parse and build entity relationship inputs from observation metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from cogkura.exceptions import ValidationError
from cogkura.models import EntityRelationshipInput, StoredEntityRelationship


def entity_relationship_id(
    *,
    tenant_id: str,
    source_entity_id: str,
    relation_type: str,
    target_entity_id: str,
) -> str:
    canonical = "\x1f".join(
        (
            tenant_id,
            source_entity_id,
            relation_type.casefold(),
            target_entity_id,
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_entity_relationship_inputs(
    metadata: Mapping[str, Any],
) -> tuple[EntityRelationshipInput, ...]:
    raw = metadata.get("relationships")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValidationError("relationships metadata must be a sequence.")
    parsed: list[EntityRelationshipInput] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValidationError(f"relationships[{index}] must be a mapping.")
        source_entity_id = item.get("source_entity_id")
        relation_type = item.get("relation_type")
        target_entity_id = item.get("target_entity_id")
        if not isinstance(source_entity_id, str) or not source_entity_id.strip():
            raise ValidationError(
                f"relationships[{index}].source_entity_id must be a non-empty string."
            )
        if not isinstance(relation_type, str) or not relation_type.strip():
            raise ValidationError(
                f"relationships[{index}].relation_type must be a non-empty string."
            )
        if not isinstance(target_entity_id, str) or not target_entity_id.strip():
            raise ValidationError(
                f"relationships[{index}].target_entity_id must be a non-empty string."
            )
        provenance = item.get("provenance")
        if provenance is not None and (not isinstance(provenance, str) or not provenance.strip()):
            raise ValidationError(
                f"relationships[{index}].provenance must be a non-empty string when provided."
            )
        parsed.append(
            EntityRelationshipInput(
                source_entity_id=source_entity_id.strip(),
                relation_type=relation_type.strip(),
                target_entity_id=target_entity_id.strip(),
                provenance=provenance.strip() if isinstance(provenance, str) else None,
            )
        )
    return tuple(parsed)


def build_stored_entity_relationship(
    *,
    relationship: EntityRelationshipInput,
    tenant_id: str,
    source_namespace: str | None,
    source_record_id: str | None,
    observed_at: datetime,
) -> StoredEntityRelationship:
    if observed_at.tzinfo is None:
        raise ValidationError("observed_at must be timezone-aware.")
    return StoredEntityRelationship(
        relationship_id=entity_relationship_id(
            tenant_id=tenant_id,
            source_entity_id=relationship.source_entity_id,
            relation_type=relationship.relation_type,
            target_entity_id=relationship.target_entity_id,
        ),
        tenant_id=tenant_id,
        source_entity_id=relationship.source_entity_id,
        relation_type=relationship.relation_type,
        target_entity_id=relationship.target_entity_id,
        provenance=relationship.provenance,
        source_namespace=source_namespace,
        source_record_id=source_record_id,
        created_at=observed_at.astimezone(UTC),
    )
