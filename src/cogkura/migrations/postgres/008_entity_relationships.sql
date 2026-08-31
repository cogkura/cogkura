CREATE TABLE IF NOT EXISTS cogkura.entity_relationships (
    tenant_id TEXT NOT NULL,
    relationship_id TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    relation_type_normalized TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    provenance TEXT,
    source_namespace TEXT,
    source_record_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (tenant_id, relationship_id),

    CONSTRAINT entity_relationships_distinct_entities_check
        CHECK (source_entity_id <> target_entity_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS entity_relationships_logical_identity_idx
    ON cogkura.entity_relationships (
        tenant_id,
        source_entity_id,
        relation_type_normalized,
        target_entity_id
    );

CREATE INDEX IF NOT EXISTS entity_relationships_tenant_source_idx
    ON cogkura.entity_relationships (tenant_id, source_entity_id);

CREATE INDEX IF NOT EXISTS entity_relationships_tenant_target_idx
    ON cogkura.entity_relationships (tenant_id, target_entity_id);
