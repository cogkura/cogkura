\c cognema_memory

CREATE SCHEMA IF NOT EXISTS cognema;

CREATE TABLE cognema.observations (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    subject_id TEXT,
    actor_id TEXT,
    source_type TEXT NOT NULL,
    source_namespace TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    source_version TEXT,
    event_type TEXT NOT NULL,
    content TEXT,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    source_created_at TIMESTAMPTZ,
    source_updated_at TIMESTAMPTZ,
    first_observed_at TIMESTAMPTZ NOT NULL,
    last_observed_at TIMESTAMPTZ NOT NULL,
    current_revision INTEGER NOT NULL DEFAULT 1,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    attention_score DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    retention_class TEXT NOT NULL DEFAULT 'full',
    policy_reasons JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_namespace, source_record_id),
    CONSTRAINT observations_attention_score_check
        CHECK (attention_score >= 0.0 AND attention_score <= 1.0)
);

CREATE INDEX observations_tenant_id_idx ON cognema.observations (tenant_id);
CREATE INDEX observations_tenant_subject_idx ON cognema.observations (tenant_id, subject_id);
CREATE INDEX observations_tenant_source_idx
    ON cognema.observations (tenant_id, source_namespace, source_record_id);
CREATE INDEX observations_tenant_event_type_idx ON cognema.observations (tenant_id, event_type);
CREATE INDEX observations_source_updated_at_idx ON cognema.observations (source_updated_at);

CREATE TABLE cognema.observation_revisions (
    id UUID PRIMARY KEY,
    observation_id UUID NOT NULL REFERENCES cognema.observations(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    source_version TEXT,
    content TEXT,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    change_type TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (observation_id, revision_number),
    CONSTRAINT observation_revisions_change_type_check
        CHECK (change_type IN ('created', 'updated', 'deleted', 'restored'))
);

CREATE TABLE cognema.connector_checkpoints (
    tenant_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    checkpoint JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, connector_id)
);

CREATE TABLE cognema.memories (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    subject_id TEXT,
    memory_type TEXT NOT NULL,
    memory_key TEXT,
    statement TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE cognema.memory_evidence (
    memory_id UUID NOT NULL REFERENCES cognema.memories(id) ON DELETE CASCADE,
    observation_id UUID NOT NULL REFERENCES cognema.observations(id) ON DELETE CASCADE,
    observation_revision INTEGER,
    sequence_number INTEGER,
    contribution_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, observation_id)
);

CREATE INDEX memories_tenant_type_key_idx
    ON cognema.memories (tenant_id, memory_type, memory_key)
    WHERE memory_key IS NOT NULL;
CREATE INDEX memories_tenant_type_active_idx
    ON cognema.memories (tenant_id, memory_type, is_active);
CREATE INDEX memories_tenant_subject_type_idx
    ON cognema.memories (tenant_id, subject_id, memory_type);
CREATE INDEX memories_valid_from_idx ON cognema.memories (valid_from);
CREATE INDEX memory_evidence_observation_idx
    ON cognema.memory_evidence (observation_id);
CREATE INDEX memory_evidence_memory_sequence_idx
    ON cognema.memory_evidence (memory_id, sequence_number);

CREATE TABLE cognema.memory_entities (
    memory_id UUID NOT NULL REFERENCES cognema.memories(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    entity_role TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, entity_id, entity_role)
);

CREATE INDEX memory_entities_entity_idx ON cognema.memory_entities (entity_id);

CREATE UNIQUE INDEX memories_tenant_id_memory_id_idx
    ON cognema.memories (tenant_id, id);

CREATE TABLE cognema.semantic_claims (
    tenant_id TEXT NOT NULL,
    memory_id UUID NOT NULL,
    slot_key TEXT NOT NULL,
    subject_entity_id TEXT,
    predicate TEXT NOT NULL,
    object_value TEXT NOT NULL,
    object_entity_id TEXT,
    polarity TEXT NOT NULL,
    cardinality TEXT NOT NULL,
    qualifiers JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    support_count INTEGER NOT NULL,
    contradiction_count INTEGER NOT NULL,
    first_supported_at TIMESTAMPTZ NOT NULL,
    last_supported_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (memory_id),
    FOREIGN KEY (memory_id) REFERENCES cognema.memories(id) ON DELETE CASCADE,
    CONSTRAINT semantic_claims_polarity_check CHECK (polarity IN ('affirm', 'deny')),
    CONSTRAINT semantic_claims_cardinality_check CHECK (cardinality IN ('one', 'many')),
    CONSTRAINT semantic_claims_status_check
        CHECK (status IN ('active', 'contested', 'superseded')),
    CONSTRAINT semantic_claims_support_count_check CHECK (support_count >= 0),
    CONSTRAINT semantic_claims_contradiction_count_check CHECK (contradiction_count >= 0)
);

CREATE INDEX semantic_claims_tenant_slot_idx
    ON cognema.semantic_claims (tenant_id, slot_key);
CREATE INDEX semantic_claims_tenant_predicate_idx
    ON cognema.semantic_claims (tenant_id, predicate);
CREATE INDEX semantic_claims_tenant_subject_idx
    ON cognema.semantic_claims (tenant_id, subject_entity_id);
CREATE INDEX semantic_claims_tenant_object_entity_idx
    ON cognema.semantic_claims (tenant_id, object_entity_id);

CREATE TABLE cognema.memory_derivations (
    tenant_id TEXT NOT NULL,
    target_memory_id UUID NOT NULL,
    source_memory_id UUID NOT NULL,
    relation TEXT NOT NULL,
    contribution_score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (target_memory_id, source_memory_id, relation),
    FOREIGN KEY (tenant_id, target_memory_id)
        REFERENCES cognema.memories(tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, source_memory_id)
        REFERENCES cognema.memories(tenant_id, id) ON DELETE CASCADE,
    CONSTRAINT memory_derivations_relation_check
        CHECK (relation IN ('supports', 'contradicts')),
    CONSTRAINT memory_derivations_score_check
        CHECK (contribution_score >= 0.0 AND contribution_score <= 1.0),
    CONSTRAINT memory_derivations_distinct_memory_check
        CHECK (target_memory_id <> source_memory_id)
);

CREATE INDEX memory_derivations_target_idx
    ON cognema.memory_derivations (target_memory_id);
CREATE INDEX memory_derivations_source_idx
    ON cognema.memory_derivations (source_memory_id);

CREATE TABLE cognema.schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO cognema.schema_migrations (version) VALUES ('001_initial');
INSERT INTO cognema.schema_migrations (version) VALUES ('002_episodic_memory');
INSERT INTO cognema.schema_migrations (version) VALUES ('003_semantic_consolidation');
