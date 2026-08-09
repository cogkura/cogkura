-- Cogkura observation storage schema (version 001)

CREATE SCHEMA IF NOT EXISTS cogkura;

CREATE TABLE IF NOT EXISTS cogkura.observations (
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_namespace, source_record_id)
);

CREATE INDEX IF NOT EXISTS observations_tenant_id_idx
    ON cogkura.observations (tenant_id);

CREATE INDEX IF NOT EXISTS observations_tenant_subject_idx
    ON cogkura.observations (tenant_id, subject_id);

CREATE INDEX IF NOT EXISTS observations_tenant_source_idx
    ON cogkura.observations (tenant_id, source_namespace, source_record_id);

CREATE INDEX IF NOT EXISTS observations_tenant_event_type_idx
    ON cogkura.observations (tenant_id, event_type);

CREATE INDEX IF NOT EXISTS observations_source_updated_at_idx
    ON cogkura.observations (source_updated_at);

CREATE TABLE IF NOT EXISTS cogkura.observation_revisions (
    id UUID PRIMARY KEY,
    observation_id UUID NOT NULL
        REFERENCES cogkura.observations(id) ON DELETE CASCADE,
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

CREATE TABLE IF NOT EXISTS cogkura.connector_checkpoints (
    tenant_id TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    checkpoint JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, connector_id)
);

CREATE TABLE IF NOT EXISTS cogkura.memories (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    subject_id TEXT,
    memory_type TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS cogkura.memory_evidence (
    memory_id UUID NOT NULL
        REFERENCES cogkura.memories(id) ON DELETE CASCADE,
    observation_id UUID NOT NULL
        REFERENCES cogkura.observations(id) ON DELETE CASCADE,
    contribution_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, observation_id)
);

CREATE TABLE IF NOT EXISTS cogkura.schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
