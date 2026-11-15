CREATE TABLE observations (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, subject_id TEXT, actor_id TEXT,
    source_type TEXT NOT NULL, source_namespace TEXT NOT NULL, source_record_id TEXT NOT NULL,
    source_version TEXT, event_type TEXT NOT NULL, content TEXT, content_hash TEXT NOT NULL,
    metadata TEXT NOT NULL, encoding_context TEXT NOT NULL,
    source_created_at TEXT, source_updated_at TEXT, first_observed_at TEXT NOT NULL,
    last_observed_at TEXT NOT NULL, current_revision INTEGER NOT NULL,
    is_deleted INTEGER NOT NULL CHECK (is_deleted IN (0, 1)),
    attention_score REAL NOT NULL, retention_class TEXT NOT NULL, policy_reasons TEXT NOT NULL,
    UNIQUE (tenant_id, source_namespace, source_record_id)
);
CREATE INDEX observations_tenant_subject_idx ON observations (tenant_id, subject_id);
CREATE TABLE observation_revisions (
    id TEXT PRIMARY KEY, observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL, source_version TEXT, content TEXT, content_hash TEXT NOT NULL,
    metadata TEXT NOT NULL, encoding_context TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK (change_type IN ('created', 'updated', 'deleted', 'restored')),
    observed_at TEXT NOT NULL, UNIQUE (observation_id, revision_number)
);
CREATE TABLE connector_checkpoints (
    tenant_id TEXT NOT NULL, connector_id TEXT NOT NULL, checkpoint TEXT NOT NULL,
    PRIMARY KEY (tenant_id, connector_id)
);
