CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    subject_id TEXT,
    memory_key TEXT NOT NULL,
    statement TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
    metadata TEXT NOT NULL,
    encoding_context TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (tenant_id, memory_key),
    UNIQUE (tenant_id, id)
);
CREATE INDEX episodes_tenant_subject_idx ON episodes (tenant_id, subject_id, is_active);
CREATE TABLE episode_evidence (
    tenant_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    observation_revision INTEGER NOT NULL,
    sequence_number INTEGER NOT NULL,
    contribution_score REAL NOT NULL,
    PRIMARY KEY (episode_id, observation_id),
    UNIQUE (episode_id, sequence_number),
    FOREIGN KEY (tenant_id, episode_id) REFERENCES episodes(tenant_id, id) ON DELETE CASCADE
);
CREATE TABLE episode_entities (
    tenant_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (episode_id, entity_id, role),
    FOREIGN KEY (tenant_id, episode_id) REFERENCES episodes(tenant_id, id) ON DELETE CASCADE
);
