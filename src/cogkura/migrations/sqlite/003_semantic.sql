CREATE TABLE semantic_memories (
    id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, subject_id TEXT, memory_key TEXT NOT NULL,
    slot_key TEXT NOT NULL, revision_key TEXT NOT NULL, revision_number INTEGER NOT NULL,
    statement TEXT NOT NULL, subject_entity_id TEXT, predicate TEXT NOT NULL, object_value TEXT NOT NULL,
    object_entity_id TEXT, polarity TEXT NOT NULL, cardinality TEXT NOT NULL, qualifiers TEXT NOT NULL,
    confidence REAL NOT NULL, importance REAL NOT NULL, status TEXT NOT NULL, support_count INTEGER NOT NULL,
    contradiction_count INTEGER NOT NULL, first_supported_at TEXT NOT NULL, last_supported_at TEXT NOT NULL,
    valid_from TEXT, valid_until TEXT, is_active INTEGER NOT NULL, metadata TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE (tenant_id, memory_key), UNIQUE (tenant_id, id)
);
CREATE INDEX semantic_memories_tenant_slot ON semantic_memories (tenant_id, slot_key);
CREATE TABLE semantic_revisions (
    tenant_id TEXT NOT NULL, revision_key TEXT NOT NULL, memory_key TEXT NOT NULL,
    revision_number INTEGER NOT NULL, status TEXT NOT NULL, valid_from TEXT, valid_until TEXT,
    confidence REAL NOT NULL, importance REAL NOT NULL, support_count INTEGER NOT NULL,
    contradiction_count INTEGER NOT NULL, first_supported_at TEXT NOT NULL, last_supported_at TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, snapshot TEXT NOT NULL,
    PRIMARY KEY (tenant_id, revision_key), UNIQUE (tenant_id, memory_key, revision_number),
    FOREIGN KEY (tenant_id, memory_key) REFERENCES semantic_memories(tenant_id, memory_key) ON DELETE CASCADE
);
CREATE TABLE semantic_derivations (
    tenant_id TEXT NOT NULL, memory_key TEXT NOT NULL, revision_key TEXT NOT NULL,
    episode_id TEXT NOT NULL, relation TEXT NOT NULL, contribution_score REAL NOT NULL,
    PRIMARY KEY (tenant_id, memory_key, revision_key, episode_id, relation),
    FOREIGN KEY (tenant_id, memory_key) REFERENCES semantic_memories(tenant_id, memory_key) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, episode_id) REFERENCES episodes(tenant_id, id) ON DELETE CASCADE
);
CREATE TABLE semantic_evidence (
    tenant_id TEXT NOT NULL, memory_key TEXT NOT NULL, observation_id TEXT NOT NULL,
    observation_revision INTEGER NOT NULL, sequence_number INTEGER NOT NULL, contribution_score REAL NOT NULL,
    PRIMARY KEY (tenant_id, memory_key, observation_id, observation_revision),
    FOREIGN KEY (tenant_id, memory_key) REFERENCES semantic_memories(tenant_id, memory_key) ON DELETE CASCADE,
    FOREIGN KEY (observation_id) REFERENCES observations(id) ON DELETE CASCADE
);
CREATE TABLE semantic_entities (
    tenant_id TEXT NOT NULL, memory_key TEXT NOT NULL, entity_id TEXT NOT NULL, role TEXT NOT NULL,
    PRIMARY KEY (tenant_id, memory_key, entity_id, role),
    FOREIGN KEY (tenant_id, memory_key) REFERENCES semantic_memories(tenant_id, memory_key) ON DELETE CASCADE
);
CREATE TABLE semantic_relations (
    tenant_id TEXT NOT NULL, left_revision_key TEXT NOT NULL, right_revision_key TEXT NOT NULL,
    relation TEXT NOT NULL, effective_at TEXT,
    PRIMARY KEY (tenant_id, left_revision_key, right_revision_key, relation),
    FOREIGN KEY (tenant_id, left_revision_key) REFERENCES semantic_revisions(tenant_id, revision_key) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, right_revision_key) REFERENCES semantic_revisions(tenant_id, revision_key) ON DELETE CASCADE
);
