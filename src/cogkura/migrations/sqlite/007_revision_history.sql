CREATE TABLE semantic_revision_history (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    revision_key TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    snapshot TEXT,
    FOREIGN KEY (tenant_id, revision_key) REFERENCES semantic_revisions(tenant_id, revision_key) ON DELETE CASCADE
);
CREATE INDEX semantic_revision_history_lookup ON semantic_revision_history (tenant_id, revision_key, sequence);
CREATE TABLE semantic_relation_history (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    left_revision_key TEXT NOT NULL,
    right_revision_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    effective_at TEXT,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY (tenant_id, left_revision_key) REFERENCES semantic_revisions(tenant_id, revision_key) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, right_revision_key) REFERENCES semantic_revisions(tenant_id, revision_key) ON DELETE CASCADE
);
CREATE INDEX semantic_relation_history_lookup ON semantic_relation_history (tenant_id, sequence);
