CREATE TABLE semantic_history (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    snapshot TEXT,
    FOREIGN KEY (tenant_id, memory_id) REFERENCES semantic_memories(tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX semantic_history_lookup_idx ON semantic_history (tenant_id, memory_id, sequence);
