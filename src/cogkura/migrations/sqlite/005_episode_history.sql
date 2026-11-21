CREATE TABLE episode_history (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    snapshot TEXT,
    FOREIGN KEY (tenant_id, episode_id) REFERENCES episodes(tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX episode_history_lookup_idx ON episode_history (tenant_id, episode_id, sequence);
