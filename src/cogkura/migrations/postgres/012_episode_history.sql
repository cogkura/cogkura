CREATE TABLE cogkura.episode_history (
    sequence BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    episode_id UUID NOT NULL REFERENCES cogkura.memories(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL,
    snapshot JSONB
);
CREATE INDEX episode_history_lookup_idx ON cogkura.episode_history (tenant_id, episode_id, sequence);
