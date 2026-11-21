CREATE TABLE cogkura.semantic_history (
    sequence BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    memory_id UUID NOT NULL REFERENCES cogkura.memories(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL,
    snapshot JSONB
);
CREATE INDEX semantic_history_lookup_idx ON cogkura.semantic_history (tenant_id, memory_id, sequence);
