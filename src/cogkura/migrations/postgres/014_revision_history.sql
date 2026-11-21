CREATE TABLE cogkura.semantic_revision_history (
    sequence BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    revision_key TEXT NOT NULL REFERENCES cogkura.semantic_claim_revisions(revision_key) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL,
    snapshot JSONB
);
CREATE INDEX semantic_revision_history_lookup ON cogkura.semantic_revision_history (tenant_id, revision_key, sequence);
CREATE TABLE cogkura.semantic_relation_history (
    sequence BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    left_revision_key TEXT NOT NULL REFERENCES cogkura.semantic_claim_revisions(revision_key) ON DELETE CASCADE,
    right_revision_key TEXT NOT NULL REFERENCES cogkura.semantic_claim_revisions(revision_key) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    effective_at TIMESTAMPTZ,
    recorded_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX semantic_relation_history_lookup ON cogkura.semantic_relation_history (tenant_id, sequence);
