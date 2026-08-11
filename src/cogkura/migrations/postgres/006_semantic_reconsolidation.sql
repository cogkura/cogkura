-- Cogkura semantic reconsolidation schema (version 006)

CREATE TABLE IF NOT EXISTS cogkura.semantic_claim_revisions (
    revision_key TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    memory_id UUID NOT NULL,
    memory_key TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    confidence DOUBLE PRECISION NOT NULL,
    importance DOUBLE PRECISION NOT NULL,
    support_count INTEGER NOT NULL,
    contradiction_count INTEGER NOT NULL,
    first_supported_at TIMESTAMPTZ NOT NULL,
    last_supported_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    FOREIGN KEY (memory_id)
        REFERENCES cogkura.memories(id)
        ON DELETE CASCADE,

    CONSTRAINT semantic_claim_revisions_status_check
        CHECK (status IN ('active', 'contested', 'superseded')),

    CONSTRAINT semantic_claim_revisions_revision_number_check
        CHECK (revision_number > 0),

    CONSTRAINT semantic_claim_revisions_support_count_check
        CHECK (support_count >= 0),

    CONSTRAINT semantic_claim_revisions_contradiction_count_check
        CHECK (contradiction_count >= 0),

    UNIQUE (memory_id, revision_number),
    UNIQUE (tenant_id, memory_key, revision_number)
);

CREATE INDEX IF NOT EXISTS semantic_claim_revisions_tenant_memory_idx
    ON cogkura.semantic_claim_revisions (tenant_id, memory_key);

CREATE INDEX IF NOT EXISTS semantic_claim_revisions_tenant_status_idx
    ON cogkura.semantic_claim_revisions (tenant_id, status);

ALTER TABLE cogkura.semantic_claims
    ADD COLUMN IF NOT EXISTS current_revision_key TEXT;

CREATE TABLE IF NOT EXISTS cogkura.semantic_revision_relations (
    tenant_id TEXT NOT NULL,
    left_revision_key TEXT NOT NULL,
    right_revision_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    effective_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant_id, left_revision_key, right_revision_key, relation),

    FOREIGN KEY (left_revision_key)
        REFERENCES cogkura.semantic_claim_revisions(revision_key)
        ON DELETE CASCADE,

    FOREIGN KEY (right_revision_key)
        REFERENCES cogkura.semantic_claim_revisions(revision_key)
        ON DELETE CASCADE,

    CONSTRAINT semantic_revision_relations_relation_check
        CHECK (relation IN ('supersedes', 'conflicts'))
);

ALTER TABLE cogkura.memory_derivations
    ADD COLUMN IF NOT EXISTS revision_key TEXT;

INSERT INTO cogkura.semantic_claim_revisions (
    revision_key,
    tenant_id,
    memory_id,
    memory_key,
    revision_number,
    status,
    valid_from,
    valid_until,
    confidence,
    importance,
    support_count,
    contradiction_count,
    first_supported_at,
    last_supported_at,
    created_at,
    updated_at
)
SELECT
    'legacy:' || m.id::text,
    m.tenant_id,
    m.id,
    m.memory_key,
    1,
    c.status,
    NULL,
    NULL,
    m.confidence,
    m.importance,
    c.support_count,
    c.contradiction_count,
    c.first_supported_at,
    c.last_supported_at,
    m.created_at,
    m.updated_at
FROM cogkura.memories AS m
JOIN cogkura.semantic_claims AS c
  ON c.memory_id = m.id
WHERE m.memory_type = 'semantic'
ON CONFLICT (revision_key) DO NOTHING;

UPDATE cogkura.semantic_claims AS c
SET current_revision_key = 'legacy:' || c.memory_id::text
WHERE c.current_revision_key IS NULL;

UPDATE cogkura.memories AS m
SET valid_from = NULL,
    valid_until = NULL
WHERE m.memory_type = 'semantic';

UPDATE cogkura.memory_derivations AS d
SET revision_key = 'legacy:' || d.target_memory_id::text
WHERE d.revision_key IS NULL
  AND EXISTS (
      SELECT 1
      FROM cogkura.memories AS m
      WHERE m.id = d.target_memory_id
        AND m.memory_type = 'semantic'
  );

ALTER TABLE cogkura.memory_derivations
    ALTER COLUMN revision_key SET NOT NULL;

ALTER TABLE cogkura.memory_derivations
    DROP CONSTRAINT IF EXISTS memory_derivations_pkey;

ALTER TABLE cogkura.memory_derivations
    ADD PRIMARY KEY (revision_key, source_memory_id, relation);

ALTER TABLE cogkura.memory_derivations
    ADD CONSTRAINT memory_derivations_revision_key_fkey
        FOREIGN KEY (revision_key)
        REFERENCES cogkura.semantic_claim_revisions(revision_key)
        ON DELETE CASCADE;
