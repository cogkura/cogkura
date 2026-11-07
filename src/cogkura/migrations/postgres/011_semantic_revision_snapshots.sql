-- Revision snapshots for authoritative corrections (version 011)

ALTER TABLE cogkura.semantic_claim_revisions
    ADD COLUMN IF NOT EXISTS snapshot JSONB NOT NULL DEFAULT '{}';

ALTER TABLE cogkura.semantic_claim_revisions
    DROP CONSTRAINT IF EXISTS semantic_claim_revisions_status_check;

ALTER TABLE cogkura.semantic_claim_revisions
    ADD CONSTRAINT semantic_claim_revisions_status_check
    CHECK (status IN ('active', 'contested', 'superseded', 'retracted'));

ALTER TABLE cogkura.semantic_claims
    DROP CONSTRAINT IF EXISTS semantic_claims_status_check;

ALTER TABLE cogkura.semantic_claims
    ADD CONSTRAINT semantic_claims_status_check
    CHECK (status IN ('active', 'contested', 'superseded', 'retracted'));

ALTER TABLE cogkura.semantic_revision_relations
    DROP CONSTRAINT IF EXISTS semantic_revision_relations_relation_check;

ALTER TABLE cogkura.semantic_revision_relations
    ADD CONSTRAINT semantic_revision_relations_relation_check
    CHECK (relation IN ('supersedes', 'conflicts', 'retracts'));
