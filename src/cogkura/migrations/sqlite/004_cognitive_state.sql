CREATE TABLE activation_references (
 id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, memory_kind TEXT NOT NULL, memory_key TEXT NOT NULL,
 reference_kind TEXT NOT NULL, referenced_at TEXT NOT NULL, request_id TEXT, weight INTEGER NOT NULL,
 metadata TEXT NOT NULL
);
CREATE UNIQUE INDEX activation_request ON activation_references
 (tenant_id,request_id,memory_kind,memory_key,reference_kind) WHERE request_id IS NOT NULL;
CREATE INDEX activation_lookup ON activation_references (tenant_id,memory_kind,memory_key,referenced_at);
CREATE TABLE memory_dynamics (
 tenant_id TEXT NOT NULL, memory_kind TEXT NOT NULL, memory_key TEXT NOT NULL, retention_state TEXT NOT NULL,
 last_base_level REAL NOT NULL, last_retention_score REAL NOT NULL, below_threshold_since TEXT,
 forgotten_at TEXT, evaluated_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY (tenant_id,memory_kind,memory_key)
);
CREATE TABLE learning_events (
 tenant_id TEXT NOT NULL, feedback_id TEXT NOT NULL, feedback_fingerprint TEXT NOT NULL, subject_id TEXT,
 context_key TEXT NOT NULL, occurred_at TEXT NOT NULL, metadata TEXT NOT NULL,
 PRIMARY KEY (tenant_id,feedback_id)
);
CREATE TABLE learning_feedback (
 tenant_id TEXT NOT NULL, feedback_id TEXT NOT NULL, memory_kind TEXT NOT NULL, memory_key TEXT NOT NULL,
 revision_key TEXT, outcome TEXT NOT NULL, metadata TEXT NOT NULL,
 PRIMARY KEY (tenant_id,feedback_id,memory_kind,memory_key),
 FOREIGN KEY (tenant_id,feedback_id) REFERENCES learning_events(tenant_id,feedback_id) ON DELETE CASCADE
);
CREATE TABLE learning_state (
 tenant_id TEXT NOT NULL, context_key TEXT NOT NULL, memory_kind TEXT NOT NULL, memory_key TEXT NOT NULL,
 helpful_count INTEGER NOT NULL, unhelpful_count INTEGER NOT NULL, incorrect_count INTEGER NOT NULL,
 first_feedback_at TEXT NOT NULL, last_feedback_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY (tenant_id,context_key,memory_kind,memory_key)
);
CREATE TABLE learned_associations (
 tenant_id TEXT NOT NULL, left_kind TEXT NOT NULL, left_key TEXT NOT NULL, right_kind TEXT NOT NULL,
 right_key TEXT NOT NULL, coactivation_count INTEGER NOT NULL, first_reinforced_at TEXT NOT NULL,
 last_reinforced_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 PRIMARY KEY (tenant_id,left_kind,left_key,right_kind,right_key)
);
CREATE TABLE inhibition_traces (
 id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, inhibited_kind TEXT NOT NULL, inhibited_key TEXT NOT NULL,
 selected_kind TEXT NOT NULL, selected_key TEXT NOT NULL, direction TEXT NOT NULL, scope_key TEXT NOT NULL,
 scope_json TEXT NOT NULL, competition_strength REAL NOT NULL, competitor_accessibility REAL NOT NULL,
 induction_pressure REAL NOT NULL, retrieval_evaluated_at TEXT NOT NULL, induced_at TEXT NOT NULL, request_id TEXT
);
CREATE UNIQUE INDEX inhibition_request ON inhibition_traces
 (tenant_id,request_id,selected_kind,selected_key,inhibited_kind,inhibited_key,scope_key) WHERE request_id IS NOT NULL;
CREATE INDEX inhibition_lookup ON inhibition_traces (tenant_id,inhibited_kind,inhibited_key,induced_at);
CREATE TABLE entity_relationships (
 tenant_id TEXT NOT NULL, relationship_id TEXT NOT NULL, source_entity_id TEXT NOT NULL,
 relation_type TEXT NOT NULL, target_entity_id TEXT NOT NULL, provenance TEXT, source_namespace TEXT,
 source_record_id TEXT, created_at TEXT NOT NULL, PRIMARY KEY (tenant_id,relationship_id)
);
CREATE INDEX entity_relationship_lookup ON entity_relationships (tenant_id,source_entity_id,target_entity_id);
