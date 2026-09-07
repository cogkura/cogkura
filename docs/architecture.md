# Architecture

## Design direction

Cogkura is a thin cognitive layer between application data and AI reasoning.

Applications keep their own persistence and model infrastructure. Customer data stays in customer-owned schemas. Cogkura owns observations, revisions, checkpoints, and derived memories.

The public API should remain stable even as internals evolve.

## 0.15 dependable-recall pipeline

```text
Observation
    ↓
episodic encoding + semantic consolidation (+ reconsolidation)
    ↓
activation + forgetting (ACT-R base-level, dynamics)
    ↓
semantic relevance + competition
    ↓
contextual association + structured relationships
    ↓
recall candidates (threshold + soft admission)
    ↓
working-memory chunks (optional grouping)
    ↓
coverage-aware bounded selection
    ↓
deterministic render (MemoryContext)
```

Responsibility boundary:

```text
application / future cogkura-ingest
    ↓ source interpretation, entities, relationships, semantic proposals
Cogkura Core
    ↓ memory lifecycle + retrieval
LLM / application reasoning
```

Core does not parse arbitrary external data. Configuration and contracts: [`configuration.md`](configuration.md).

## Public API

| API | Purpose |
|-----|---------|
| `observe` / `ingest` | Write path for observations |
| `process` | Encode episodes and consolidate semantics at one `as_of` |
| `recall` / `inspect_recall` | Declarative activation ranking and diagnostics |
| `select_working_memory` / `prepare_context` | Bounded context preparation |
| `record_context_use` | Record consumption of prepared context |
| `apply_forgetting` / `record_access` / `learn` | Dynamics, reinforcement, feedback |
| `list_semantic_memories` / `list_semantic_revisions` | Semantic authority and history |

Typed models: `ObservationInput`, `ObservationContext`, `MemoryContextSignature`, `MemoryContext`, `RecallResult`, `WorkingMemorySnapshot`, `RecallInspectionResult`.

## Layers

### Observation pipeline

1. Application mapper converts source records to `ObservationInput`
2. Policy evaluates attention and acceptance
3. Retention mode transforms content before storage
4. Observation store persists with revision history
5. Entity relationships from `metadata["relationships"]` persist on `observe()`
6. Checkpoint store advances only after successful ingest batches

### Cognitive algorithms

[`src/cogkura/algorithms/`](../src/cogkura/algorithms/):

- `episodic.py` — deterministic episode encoding
- `semantic.py` — semantic consolidation
- `reconsolidation.py` — cardinality-one temporal reconciliation
- `activation.py` — ACT-R declarative activation, admission, relevance
- `spreading.py` — bounded spreading activation
- `forgetting.py` — retention lifecycle from base-level
- `working_memory.py` — chunking, selection, render serialization
- `learning.py` / `metamemory.py` — feedback and monitoring

### Storage protocols

[`src/cogkura/storage/base.py`](../src/cogkura/storage/base.py): observations, episodes, semantics, activation references, dynamics, learning, entity relationships. PostgreSQL implementations live behind `cogkura[postgres]`.

### Retrieval and context

- `recall()` ranks episodic + semantic memories by activation; soft admission bypasses threshold only.
- `valid_at` selects semantic validity time; `as_of` is cognitive evaluation time.
- `prepare_context()` runs recall once, selects bounded working-memory chunks, and returns metamemory assessment.
- Chunks are ephemeral; SUPPORT derivations provide provenance; ASSOCIATION paths are recall-time bridges only.

## Encoding specificity and contextual memory (0.16.x)

> Context affects memory **accessibility**. It does not replace topical relevance, candidate generation, tenant/subject scoping, or working-memory selection.

> Cogkura accepts **structured context**. It does not use an LLM to infer goal, activity, domain, conversation, or thread from natural-language prose.

```text
ENCODING
ObservationContext (optional)
    ↓ episodic encoding
MemoryContextSignature on StoredEpisode.encoding_context

RETRIEVAL
RetrievalContext (optional on recall APIs)
    ↓ DeterministicContextMatcher
ContextMatch (episodic diagnostics)

EPISODIC ACCESSIBILITY (0.16.2+)
ContextMatch → ContextReinstatement → Cᵢ = λctx × (M × V)
positive-only: mismatch / unavailable / no cue → zero bonus

SEMANTIC ACCESSIBILITY (0.16.3+)
SemanticMemory → unique SUPPORTS episodes → SemanticSupportContextEvidence
Rₛ = ΣRⱼ/K, Cₛ = λsem × Rₛ

METAMEMORY (0.16.4+, hardened 0.16.5)
Retrieval-level RetrievalContextDiagnostics on inspect_recall
additive MemoryAssessment.context from recall pool
```

**Engineering weights (not cognitive-science constants):**

| Parameter | Default | Role |
|-----------|---------|------|
| `context_reinstatement_weight` | `0.50` | Episodic λctx; `0.25` is smallest tested observable effect |
| `semantic_context_reinstatement_weight` | `0.25` | Semantic λsem relative to episodic default |

**Metamemory states (frozen in 0.16.5):**

| State | Meaning |
|-------|---------|
| `CONTEXT_NOT_PROVIDED` | No structured retrieval context supplied |
| `CONTEXT_UNAVAILABLE` | Cue exists but stored traces have no comparable encoding context |
| `CONTEXT_CONFLICT` | Comparable evidence exists but no plausible candidate has positive correspondence |
| `CONTEXT_UNDERSPECIFIED` | Positive matches remain ambiguous among plausible candidates |
| `CONTEXT_SUFFICIENT` | Contextual evidence meaningfully discriminates |

Decision order: not provided → unavailable → conflict → underspecified → sufficient. Structured diagnostics on `RecallInspectionResult.context` and `RetrievalDiagnostics` are the source of truth.

**Semantic generalization:** knowledge supported across many contexts averages reinstatement over the full unique support base (`K`), so broadly supported semantics become less dependent on reinstating any single encoding context.

**`MemoryContextSignature.concept_ids`:** reserved for future structured concept context; the default episodic encoder leaves it empty (`()`). Cogkura does not infer concepts from text.

**Known intentional limitations (0.16 freeze):** exact normalized matching only; no fuzzy/synonym matching; no embeddings; no negative mismatch activation; no automatic context extraction or clarification; custom attributes excluded from primary match score.

Release lineage: **`0.16.0`** encoding capture · **`0.16.1`** matching diagnostics · **`0.16.2`** episodic reinstatement · **`0.16.3`** semantic support propagation · **`0.16.4`** observability · **`0.16.5`** hardening and architecture freeze.

Design notes: [`design-encoding-context-0.16.0.md`](design-encoding-context-0.16.0.md) through [`design-context-hardening-0.16.5.md`](design-context-hardening-0.16.5.md).

Prefer coarse contextual identifiers for `location` when possible; the core library remains agnostic but applications should minimise sensitive detail.

## Deployment models

1. Same database, separate `cogkura` schema
2. Separate source and memory databases (canonical Docker example)
3. Custom storage via store protocols
4. In-memory backends (default `Memory()` for local use and tests)

## Package layout

```text
src/cogkura/
  memory.py
  observations/
  sources/
  mappers/
  storage/
  migrations/postgres/
  algorithms/
tests/
examples/
docs/
```

## Current implementation boundary (0.16.5)

Implemented:

- observation ingestion and Postgres connectors
- episodic and semantic memory with reconsolidation
- ACT-R activation, spreading, forgetting, learning
- gated admission, evidence-linked relevance, contextual association, structured relationships
- working-memory chunking with semantic structural primary (0.15.10) and frozen semantic-only support render (0.15.11–0.15.12)
- `prepare_context` / `MemoryContext` application boundary with explicit members-vs-rendered-text provenance contract
- metamemory assessment and recall inspection
- **encoding-context capture** on observations and episodes (`ObservationContext`, `MemoryContextSignature`); **`concept_ids` reserved** on default encoder
- **retrieval-context matching, reinstatement, semantic support propagation, and contextual metamemory** (0.16.1–0.16.5); architecture frozen after 0.16.5

**Provenance contract (0.15.12):** recall members → chunk → compact model-facing `serialized_text`. Supporting episodes may remain chunk members and `record_context_use` targets even when support prose is omitted from rendered context.

**Encoding-context contract (0.16.0):** encoding context is part of the episodic memory trace and is visible via `list_episodes()` / `inspect_recall()`.

**Retrieval-context contract (0.16.1–0.16.5):** retrieval context is supplied structurally on recall APIs. Episodic activation may receive positive bounded reinstatement; semantic activation may receive positive bounded support-context contribution. **`0.16.4–0.16.5`** add inspect-only contextual metamemory including `CONTEXT_CONFLICT` (0.16.5). No-context and disabled weights preserve prior release retrieval behaviour. Diagnostics do not feed working-memory render.

Planned later: additional connectors, embedding/LLM provider interfaces, benchmark suites in separate packages.
