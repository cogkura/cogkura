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

## Encoding context (0.16.0)

Encoding context captures the circumstances under which a memory was formed. It is **not** arbitrary application metadata and is **not** used during retrieval in `0.16.0`.

```text
Source record
    ↓
Application mapper
    ↓
ObservationInput
 ├── statement / entities / source / time
 └── ObservationContext (optional)
    ↓
Episodic encoding
    ↓
StoredEpisode.encoding_context (MemoryContextSignature)
```

**Retrieval context (0.16.1–0.16.4):** applications may supply `RetrievalContext` on recall APIs. Cogkura compares it to stored encoding context via `DeterministicContextMatcher`. In **`0.16.2`**, `DeterministicContextReinstatementPolicy` converts match evidence into bounded episodic activation contribution (`Rᵢ = Mᵢ × Vᵢ`, `Cᵢ = λctx × Rᵢ`, `Aᵢ' = Aᵢ + Cᵢ`). In **`0.16.3`**, `DeterministicSemanticSupportContextPolicy` aggregates unique `SUPPORTS` episode matches into bounded semantic activation (`Rₛ = ΣRⱼ/K`, `Cₛ = λsem × Rₛ`, `Aₛ' = Aₛ + Cₛ`). In **`0.16.4`**, `DeterministicRetrievalContextPolicy` classifies retrieval-level contextual evidence on `inspect_recall` (and fills additive `MemoryAssessment.context`) without changing activation or ranking. Reinstatement is **accessibility, not relevance**: it does not generate candidates or bypass admission gates. Semantic candidates keep `context_match=None`. No-context calls and disabled weights reproduce prior release behaviour.

```text
RetrievalCue + optional RetrievalContext
    ↓
Declarative activation (base-level + partial match + spreading + …)
    ↓
ContextMatcher → ContextMatch → ContextReinstatement → Cᵢ (episodes)
    ↓                              SemanticSupportContextEvidence → Cₛ (semantics)
    ↓
A' = A + C → threshold / rank / latency / presentation
    ↓
inspect_recall → RetrievalContextDiagnostics (0.16.4 observability only)
```

Applications supply structured context when it is already available (conversation, goal, activity, domain, and so on). Cogkura does not infer context from natural-language observation text.

**`0.16.0`** captures and persists encoding context only. **`0.16.1`** adds matching diagnostics. **`0.16.2`** applies positive-only reinstatement on episodic activation when retrieval context is populated and `context_reinstatement_weight > 0`. **`0.16.3`** propagates support-context evidence onto semantic activation when `semantic_context_reinstatement_weight > 0`. **`0.16.4`** explains contextual evidence on `inspect_recall` and fills additive `MemoryAssessment.context` without changing retrieval behaviour.

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

## Current implementation boundary (0.16.0)

Implemented:

- observation ingestion and Postgres connectors
- episodic and semantic memory with reconsolidation
- ACT-R activation, spreading, forgetting, learning
- gated admission, evidence-linked relevance, contextual association, structured relationships
- working-memory chunking with semantic structural primary (0.15.10) and frozen semantic-only support render (0.15.11–0.15.12)
- `prepare_context` / `MemoryContext` application boundary with explicit members-vs-rendered-text provenance contract
- metamemory assessment and recall inspection
- **encoding-context capture** on observations and episodes (`ObservationContext`, `MemoryContextSignature`); retrieval remains unchanged

**Provenance contract (0.15.12):** recall members → chunk → compact model-facing `serialized_text`. Supporting episodes may remain chunk members and `record_context_use` targets even when support prose is omitted from rendered context.

**Encoding-context contract (0.16.0):** encoding context is part of the episodic memory trace and is visible via `list_episodes()` / `inspect_recall()`. It does not alter recall ranking, activation, admission, or working-memory selection in this release.

**Retrieval-context contract (0.16.1):** retrieval context is supplied structurally on recall APIs; `ContextMatch` is inspectable on `RetrievalDiagnostics`.

**Context reinstatement (0.16.2–0.16.4):** when retrieval context is populated, episodic activation receives a positive bounded contribution from direct encoding-context match (`context_reinstatement_weight`), and semantic activation may receive a positive bounded contribution aggregated from unique `SUPPORTS` episode matches (`semantic_context_reinstatement_weight`). **`0.16.4`** adds inspect-only contextual metamemory (`RetrievalContextDiagnostics`, rank/threshold attribution) that does not feed working-memory render or change recall. No-context and disabled weights preserve prior release retrieval behaviour.

Planned later: additional connectors, embedding/LLM provider interfaces, benchmark suites in separate packages.
