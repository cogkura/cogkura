# CogKura Core 0.20.3 — Architecture Freeze and Agent Handoff

**Status:** Proposed design — awaiting approval  
**Date:** 25 November 2026  
**Baseline:** CogKura Core 0.20.2  
**Baseline commit:** `fb72911ce58173bf3669181d0e2c25eeefeaa58d`  
**Repositories:** `cogkura`, `cogkura-bench`  
**Next development milestone:** CogKura Agent V1

---

## 1. Objective

Complete the CogKura Core 0.20.x development programme by establishing a stable, validated cognitive-memory foundation for CogKura Agent.

The 0.20 series has addressed the major retrieval-performance limitation identified in 0.19.6, improved historical memory reconstruction, and validated the embedded SQLite runtime.

Core 0.20.2 completed its hardening release with independent performance, state-integrity, historical correctness, upgrade, compatibility and packaging evidence.

The remaining work is to:

1. Close the identified Agent-readiness questions.
2. Confirm that the public Core APIs meet the Agent's integration requirements.
3. Consolidate and freeze the cognitive and persistence contracts.
4. Verify that the release candidate preserves approved behaviour.
5. Establish clear operational and performance boundaries.
6. Publish an independently validated Core version suitable for pinning by CogKura Agent.

**0.20.3 is an architecture-freeze release, not a new cognitive capability release.**

The primary deliverable is a stable Core integration contract.

## 2. Design principles

The following principles govern 0.20.3.

### 2.1 Preserve cognitive behaviour

Existing retrieval, activation, consolidation, reconsolidation, forgetting, learning, inhibition and metamemory behaviour must remain unchanged unless a demonstrated correctness defect requires a separately approved correction.

Performance optimisation alone does not justify modifying established cognitive semantics.

### 2.2 Keep Core independent of Agent implementations

Core remains a provider-neutral cognitive-memory library.

It must not acquire dependencies on:

- Cursor, Claude Code or Codex.
- MCP transport.
- Agent sessions or orchestration.
- LLM providers.
- Project-name interpretation.
- Cloud memory services.

Agent integration must use existing public APIs wherever possible.

### 2.3 Preserve the embedded-first architecture

Local CogKura Agent must be able to operate using SQLite without installing or operating a database server.

PostgreSQL remains supported for hosted applications.

Both environments must use the same cognitive algorithms.

### 2.4 Prefer validation over new implementation

An existing capability that passes its required contract does not need to be redesigned.

Remaining limitations may be accepted when they are measurable, documented, bounded and compatible with the intended Agent deployment.

Supported correctness failures cannot be accepted merely to complete the release.

### 2.5 Maintain exact release provenance

Candidate-specific benchmark approvals, immutable frozen gold, reproducible comparisons and verified published artifacts remain mandatory.

Previous candidate approvals do not automatically transfer to 0.20.3.

---

## 3. Established 0.20.2 baseline

### 3.1 Completed capabilities

Core 0.20.2 includes:

- Optimised declarative retrieval.
- Embedded SQLite storage and lifecycle.
- In-memory and PostgreSQL storage.
- Episodic and semantic history.
- Historical reconstruction from retained snapshots.
- Corrections and withdrawals.
- Source provenance and metamemory.
- Working-memory selection and context preparation.
- Persistent activation, learning and inhibition.
- Storage migrations and restart support.
- Cross-backend cognitive validation.

### 3.2 Performance baseline

The established reference system is the selected 2018 Mac, with the agreed Python 3.11.15 testing configuration.

The accepted synthetic retrieval targets are:

| Workload | Required result |
|---|---|
| 100 observations | Warm median under 1 second |
| 250 observations | Warm median under 1 second |
| 1,000 observations | p95 under 2 seconds |

These targets passed during 0.20.2 validation.

A separate `prepare_context()` target of **warm p95 under 2.5 seconds** was proposed for the fixed 1,000-observation narrow-query configuration. Three independent 100-sample processes per backend passed that proposed threshold.

The target has not yet been formally approved.

### 3.3 Outstanding decisions carried into 0.20.3

The 0.20.2 acceptance review explicitly identifies:

- Broader correctness validation for subject-scoped processing and unaffected-subject history.
- Investigation of tenant-wide SQLite semantic hydration during subject-scoped reads.
- Formal acceptance of the proposed context-preparation performance target.
- Consolidation of historical and concurrency limitations.
- A stable Agent-facing memory and answerability contract.

These form the main implementation and review scope of 0.20.3.

---

## 4. Workstream A — Subject-scoped processing contract

**Priority: P0**

### Objective

Establish subject-scoped memory processing as a dependable operation for applications that know which subject has changed.

CogKura Agent will normally associate observations with a project or repository.

The Agent should process the affected subject rather than repeatedly processing the entire tenant.

### Established finding

A 250-observation SQLite probe measured approximately:

| Processing strategy | Initial database | After five updates |
|---|---:|---:|
| Entire tenant | 8.2 MB | 18.7 MB |
| Affected subject only | 8.2 MB | 8.25 MB |

The measurement demonstrates substantial storage amplification when the full tenant is repeatedly processed.

It does not establish a universal growth factor or mean that all additional bytes represent newly relevant history.

### Required contract

`Memory.process(tenant_id=..., subject_id=...)` must preserve the expected lifecycle for the selected subject.

Required behaviours:

1. Only the intended subject's observations are selected for the scoped processing operation.
2. Episode creation, updates and deactivation remain correct.
3. Semantic consolidation remains correct for the supported scoped workload.
4. Corrections and withdrawals retain valid source and historical relationships.
5. Reprocessing an unchanged subject does not create unexplained persistent history.
6. Unrelated subjects do not acquire unintended episode, semantic or revision-history changes.
7. Tenant isolation remains intact.

### Cross-subject dependencies

Some application updates may affect evidence or relationships extending beyond one subject.

The test programme must establish whether each supported scenario can safely use a single-subject operation.

Where broader processing is required, document the limitation and the caller's responsibility.

Do not silently assume that all semantic relationships are isolated to one project.

### Required validation

Test:

- Repeated updates to one subject.
- Corrections and withdrawals.
- Multiple projects within one tenant.
- Shared entity and supporting-evidence relationships.
- Changes affecting more than one subject.
- Reprocessing without new observations.
- Database reopen and restart.
- Historical retrieval after updates.
- Unaffected-subject history before and after processing.

Use in-memory, SQLite and PostgreSQL, with reopened SQLite where persistence matters.

### Acceptance

Subject-scoped processing is validated for the supported scenarios.

Any cross-subject dependency requiring broader processing is explicit.

No unexplained mutation or history amplification occurs for unaffected subjects in the tested contract.

**Agent scheduling, dirty-subject tracking and update coalescing remain outside Core.**

---

## 5. Workstream B — SQLite scoped-read scaling

**Priority: P0 investigation; implementation conditional**

### Objective

Determine whether tenant-wide semantic hydration creates an unacceptable scaling limitation for the initial Agent deployment.

### Existing evidence

The 0.20.2 project workload found that SQLite subject-scoped reads still hydrate the tenant's semantic inventory before applying some subject filters.

In a repeated ten-project fixture, scoped historical recall had a median of approximately 51 ms.

At 50 projects, the median increased to approximately 178 ms, although the number of candidates considered by the scoped queries remained unchanged.

The evidence suggests storage materialisation contributes to the growth, but it does not establish the entire cause.

### Investigation

Repeat the project workload while independently varying:

- Number of projects in the tenant.
- Number of observations for the requested project.
- Unrelated semantic memories.
- Historical revisions per project.
- Shared relationships and source evidence.

Measure:

- SQLite read and hydration time.
- Rows and objects materialised.
- Complete scoped recall latency.
- `prepare_context()` latency.
- CPU and memory consumption.
- Equivalent in-memory and PostgreSQL behaviour.

Use the existing ten- and 50-project fixtures, extending them only as needed to establish the relevant growth characteristics.

### Decision criteria

A storage optimisation is justified if the investigation demonstrates a material Agent-facing limitation within the agreed initial deployment scale.

Potential improvements may include more selective database hydration or indexed subject-specific reads.

Any implementation must preserve:

- Semantic reconciliation.
- Historical projections.
- Corrections and withdrawals.
- Shared support relationships.
- Deterministic ordering.
- Cross-backend result equivalence.

Do not introduce a new cognitive ranking strategy or approximate prefilter as part of this work.

### Acceptance

The scaling mechanism is understood and documented.

If the measured limitation is acceptable for the initial Agent workload, retain the current implementation and record the supported operating envelope.

If it prevents the initial Agent from meeting an agreed performance requirement, implement and validate the smallest semantics-preserving correction.

**Investigation is mandatory; optimisation is conditional.**

---

## 6. Workstream C — Context-preparation performance contract

**Priority: P0**

### Objective

Establish a specific, independently reproducible performance target for the API most directly consumed by CogKura Agent.

### Proposed acceptance target

For the established 1,000-observation mixed fixture:

- Narrow retrieval query.
- Structured goal and retrieval context.
- Previous working-memory snapshot.
- Candidate pool size 50.
- Prompt budget of 2,048 tokens.
- Learning enabled.
- Fixed cognitive evaluation time.

**Proposed target: warm `prepare_context()` p95 below 2.5 seconds.**

The target applies separately to in-memory and SQLite storage on the selected reference system.

### Measurement protocol

Use:

- Three independent processes per backend.
- At least 100 fixed-state preparations per process.
- The same fixture and configuration.
- No profiling instrumentation.
- All individual samples retained.
- Matching rendered output and relevant diagnostic state.

The previous working-memory snapshot is prepared before measurement.

Do not interleave state-changing recorded-use operations with fixed-state preparation measurements.

### Acceptance

The release owner must explicitly approve the 2.5-second target or a justified alternative.

The accepted target must then pass on the exact 0.20.3 candidate or be supported by appropriately scoped, demonstrably equivalent runtime evidence.

Document the result as a reference-workload target, not a general production SLA.

### Timing and state-integrity protocols

0.20.2 explicitly replaced its complete per-timed-run state requirement with separate timing and state-integrity validation.

That waiver was limited to 0.20.2 and must not be silently transferred.

For 0.20.3:

1. Define the accepted timing and state-integrity protocol before freeze validation.
2. Prefer reusing the established independent timing and snapshot-audit methods when appropriate.
3. Preserve the distinction between timing evidence and modified-environment snapshot experiments.
4. Require an explicit review decision for any departure from the written 0.20.3 protocol.

If no relevant runtime change has occurred, 0.20.2 evidence may support the analysis, but it does not automatically approve a new candidate.

---

## 7. Workstream D — Agent-facing public API contract

**Priority: P0**

### Objective

Confirm that CogKura Agent can perform its essential memory lifecycle using supported public APIs.

The initial integration should require no new public Core functionality.

### Required operations

| API | Agent use |
|---|---|
| `EmbeddedMemory.open()` | Open local persistent memory |
| `Memory.observe()` | Record a new observation |
| `Memory.process()` | Process the affected subject |
| `Memory.prepare_context()` | Retrieve bounded cognitive context |
| `Memory.recall()` | Retrieve ranked memories when needed |
| `Memory.inspect_recall()` | Obtain retrieval diagnostics |
| `Memory.record_context_use()` | Record actual context consumption |
| `Memory.resolve_provenance()` | Assess supporting sources |
| `Memory.learn()` | Apply explicit feedback where appropriate |
| `EmbeddedMemory.close()` | Release local storage resources |

Historical and source inspection operations must remain available through their existing public contracts.

### Local lifecycle smoke test

Use an installed Core distribution and a fresh temporary SQLite database.

The test must:

1. Open the embedded runtime.
2. Ingest project observations.
3. Process the project's subject.
4. Prepare a bounded context for a project question.
5. Verify relevant evidence and its source identity.
6. Record use of an actually consumed memory.
7. Apply a subsequent correction.
8. Confirm current and historical retrieval.
9. Close the runtime.
10. Reopen the same database in a new process.
11. Verify persisted memory and cognitive state.
12. Close cleanly.

The test should also verify that a second unrelated project can coexist without being unintentionally modified by the first project's scoped processing.

### Acceptance

The complete local lifecycle passes using public APIs.

No direct manipulation of internal Core stores is required by the integration test.

Do not implement MCP endpoints inside Core.

---

## 8. Workstream E — Answerability and missing-knowledge contract

**Priority: P0 documentation and validation**

### Objective

Define how applications must interpret cognitive memory retrieval when the available evidence does not answer the query.

### Existing behaviour

Core may select and render memories even when its metamemory assessment includes `MISSING_KNOWLEDGE`.

A 0.20.2 fixture demonstrated this with an unrelated astronomy memory returned for a production-database question.

This is not automatically a cognitive retrieval defect.

Retrieval and assessment serve different purposes.

### Contract

`MemoryContext` contains:

- Selected working-memory items.
- A deterministic rendered representation.
- A separate metamemory assessment.
- Flags and signals describing the retrieved evidence.

**A non-empty rendered context does not imply that the requested knowledge is available.**

When `MISSING_KNOWLEDGE` is present, the caller must not treat the rendered memories as sufficient evidence for answering the question.

The caller may inspect relevant partial evidence but must not invent a missing fact.

The caller should not record unrelated returned items as consumed merely because Core selected them.

### Responsibilities

Core:

- Preserves the assessment flag.
- Preserves existing retrieval and rendering behaviour.
- Exposes diagnostics and source evidence.
- Does not silently replace missing knowledge with fabricated answers.

Agent:

- Interprets `MISSING_KNOWLEDGE`.
- Distinguishes relevant evidence from unrelated rendered items.
- Produces an explicit answerability warning.
- Applies appropriate fallback or clarification behaviour.
- Records only genuinely consumed context.

### Validation

Create a contract regression demonstrating that:

1. A query lacks sufficient answer-bearing evidence.
2. Core reports `MISSING_KNOWLEDGE`.
3. Working memory may still contain an unrelated item.
4. The assessment remains available and unchanged.
5. The result is consistent across in-memory and SQLite storage.

The test documents the caller contract; it does not change ranking.

### Acceptance

The contract is documented, independently tested and included in the Agent handoff.

Implementation of the Agent's warning logic belongs to CogKura Agent V1.

---

## 9. Workstream F — Project-specific retrieval contract

**Priority: P0**

### Objective

Freeze the distinction between subject-scoped and tenant-wide retrieval.

### Project-specific queries

When the application knows that a question concerns a particular project, it should pass that project's `subject_id`.

The existing 0.20.2 regressions require known incident and migration evidence to appear within the top ten results when the appropriate subject filter is supplied.

Retain these requirements.

### Tenant-wide queries

When `subject_id` is omitted, Core performs tenant-wide cognitive retrieval.

A project name appearing in the natural-language query does not automatically establish a subject filter.

The known ten-project fixture returned incident and migration evidence at ranks 26 and 27 without the subject filter.

This remains a documented discoverability limitation, not a reason to alter Core's frozen ranking.

### Acceptance

- Scoped retrieval continues to recover the expected project evidence.
- Unscoped retrieval remains governed by the existing cognitive ranking contract.
- Broad questions continue to support tenant-wide retrieval.
- Multi-project orchestration is assigned to the Agent.
- No automatic natural-language project resolution is added to Core.

The Agent must use repository or project metadata to resolve subject identity.

---

## 10. Workstream G — Cognitive architecture freeze

**Priority: P0**

### Objective

Consolidate the previously validated cognitive contracts into one Core 0.20.3 invariant matrix.

### Frozen capabilities

| Capability | Required contract |
|---|---|
| Episodic memory | Observation-derived episodes and evidence identity |
| Semantic memory | Reconciliation, support and historical revisions |
| ACT-R activation | Existing activation calculations and ranked retrieval |
| Spreading activation | Existing associative propagation and bounds |
| Forgetting | Established accessibility and retention behaviour |
| Reconsolidation | Existing semantic update and temporal rules |
| Context reinstatement | Structured encoding and retrieval-context matching |
| Competition | Existing eligibility, ranking and diagnostic contract |
| Interference | Existing optional behavioural effect and diagnostics |
| Inhibition | Existing persistent trace and application rules |
| Learning | Explicit feedback, state updates and utility |
| Metamemory | Evidence assessment, signals and flags |
| Working memory | Bounded selection, chunking and rendering |
| Provenance | Revision-aware source attribution and inspection |
| Historical reconstruction | Supported snapshot and temporal-state semantics |

### Required invariants

The freeze must preserve:

- Tenant and subject isolation.
- Memory identity.
- Deterministic ordering.
- Activation and score values under the approved contract.
- Temporal evidence visibility.
- Knowledge time versus factual validity time.
- Correction and withdrawal semantics.
- Provenance and source identities.
- Context budget enforcement.
- Learning, activation and inhibition state.
- SQLite and PostgreSQL logical equivalence.

### Historical limits

The freeze must not claim:

- Reconstruction of historical state that was never retained.
- Universal correctness for corrupt or missing legacy snapshots.
- Coherent historical reads during arbitrary concurrent mutations.
- Guarantees for custom stores beyond their implemented contracts.

Existing `HistoricalStateUnavailableError` behaviour must remain documented.

### Acceptance

The consolidated invariant suite passes against the exact candidate.

No new cognitive algorithm or significant behavioural change is introduced without a separately reviewed correction.

---

## 11. Workstream H — Independent Bench and compatibility validation

**Priority: P0**

### CogKuraBench

The independent benchmark must verify the exact 0.20.3 candidate.

Required checks:

- Existing frozen query fingerprints.
- Complete result identities and rankings.
- Relevant diagnostic and score fields.
- Source provenance and assessment.
- Corrections and withdrawals.
- Learning, interference and inhibition workloads.
- Recorded use and cognitive-state effects.
- In-memory, SQLite, reopened SQLite and PostgreSQL comparisons.
- PostgreSQL repeated-run consistency.

### Frozen-gold policy

The original frozen gold remains unchanged.

All raw failures remain visible.

Any intentional differences must receive exact candidate-specific review.

Previously approved 0.20.2 differences, including the withdrawn-readmission correction and other approved fields, may be used as comparison evidence but do not constitute automatic approval for 0.20.3.

Approval manifests must specify the candidate, environment and exact field differences.

Cross-platform differences must be reviewed within their actual scope. Do not claim a universal strict pass from an approval that applies only to one platform and Python version.

### Python compatibility

Retain:

- CPython 3.11–3.15.
- Linux, macOS and Windows.
- SQLite wheel installation and restart.
- PostgreSQL integration.
- Static typing, formatting and linting.
- Existing security analysis gates.

The established Core 25-job CI matrix and independent Bench CI remain required.

### Release packaging

Verify:

- Exact source commit.
- Tagged commit identity.
- Wheel and source-distribution contents.
- Published artifacts against validated artifacts.
- Clean installation.
- No mandatory PostgreSQL dependency for SQLite deployment.
- Migration and fresh-process restart.
- Import and public API compatibility.

No release approval is implied merely by passing CI.

---

## 12. Workstream I — Documentation and operational freeze

**Priority: P0**

### Objective

Provide an unambiguous reference for applications using Core 0.20.3.

### Required documentation

Update or consolidate:

- Architecture and cognitive pipeline.
- Public API usage.
- Embedded SQLite lifecycle.
- Supported storage backends.
- Processing scope and dependency responsibilities.
- Historical reconstruction boundaries.
- Source provenance and answerability.
- Reference-workload performance.
- Persistence and snapshot retention.
- Error handling and unavailable-state semantics.
- Deployment limitations and security considerations.

### Operational boundaries

Document that:

- SQLite is the recommended local embedded backend.
- SQLite WAL is not supported on network filesystems under the existing deployment contract.
- Local database encryption is not provided automatically.
- PostgreSQL is optional and intended for environments requiring managed shared persistence.
- A call to `Memory.process()` is not a single atomic transaction over the entire cognitive processing pipeline.
- Applications must respect write ordering and historical-read limitations.
- Tenant-wide processing may amplify retained history.
- Reference-workload benchmarks are not production SLAs.

### Release history

Update the roadmap to identify:

- 0.20.0 as the profiling investigation, not a separately published release.
- 0.20.1 as the released optimisation milestone.
- 0.20.2 as the released hardening milestone.
- 0.20.3 as the architecture-freeze milestone.

Reconcile obsolete candidate-status statements in release documentation without deleting the original development evidence.

---

## 13. Release implementation plan

### Phase 1 — Close Agent-readiness questions

**Deliverables:**

- Subject-scoped processing invariant tests.
- Unaffected-subject history checks.
- Cross-subject dependency disposition.
- SQLite scoped-read scaling investigation.
- Reviewed context-preparation target.
- Answerability and project-scoping contracts.

**Gate:** No unresolved integration requirement that prevents the initial Agent using public Core APIs safely.

### Phase 2 — Architecture and API audit

**Deliverables:**

- Consolidated cognitive invariant matrix.
- Public API compatibility assessment.
- Embedded lifecycle smoke test.
- Historical-state contract audit.
- Storage and performance operating envelope.

**Gate:** Core can support the agreed initial Agent lifecycle without architecture changes.

### Phase 3 — Independent validation

**Deliverables:**

- Exact frozen candidate comparison.
- Cross-backend parity.
- Python compatibility matrix.
- Reproducible performance evidence.
- Candidate-scoped approvals for any intentional differences.

**Gate:** All mandatory Core and Bench validations pass under their documented approval scopes.

### Phase 4 — Freeze and release

**Deliverables:**

- Final architecture-freeze findings.
- Verified wheel and source distribution.
- Published GitHub release and PyPI package.
- Final roadmap and changelog.
- Agent integration handoff.

**Gate:** Explicit release approval obtained before tagging or publication.

The exact candidate must be fixed before final approval. New runtime changes after validation require renewed checks of the affected contracts.

---

## 14. Release acceptance criteria

Core 0.20.3 is ready for release when:

### Agent readiness

- [ ] Subject-scoped processing passes the agreed lifecycle and isolation tests.
- [ ] Unaffected subjects do not acquire unexplained persistent history in supported scenarios.
- [ ] Cross-subject dependency limits are documented.
- [ ] SQLite tenant-wide hydration scaling is investigated and explicitly dispositioned.
- [ ] Scoped project evidence remains recoverable within the top ten on the frozen fixture.
- [ ] `MISSING_KNOWLEDGE` and non-empty working memory are covered by the documented integration contract.
- [ ] A complete embedded Agent-style lifecycle passes through public Core APIs.

### Performance

- [ ] The Agent-facing `prepare_context()` reference target is explicitly approved.
- [ ] The target is satisfied with reproducible, appropriately scoped evidence.
- [ ] Existing recall targets remain satisfied without material regression.
- [ ] Timing and cognitive-state validation follow the explicitly accepted 0.20.3 protocol.

### Cognitive correctness

- [ ] All consolidated cognitive invariants pass.
- [ ] Historical reconstruction retains the supported temporal contract.
- [ ] Corrections and withdrawals remain correct.
- [ ] Provenance and metamemory remain consistent.
- [ ] No unapproved cognitive behaviour changes occur.

### Release integrity

- [ ] Independent CogKuraBench review passes.
- [ ] All backend comparisons pass or receive separately approved exact dispositions.
- [ ] CPython 3.11–3.15 CI passes.
- [ ] PostgreSQL integration passes without unintended skips.
- [ ] Wheel and source distributions are verified.
- [ ] Clean embedded installation and restart pass.
- [ ] Architecture and operational documentation is complete.
- [ ] Final release approval is recorded.

---

## 15. Non-goals

0.20.3 does not include:

- New cognitive capabilities.
- New ACT-R or forgetting formulas.
- New relevance or ranking policies.
- Semantic search using embeddings.
- Vector database integration.
- New memory storage backends.
- Cloud memory synchronisation.
- Agent sessions or context reuse orchestration.
- MCP servers or IDE integrations.
- Automatic project identification from natural language.
- Agent-specific source ingestion pipelines.
- Automatic repair of incomplete historical records.
- Full historical consistency during arbitrary concurrent writes.
- A general database storage rewrite.
- Commercial platform development.

A narrowly scoped correction may be introduced only when required to satisfy an established release contract.

Any substantial additional capability belongs to a later Core version or a separate component.

---

## 16. Required deliverables

### CogKura Core

- `docs/design-core-architecture-freeze-0.20.3.md`
- `docs/findings/0.20.3-agent-readiness.md`
- `docs/findings/0.20.3-scoped-processing.md`
- `docs/findings/0.20.3-scoped-read-scaling.md`
- `docs/findings/0.20.3-invariant-matrix.md`
- `docs/findings/0.20.3-architecture-freeze.md`
- `docs/findings/0.20.3-release.md`
- Public API integration smoke tests.
- Updated architecture and operational documentation.
- Updated roadmap and changelog.

### CogKuraBench

- Exact candidate frozen validation.
- Backend parity evidence.
- Candidate-specific approval records.
- Source and cognitive invariant comparisons.
- Final independent architecture-freeze review.

### Agent handoff

A concise integration contract documenting:

- Core version and artifact identity.
- SQLite runtime ownership.
- Subject and tenant identity conventions.
- Observation and processing boundaries.
- Context preparation and answerability.
- Provenance and historical errors.
- Recorded-use and learning responsibilities.
- Performance and memory limitations.
- Supported recovery and restart behaviour.

Retain reproducible evidence and reviewed results, but avoid committing unnecessary transient profiler outputs.

---

## 17. CogKura Agent V1 handoff

After the 0.20.3 release, CogKura Agent development resumes against a pinned Core version.

### Intended local architecture

```text
Cursor / Claude Code / Codex
            |
            v
   CogKura Agent (MCP)
            |
            +-- Session and context lifecycle
            |
            +-- Project identity resolution
            |
            +-- Observation intake and batching
            |
            +-- Dirty-subject processing
            |
            +-- Answerability and context-use policy
            |
            v
       CogKura Core
            |
            v
       Embedded SQLite
```

### Agent responsibilities

The Agent will implement:

1. Deterministic project/repository subject resolution.
2. Durable observation-intake and processing scheduling.
3. Batching and coalescing of project updates.
4. Session-aware context reuse.
5. Avoidance of unnecessary repeated `cogkura_prepare` calls.
6. Explicit handling of missing knowledge and low-confidence evidence.
7. Selective use of returned memory and feedback recording.
8. MCP integration with supported coding-Agent hosts.
9. Comparative evaluation against no-memory and static-notes baselines.

These are not additional 0.20.3 implementation requirements.

### Hosted future

The commercial platform may later replace embedded persistence with managed PostgreSQL behind the same Core cognitive engine.

Cloud tenancy, synchronisation, organisational sharing and managed operations remain outside the current Core freeze.

---

## 18. Final decision

**Proceed with CogKura Core 0.20.3 as the final architecture-freeze milestone of the 0.20.x series.**

The primary engineering work is restricted to closing the known subject-scoping and SQLite scaling questions where they affect Agent readiness.

The remainder of the milestone is contract consolidation, independent verification and release preparation.

Do not introduce speculative cognitive features or broad infrastructure changes.

The release should establish a stable Core that can be consumed by CogKura Agent without further changes to the memory architecture.

**0.20.3 is complete when CogKura Core is demonstrably ready for Agent integration—not when every possible memory feature or optimisation has been implemented.**
