# CogKura Core 0.20.2 — Retrieval and Historical Memory Hardening

**Status:** Approved for implementation; release acceptance remains pending  
**Date:** 22 November 2026  
**Baseline:** CogKura Core 0.20.1  
**Baseline commit:** `4c53eb6a5f77cb9c275c7b39529f08fac7e809b1`  
**Repositories:** `cogkura`, `cogkura-bench`  
**Next milestone:** 0.20.3 — Core architecture freeze and Agent handoff

---

## 1. Objective

Validate and harden the optimised CogKura retrieval engine before its final 0.20.x architecture freeze.

Core 0.20.1 significantly improved retrieval performance, strengthened historical reconstruction and passed independent cognitive and cross-backend validation.

However, several areas require additional evidence before Core can be considered a dependable foundation for production Agent integration:

1. Retrieval performance must remain consistent across independent executions.
2. `prepare_context()` must be evaluated as the primary Agent-facing operation.
3. Historical reconstruction must behave correctly within its supported boundaries.
4. Unsupported historical states must not produce misleading results.
5. Database upgrades, retained snapshots and persistent state must be verified.
6. Local SQLite and hosted PostgreSQL must preserve equivalent cognitive behaviour.
7. Remaining operational limitations must be explicit and testable.

**0.20.2 is a hardening release, not a new cognitive capability release.**

The principal objective is confidence in the existing implementation.

## 2. Established baseline

### 2.1 Retrieval performance

The released 0.20.1 implementation recorded the following results on the established 2018 Mac reference system:

| Workload | In-memory | SQLite | Target |
|---|---:|---:|---|
| 100 observations | 80 ms median | 118 ms median | Under 1 second |
| 250 observations | 207 ms median | 304 ms median | Under 1 second |
| 1,000 observations | 1.191 s p95 | 1.741 s p95 | Under 2 seconds |

The workload has five observation revisions per base observation and six additional correction/withdrawal operations.

The results were measured with the established synthetic fixture. They are not production guarantees or evidence of a universal supported memory capacity.

### 2.2 Cognitive and storage baseline

The following capabilities must be preserved:

- ACT-R activation and spreading activation.
- Episodic and semantic memory.
- Encoding context and contextual reinstatement.
- Forgetting, reinforcement and reconsolidation.
- Semantic authority and correction handling.
- Source provenance and source-aware metamemory.
- Interference, competition and inhibition.
- Working-memory selection and context preparation.
- Persistent SQLite and PostgreSQL cognitive state.
- Historical episode, semantic, revision and relationship snapshots.
- Knowledge-time and factual-validity-time semantics.
- Deterministic retrieval and diagnostic behaviour.

The 0.20.1 release is the behavioural reference for this work, subject to its explicitly documented limitations and approved benchmark differences.

## 3. Scope and priorities

| Workstream | Priority | Outcome |
|---|---|---|
| A. Performance repeatability | P0 | Reproducible retrieval latency |
| B. Context preparation performance | P0 | Measured Agent-facing latency |
| C. Historical reconstruction | P0 | Correct or explicitly unavailable historical state |
| D. Database upgrades and snapshot integrity | P0 | Safe upgrade and durable history |
| E. Cross-backend consistency | P0 | Equivalent logical memory behaviour |
| F. Representative project workloads | P1 | Confidence beyond uniform synthetic fixtures |
| G. Resource and operational limits | P1 | Documented scaling and concurrency behaviour |

Correctness defects discovered during these workstreams take precedence over performance improvements.

No speculative optimisation should be introduced without evidence of a measurable problem.

---

## 4. Workstream A — Performance repeatability

**Priority: P0**

### Objective

Establish that the improvements delivered in 0.20.1 are reproducible rather than dependent on a particular run or machine state.

### Workload

Preserve the established mixed-memory benchmark unchanged.

Run each workload using:

- 100 base observations with five revisions.
- 250 base observations with five revisions.
- 1,000 base observations with five revisions.
- The same correction and withdrawal operations.
- The same fixed query sequence.
- The same cognitive configuration and evaluation time.

Use both in-memory and SQLite storage.

PostgreSQL should be measured separately to establish its overhead and scaling characteristics.

### Measurement procedure

For each backend and workload:

1. Construct an identical fresh fixture.
2. Record runtime and storage configuration.
3. Run the benchmark without profiling instrumentation.
4. Execute at least 100 fixed-sequence retrieval queries.
5. Repeat the complete measurement in at least three independent processes.
6. Preserve each individual latency observation.
7. Report warm median and nearest-rank p95 as applicable.
8. Record CPU usage, process RSS and host activity.
9. Verify ranked outputs and cognitive state after every run.

At 1,000 observations, the p95 calculation must include the first recall, consistent with the established 0.20.1 acceptance protocol.

For the smaller workloads, separately report the first recall and the median of subsequent warm queries.

### Performance gates

Retain the previously agreed targets on the established reference system:

| Workload | Acceptance |
|---|---|
| 100 observations | Warm median under 1 second |
| 250 observations | Warm median under 1 second |
| 1,000 observations | p95 under 2 seconds |

The target applies separately to SQLite and in-memory storage.

For the release gate, each of the three independent runs should satisfy the corresponding target, rather than accepting a passing average that conceals failed runs.

Host contention and environmental interference must be recorded. A failed run may be classified as invalid only with supporting evidence and a documented rerun; it cannot simply be discarded.

A separate comparison with 0.20.1 on the same host and workload should identify material regressions even where the absolute target still passes.

These results establish acceptance for the selected reference fixture, not an application-wide latency SLA.

### Deliverable

`docs/findings/0.20.2-performance-repeatability.md`

Include the individual results, conditions, accepted measurements and any investigated failures.

---

## 5. Workstream B — Context preparation performance

**Priority: P0**

### Objective

Measure the actual memory preparation operation expected to be used by CogKura Agent.

The earlier performance investigation focused primarily on `recall()`. Agent applications will typically invoke `prepare_context()`, which also performs working-memory selection and metamemory assessment.

These additional operations must not invalidate the retrieval improvements.

### Required measurements

Compare:

- `recall()`
- `prepare_context()`
- `select_working_memory()`
- `assess_memory()`
- `record_context_use()`

Measure complete operations rather than adding overlapping internal stage timings.

Capture the additional cost of working-memory selection and metamemory processing.

### Test conditions

Use the same 100, 250 and 1,000-observation fixtures.

Vary:

- Narrow and broad queries.
- Working-memory candidate pool size.
- Context budget.
- Goal and structured retrieval context.
- Previous working-memory snapshot.
- Learning enabled and disabled.
- Relevant and irrelevant memory populations.

### Repeated context preparation

Measure repeated identical preparations under unchanged state.

This is an observational test, not a request to implement caching.

Confirm that repeated calls do not produce unexpected differences when the effective evaluation time and state are fixed.

Time-dependent activation and learning state must be treated explicitly.

### Acceptance

- Context preparation costs are measured independently of raw recall.
- No unexpected repeated work causes a major new scaling problem.
- Cognitive outputs remain correct.
- A specific context-preparation latency target is proposed from the measurements and approved before the 0.20.3 freeze.

Do not automatically apply the raw-recall two-second target to `prepare_context()` before measuring its additional work.

Agent session reuse and repeated `cogkura_prepare` policy remain responsibilities of CogKura Agent.

### Deliverable

`docs/findings/0.20.2-context-preparation.md`

---

## 6. Workstream C — Historical reconstruction hardening

**Priority: P0**

### Objective

Validate the historical state reconstruction introduced and extended in 0.20.1.

Historical retrieval must return the state that can be justified from retained evidence, rather than unintentionally reconstructing earlier knowledge from present-day records.

### Required scenarios

#### 6.1 Episode history

Verify:

- Original episode state.
- Later episode updates.
- Mixed-time episodic evidence.
- Inactive episodes.
- Episode updates following reinforcement.
- Episode history following deletion or deactivation.
- Stable memory identities across snapshots.
- Correct ordering of snapshots with equal write times.

#### 6.2 Semantic history

Verify:

- Initial semantic admission.
- Multiple supporting observations.
- Reinforcement changing confidence or evidence.
- Later corrections.
- Later withdrawals.
- Return to an earlier claim through a new revision.
- Contested claims and contradiction counts.
- Previously superseded semantic values.
- Historical source relationships and support evidence.

#### 6.3 Temporal boundaries

Test factual and knowledge time independently.

Examples must include:

- Before a correction was known.
- After a correction was known.
- Before a correction became factually effective.
- After factual validity changed.
- Before withdrawal.
- After withdrawal.
- A historical query after several subsequent modifications.

Queries must not expose evidence that did not exist at the requested knowledge time.

#### 6.4 Shared and linked evidence

Test cases where multiple semantic memories share an episode.

Confirm that historical restoration does not reintroduce an episode without the complete required compatible semantic state.

When only part of the linked historical state is available, the runtime must follow its documented unavailable-state policy rather than fabricate missing evidence.

### Unsupported historical states

0.20.1 introduced tenant-scoped ordering records and `HistoricalStateUnavailableError` for certain ambiguous backdated histories.

0.20.2 must verify:

- Backdated writes.
- Equal-time writes.
- Missing snapshots.
- Corrupt or incompatible snapshot payloads.
- Legacy databases with incomplete history.
- Custom stores lacking shared ordering records.
- Tenant isolation of ordering state.

**The system must distinguish known historical state from unavailable historical state.**

The unavailable-state exception must not be described as covering every conceivable unsupported history when the storage implementation cannot establish that guarantee.

### Concurrency boundary

Validate supported reader/writer concurrency, especially updates occurring around historical retrieval.

A preliminary ordering check is not equivalent to a consistent read transaction.

If coherent historical reads cannot be guaranteed during concurrent mutation, either implement a narrowly scoped consistency mechanism or explicitly define the unsupported boundary.

Do not silently claim transactionally consistent historical reconstruction across independently executed reads.

### Acceptance

- All newly supported historical scenarios return correct state.
- Incomplete or ambiguous scenarios follow documented behaviour.
- No future evidence leaks into historical results.
- No unrelated tenant can affect historical state.
- The existing current-state retrieval remains unchanged unless an explicit correctness fix is approved.
- No unsupported guarantee of universal historical reconstruction is introduced.

### Deliverable

`docs/findings/0.20.2-historical-integrity.md`

---

## 7. Workstream D — Database upgrade and snapshot integrity

**Priority: P0**

### Objective

Ensure users can upgrade existing CogKura databases without losing stored memory or being misled about the availability of historical snapshots.

### SQLite

Test upgrades from the supported 0.19.x database schemas through the current 0.20.1 schema, including migration `008_history_order`.

### PostgreSQL

Test equivalent upgrades through migration `015_history_order`.

### Upgrade fixtures

Construct databases containing:

- Observations and revisions.
- Encoded episodes.
- Semantic memories and historical revisions.
- Entity relationships.
- Activation and reinforcement traces.
- Learning state.
- Persistent inhibition.
- Corrections and withdrawals.
- Multiple tenants.

Prepare the fixture using the prior released version wherever possible, then upgrade using the new installed distribution.

Verify state before and after migration.

### Required guarantees

- Existing observations and revision histories remain intact.
- Existing cognitive state remains available.
- Migration is repeatable or safely recognises an already upgraded schema.
- A failed migration follows the backend's documented rollback and recovery contract.
- Missing historical snapshots are not fabricated.
- Unsupported pre-upgrade history is explicitly identifiable.
- New snapshots are stored correctly following upgrade.
- Existing tenant scoping remains intact.
- Schema changes do not introduce unexpected runtime dependencies.

### Snapshot consistency

Verify that episode and semantic snapshot writes remain atomic with their associated database operations where the storage contract promises atomicity.

Inspect failures during:

- Episode updates.
- Semantic reinforcement.
- Corrections.
- Withdrawals.
- Relationship changes.
- Tenant clearing.
- Storage rollback.

A failed transaction must not leave committed history that contradicts the committed current state.

### Snapshot retention

Assess database growth caused by the new snapshot records.

Measure:

- Snapshot count per memory.
- Bytes per episode and semantic snapshot.
- Database and index growth.
- Repeated reinforcement costs.
- Backup size.
- Memory and read overhead.

Document how snapshot retention interacts with observation retention and content minimisation.

Do not add silent snapshot pruning or preserve sensitive content beyond an established retention policy without a separate design.

### Deliverable

`docs/findings/0.20.2-storage-upgrade-and-snapshots.md`

---

## 8. Workstream E — Cross-backend consistency

**Priority: P0**

The same cognitive engine must remain usable with:

1. In-memory storage.
2. SQLite.
3. Reopened SQLite.
4. PostgreSQL.

### Required comparison

Run representative workloads containing:

- Ordinary current-state retrieval.
- Historical retrieval.
- Semantic reinforcement.
- Corrections and withdrawals.
- Shared supporting episodes.
- Learning and activation state.
- Persistent inhibition.
- Source provenance and metamemory.
- Missing-history responses.
- Tenant isolation.

Compare complete logical results, diagnostics and source evidence.

Physical storage identities and database implementation details are not required to be identical, but differences in visible results must be accounted for.

### Independent validation

CogKuraBench remains responsible for the independent comparison.

Preserve the original frozen gold and raw outputs.

Any new 0.20.2 candidate must receive its own exact, reviewed acceptance of intentional differences.

Previously approved 0.20.1 deviations must not automatically authorise a future change.

### Acceptance

No unexplained difference in cognitive retrieval, historical state or recorded-use behaviour across the compared backends.

---

## 9. Workstream F — Representative software-project workloads

**Priority: P1**

### Objective

Validate CogKura against memory patterns closer to those encountered by coding agents.

The existing synthetic workload remains necessary for longitudinal performance comparison, but it should not be the sole evidence of practical usability.

### Workload composition

Expand the existing heterogeneous software-project fixture to include:

- Architecture decisions and superseded alternatives.
- Database schema and migration histories.
- API and integration constraints.
- Deployment procedures and incidents.
- Configuration changes.
- Technical corrections and withdrawn guidance.
- Entity relationships and shared supporting evidence.
- Repeated work across multiple sessions.

Create narrow, broad, associative, current-state and historical questions with explicit expected evidence.

### Measurements

Measure both retrieval correctness and latency.

Include growth through multiple project updates rather than merely duplicating observations.

Measure memory sizes, retrieval candidate counts and the number of relevant versus unrelated memories.

### Acceptance

The workload produces correct supported results, maintains SQLite/in-memory parity and has no unexplained performance degradation.

Do not impose the synthetic benchmark's exact timing target automatically on this different workload.

Any performance concern should be classified using its measured cause and expected Agent impact.

---

## 10. Workstream G — Operational reliability

**Priority: P1**

### SQLite

Revalidate:

- Database open and close.
- Fresh-process restart.
- WAL and locking behaviour.
- Busy timeout.
- Write cancellation.
- Transaction rollback.
- Backup and restore.
- Unsupported SQLite version rejection.
- Concurrent readers and serialized writes.
- Database file and snapshot growth.

### PostgreSQL

Revalidate:

- Transaction rollback.
- Concurrent readers and writers.
- Tenant isolation.
- Connection lifecycle.
- Migration failure behaviour.
- Snapshot atomicity.
- Repeated processing and cleanup.

### Shared runtime guarantees

Confirm that:

- `Memory.process()` does not claim a single transaction across all processing stages.
- Write cancellation does not imply rollback of an already accepted write.
- Unsupported historical data is not presented as complete.
- SQLite remains local-first and does not require PostgreSQL.
- PostgreSQL remains an optional Core dependency.
- Security and storage boundaries remain explicit.

Do not expand the Core storage protocols without an identified, unavoidable correctness requirement.

---

## 11. Implementation phases

### Phase 1 — Baseline and performance verification

1. Pin released Core 0.20.1.
2. Capture the agreed reference environment.
3. Repeat retrieval benchmarks.
4. Measure context preparation.
5. Identify any significant regression or remaining bottleneck.

**Gate:** Reproducible performance results and a clear disposition for any failed target.

### Phase 2 — Historical integrity and migration

1. Expand temporal and shared-evidence regression tests.
2. Validate missing and ambiguous history behaviour.
3. Test actual database upgrades.
4. Verify snapshot transaction integrity.
5. Measure history-related storage growth.

**Gate:** No unresolved supported-history correctness or data-integrity defect.

### Phase 3 — Project workload and backend validation

1. Run the heterogeneous software-project fixture.
2. Compare in-memory, SQLite and PostgreSQL.
3. Verify restart and persistence.
4. Confirm diagnostics and provenance.
5. Assess any newly identified scaling limitations.

**Gate:** No unexplained cognitive or backend inconsistency.

### Phase 4 — Release candidate and acceptance

1. Freeze the candidate commit.
2. Run full Core and independent Bench suites.
3. Run the CPython 3.11–3.15 compatibility matrix.
4. Validate the candidate-specific frozen comparison.
5. Build and test wheel and source distributions.
6. Install the wheel in clean environments.
7. Verify migration, retrieval and fresh-process restart.
8. Complete performance and historical-integrity findings.
9. Review release blockers and residual limitations.
10. Obtain separate approval before tagging and publication.

**Gate:** All mandatory hardening criteria resolved.

---

## 12. Acceptance criteria

0.20.2 may be released when:

- [ ] Retrieval benchmarks have been repeated independently.
- [ ] Existing 100, 250 and 1,000-observation performance targets are satisfied under the agreed protocol.
- [ ] `prepare_context()` latency and incremental costs are measured.
- [ ] A proposed Agent-facing context-preparation target is documented for 0.20.3 approval.
- [ ] Representative software-project workloads have been validated.
- [ ] Supported historical states reconstruct correctly.
- [ ] Missing and ambiguous historical states follow documented unavailable-state behaviour.
- [ ] No future evidence leaks into historical retrieval.
- [ ] Shared semantic and episodic histories retain appropriate consistency.
- [ ] SQLite and PostgreSQL upgrade tests pass.
- [ ] Snapshot writes and rollback behaviour are verified.
- [ ] Snapshot growth and retention implications are documented.
- [ ] Independent four-backend comparisons pass without unapproved differences.
- [ ] Frozen cognitive behaviour remains intact except for specifically approved correctness changes.
- [ ] Python 3.11–3.15 support remains verified.
- [ ] Exact release candidate wheel and source distributions are validated.
- [ ] No unresolved data-loss, historical-consistency or release-blocking performance defect remains.

A failed non-critical exploratory test may be accepted as a documented limitation only when its effect, supported scope and impact on Agent readiness are understood and explicitly reviewed.

Correctness failures in supported scenarios cannot be reclassified as limitations merely to release the candidate.

---

## 13. Non-goals

0.20.2 does not include:

- New cognitive algorithms.
- Changes to ACT-R activation or forgetting formulas.
- New competition or inhibition behaviour.
- Vector retrieval or embeddings.
- A new candidate-pruning policy.
- A general retrieval architecture redesign.
- Mandatory PostgreSQL deployment.
- Cloud memory synchronisation.
- MCP or IDE integration.
- Agent sessions or repeated-context caching.
- Full historical reconstruction of every possible legacy state.
- Automatic reconstruction of previously unrecorded snapshots.
- Major new storage protocols.
- Commercial platform functionality.

A narrowly scoped performance or correctness fix may be accepted if supported by investigation and independent regression validation.

Any larger architectural change requires its own design and approval.

---

## 14. Documentation deliverables

### CogKura Core

- `docs/design-retrieval-hardening-0.20.2.md`
- `docs/findings/0.20.2-performance-repeatability.md`
- `docs/findings/0.20.2-context-preparation.md`
- `docs/findings/0.20.2-historical-integrity.md`
- `docs/findings/0.20.2-storage-upgrade-and-snapshots.md`
- `docs/findings/0.20.2-release.md`
- Updated `docs/roadmap.md`
- Updated `docs/architecture.md`
- Updated `docs/sqlite-storage.md`
- Updated `CHANGELOG.md`

### CogKuraBench

- Updated 0.20.2 Core compatibility.
- Exact frozen comparison and candidate-specific reviews.
- Historical and cross-backend regression evidence.
- Performance comparison results.
- Updated independent validation guide.

Do not commit every transient profiling output.

Retain the raw and reviewed evidence required for independently reproducing release conclusions.

---

## 15. Boundary with 0.20.3

0.20.2 establishes that the implemented architecture behaves dependably under agreed workloads and operational conditions.

0.20.3 will establish the final Core integration baseline for Agent development.

### 0.20.2 owns

- Performance repeatability.
- Context-preparation performance.
- Historical correctness.
- Migration and snapshot safety.
- Cross-backend regression validation.
- Operational hardening.
- Explicit supported and unsupported boundaries.

### 0.20.3 owns

- Final Core architecture and API audit.
- Consolidated invariant and compatibility matrix.
- Independent architecture-freeze validation.
- Exact release packaging and documentation.
- Minimal Agent-facing public API smoke test.
- Final approval of the Core integration contract.
- Release handoff to CogKura Agent.

0.20.2 must not become an open-ended optimisation programme.

If its acceptance criteria pass, remaining speculative performance improvements can be considered for later releases.

---

## 16. Decision

**Proceed with CogKura Core 0.20.2 as a bounded hardening release.**

Retain the 0.20.1 cognitive retrieval architecture and its demonstrated performance improvements.

Focus implementation on evidence, historical integrity, upgrade safety and reproducible behaviour.

No new cognitive capabilities are authorised by this design.

Once the hardening gates pass, proceed to **0.20.3 — Core architecture freeze**, followed by production development of CogKura Agent.

The end goal is a stable Core suitable for:

**Local Agent → CogKura Core → SQLite**

**Hosted Platform → CogKura Core → PostgreSQL**

Both environments must use the same cognitive engine with consistent, independently validated behaviour.

**0.20.2 is complete when the existing implementation is demonstrably dependable—not when every conceivable optimisation or historical edge case has been implemented.**
