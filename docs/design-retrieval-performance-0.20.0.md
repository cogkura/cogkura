CogKura Core 0.20.x — Retrieval Performance Investigation and Optimisation
Status: Proposed design — awaiting approval
Date: 17 November 2026
Baseline: CogKura Core 0.19.6
Baseline commit: 9745201bfcc5410d678a07737a4c935dd294e5b4
Repositories: cogkura, cogkura-bench
Scope: Cognitive retrieval performance, scaling and preservation of behavioural correctness
1. Objective
Investigate and resolve the significant retrieval-performance degradation observed as CogKura's memory population grows.
Core 0.19.6 established embedded storage correctness and cross-backend compatibility but identified an important scaling limitation.
A mixed workload containing approximately 1,000 observations and 5,000 revisions required roughly 83 seconds for a single recall.
Both SQLite and in-memory storage exhibited similar timings.
The objective of 0.20.x is to make CogKura's retrieval engine suitable for interactive Agent applications while preserving the existing cognitive capabilities and their verified behaviour.
Optimise the implementation of cognition, not simplify or weaken the cognitive model to obtain faster benchmarks.
Success criteria
1. Identify the dominant contributors to retrieval latency.
2. Establish the observed computational complexity of the expensive paths.
3. Distinguish storage costs from shared cognitive processing costs.
4. Implement targeted, measured optimisations.
5. Preserve retrieval correctness, historical semantics and cognitive behaviour.
6. Demonstrate materially improved scaling on representative workloads.
7. Establish practical performance expectations for local and hosted deployment.
8. Complete independent CogKuraBench validation before freezing the optimised architecture.
2. Baseline evidence
The 0.19.6 performance investigation recorded:
Workload	SQLite recall	In-memory recall
Approximately 100 observations	0.90 s median	Not recorded in the same mixed-workload summary
Approximately 250 observations	5.32 s median	5.10 s median
Approximately 1,000 observations	82.73 s, one sample	82.21 s, one sample
The workloads contain multiple observation revisions and a small number of authoritative corrections and withdrawals.
They were executed on a 2018 x86-64 macOS machine. The 1,000-observation comparison contains only one recall per backend.
These observations suggest substantial scaling costs in shared processing, but they do not establish a precise complexity class or isolate a particular function.
The increase from 250 to 1,000 observations is approximately fourfold, while the measured recall time increases approximately sixteenfold.
This is consistent with possible quadratic behaviour, but remains a hypothesis requiring profiling and controlled measurements.
Important boundaries
* SQLite has not been demonstrated to be the principal bottleneck.
* PostgreSQL has not yet been benchmarked against this exact workload.
* In-memory parity does not prove storage has zero cost.
* Similar backend timings suggest substantial work in shared processing but do not identify the responsible algorithm.
* No production latency or supported population limit has been established.
These distinctions must be maintained throughout the investigation.
3. Existing retrieval architecture
The current shared retrieval pipeline is centred on:
Memory.recall()
→ Memory._rank_declarative_results()
→ ACTRDeclarativeActivator.rank_evaluated()
Memory.prepare_context() uses the same declarative retrieval path through _prepare_retrieval_state() and subsequently performs working-memory selection and metamemory assessment.
3.1 Storage and historical preparation
_rank_declarative_results() currently performs operations including:
* Loading episodes and semantic memories.
* Loading superseded and retracted semantic states.
* Constructing episode-support and semantic-slot indexes.
* Loading semantic revision history.
* Projecting semantic knowledge for the requested time.
* Hiding unknown, superseded or withdrawn evidence as appropriate.
* Filtering stale episodic content.
* Constructing activation candidates.
* Applying forgetting eligibility.
* Loading activation traces and learned associations.
* Loading entity relationships and relevant inhibition traces.
These operations should be measured independently.
In particular, semantic history projection and stale-content filtering must be investigated as possible contributors to growing-history costs.
3.2 Declarative activation and ranking
ACTRDeclarativeActivator.rank_evaluated() currently performs work including:
* Query-derived entity and tag seeding.
* Spreading activation.
* Candidate IDF calculations.
* Semantic-slot matching and admission.
* Structured slot-fit calculation.
* Context reinstatement.
* Individual candidate scoring.
* Semantic relevance and association processing.
* Authoritative current-state admission.
* Persistent inhibition when enabled.
* Competition and interference evaluation.
* Eligibility, sorting and duplicate collapse.
The implementation constructs several intermediate indexes and performs multiple passes over candidates.
The number of candidates, semantic supports, relationships and historical revisions may affect these operations differently.
No individual stage should be declared responsible until the profiles demonstrate its actual contribution.
4. Investigation hypotheses
ID	Hypothesis	Investigation
H1	Semantic history projection scales poorly with revision count	Profile projection against independent observation/revision growth
H2	Repeated feature extraction or comparison creates excessive CPU work	Count tokenisation, semantic matching and feature construction calls
H3	Association and relationship evaluation repeatedly traverses overlapping evidence	Profile semantic relevance and relationship-path computation
H4	Competition evaluation scales with the number of candidate pairs	Measure evaluated pairs and time against candidate count
H5	Candidate construction and ranking repeatedly process unrelated memories	Measure candidate population at each pipeline stage
H6	SQLite semantic inventory materialisation adds substantial overhead	Separate SQL, hydration and shared processing timings
H7	Repeated context preparation duplicates expensive work	Compare repeated identical queries without altering production behaviour
H8	Historical corrections and withdrawal checks introduce revision-sensitive scaling	Isolate current, historical, corrected and withdrawn memory workloads
The investigation may reject any or all hypotheses.
5. Workstream A — Deterministic performance baseline
Proposed release: 0.20.0
Priority: P0
Objective
Create a reproducible benchmark that separates workload size, revision depth and cognitive configuration.
Workload matrix
Profile	Observations	Revisions per observation	Approximate total revisions
XS	25	1	25
Small	100	5	500
Medium	250	5	1,250
Large	1,000	5	5,000
Revision-heavy	100	20	2,000
Population-heavy	1,000	1	1,000
Use additional sizes between these points where needed to distinguish growth rates.
The original 0.19.6 mixed workload must be retained unchanged for longitudinal comparison.
Additional fixtures should vary:
* Episodic versus semantic memory ratios.
* Number of historical revisions.
* Corrections and withdrawals.
* Entity and relationship density.
* Number of semantic supporting episodes.
* Repeated versus distinct query features.
* Current versus historical retrieval.
* Narrow versus broad queries.
Include a realistic software-project workload containing architecture decisions, migrations, incidents, revisions and superseded technical guidance.
Do not assume that uniform synthetic observations represent actual coding-Agent memory.
Benchmark controls
Fix:
* Input data and seed.
* Query sequence.
* Simulated evaluation time.
* Cognitive configuration.
* Retrieval limits and candidate limits.
* Python and SQLite versions.
* Core source commit.
* Host and filesystem.
* Process and cache conditions.
Run the same logical workload against SQLite and in-memory storage.
Add PostgreSQL as a storage comparison, but do not require PostgreSQL to outperform SQLite.
Measurements
Record:
* End-to-end wall time.
* Process CPU time.
* Peak resident memory.
* Allocation and object counts in separate profiling runs.
* Candidate and revision populations.
* Storage statements and database-read volume.
* Median and p95 latency when samples are sufficient.
* Cold-process and warm-process behaviour.
* Exact logical outputs and benchmark fingerprints.
Record individual observations separately from aggregate statistics.
Do not report p95 from one or five samples as a meaningful tail-latency estimate.
Performance results are observational and must not be mixed into correctness fingerprints.
Deliverable
A reproducible 0.20.0 baseline, preserving the exact 0.19.6 results and capturing matched new runs on the same environment.
6. Workstream B — Retrieval pipeline profiling
Proposed release: 0.20.0
Priority: P0
Objective
Identify where total recall time is spent, down to the expensive functions and data structures.
Stage instrumentation
Add optional diagnostic instrumentation around the retrieval pipeline.
The initial stage inventory should include:
Stage	Measurements
Episode loading	Time, rows and objects
Semantic loading	Time, current and historical objects
Revision loading	Time, revision count
Historical projection	Time, processed semantic revisions
Stale episode filtering	Time, episodes and evidence examined
Candidate construction	Time, candidate count
Forgetting eligibility	Time, candidates rejected
Activation and learning-state reads	Time, traces and associations
Spreading activation	Time, candidate/entity counts
Slot matching and admission	Time, evaluated candidates
Candidate scoring	Time and scoring-call count
Semantic relevance	Time, support and relationship work
Competition and inhibition	Time and evaluated pairs
Final ordering and collapse	Time, eligible and returned counts
Context preparation	Additional working-memory and metamemory time
Use both wall-clock and CPU profiling to distinguish waits from computation.
Where asyncio.gather() overlaps operations, do not present summed substage wall times as an exact decomposition of total latency.
Profiling techniques
Use the Python standard-library profiler as a baseline, supplemented by sampling or line-level profiling where necessary.
Optional diagnostic tools may be used in development without introducing new mandatory Core dependencies.
Profile representative executions from fresh processes.
Capture:
* Inclusive and exclusive function time.
* Invocation count.
* Time per invocation.
* Allocation hotspots.
* Call-stack distribution.
* Repeated feature construction.
* Collection sizes and growth.
Avoid per-candidate logging in the primary timing run because it may substantially distort results.
Production APIs should not incur profiling overhead when diagnostics are disabled.
Output
Produce an evidence-based cost breakdown showing the most expensive stages and their growth as memory populations increase.
Investigate the dominant contributors first rather than optimising every stage.
Acceptance
The investigation must identify a quantified set of hotspots explaining the majority of measured recall time.
If the initial profiling cannot explain the behaviour, extend the investigation rather than selecting an optimisation based on speculation.
7. Workstream C — Complexity and algorithmic analysis
Proposed release: 0.20.0
Priority: P0
For each significant hotspot:
1. Identify its input dimensions.
2. Determine the relevant loops and repeated operations.
3. Describe its expected computational complexity.
4. Compare expected growth against measured growth.
5. Identify repeated calculations or unnecessary materialisation.
6. Establish which results must remain globally available for correct retrieval.
7. Propose one or more semantics-preserving improvements.
Possible complexity dimensions include:
* (N): number of candidate memories.
* (R): number of historical revisions.
* (E): number of evidence or relationship edges.
* (T): number of reference traces.
* (C): number of evaluated competitive pairs.
Do not infer complexity solely from elapsed-time ratios.
Example investigation
If semantic relevance processing repeatedly evaluates each candidate against a large collection of other candidates, inspect whether reusable indexes can reduce repeated searches.
If the same canonical feature set is regenerated multiple times, investigate computing it once per relevant object and reusing it safely.
If historical projection repeatedly reconstructs unchanged information, determine whether a revision-aware index or reusable representation can preserve the same results.
Each proposal must identify the existing behavioural contract that it preserves.
8. Workstream D — Targeted optimisation
Proposed release: 0.20.1
Priority: P0 after profiling
The optimisation strategy must follow measured results.
Candidate approaches include:
8.1 Reduce repeated computation
* Reuse canonical text features within a retrieval.
* Build shared indexes once per retrieval.
* Avoid repeated iteration over the same support population.
* Replace repeated linear searches with appropriate lookups.
* Reduce unnecessary intermediate object construction.
These are candidates for early optimisation because they may preserve the existing computation while reducing duplicated work.
8.2 Improve data structures and traversal
* Introduce indexed evidence and relationship lookups.
* Reduce repeated association-path evaluation.
* Avoid unnecessary cross-products.
* Reuse previously constructed support mappings.
* Preserve deterministic iteration and tie-breaking.
Any change to ordering must be investigated explicitly.
8.3 Optimise historical processing
* Avoid reconstructing identical revision state repeatedly.
* Separate current-state and historical requirements where equivalent.
* Make exact revision lookups more selective.
* Preserve effective-time and knowledge-time boundaries.
* Maintain withdrawn-evidence suppression and historical visibility.
Historical correctness takes precedence over performance.
8.4 Selective candidate processing
Candidate pruning is potentially valuable but also the highest-risk optimisation.
CogKura's existing retrieval semantics include global ranking, spreading activation, IDF, semantic admission, relationship evidence, contextual activation and competition.
These mechanisms may depend on candidates that do not initially appear relevant.
Therefore, replacing global retrieval with a simple top-K text or vector prefilter is not automatically behaviour-preserving.
Any selective retrieval strategy must either:
* Demonstrate that omitted candidates cannot influence the specified result under the existing contract; or
* Be introduced as an explicitly new retrieval policy with separate design approval and benchmark expectations.
No silent prefiltering is permitted.
8.5 Caching
Evaluate caching only after the dominant costs are known.
Cache keys and invalidation must account for:
* Tenant and subject.
* Query and retrieval context.
* Evaluation time and valid_at.
* Cognitive configuration.
* Observation, episode and semantic updates.
* Learning and reinforcement.
* Activation and forgetting state.
* Inhibition.
* Authority corrections and withdrawals.
Time-dependent cognitive state means a cache cannot simply assume that an unchanged database always produces an unchanged retrieval result.
Cross-session Agent context reuse remains a separate integration concern.
9. Workstream E — Storage-specific optimisation
Proposed release: 0.20.1–0.20.2, conditional
Storage changes are only required if profiling demonstrates that they materially contribute to the overall cost.
SQLite
Investigate:
* Tenant-wide semantic inventory loading.
* Historical revision hydration.
* Child-row materialisation.
* Query indexes and execution plans.
* Unnecessary round trips or repeated scans.
* The number of hydrated records relative to the records actually required.
PostgreSQL
Run equivalent representative workloads to establish whether its query execution or concurrency characteristics materially improve retrieval.
Any PostgreSQL-specific improvement must preserve the shared cognitive semantics.
Architectural constraint
CogKura must continue to support:
* In-memory storage.
* SQLite for local embedded deployment.
* PostgreSQL for hosted deployment.
Do not add a mandatory PostgreSQL dependency to local CogKura Core.
Do not create separate cognitive algorithms for SQLite and PostgreSQL.
Backend-specific implementations may optimise equivalent storage operations, but behavioural differences require explicit review.
10. Workstream F — Correctness and independent validation
Applies to every optimisation
Priority: P0
Performance improvements must not invalidate the established cognitive contracts.
Frozen behavioural baseline
Use released Core 0.19.6 as the reference implementation.
Retain:
* Original frozen gold.
* Raw benchmark results.
* Previously reviewed correction fingerprints.
* Previously reviewed supporting-event ordering differences.
* Python and platform-specific exact acceptance rules.
Existing approvals are bound to their identified candidates and must not automatically transfer to new 0.20.x implementations.
Required invariants
Verify no unintended change in:
* Retrieved memory identities.
* Ranking and ordering.
* Activation and scoring.
* Semantic admission.
* Source evidence and provenance.
* Historical revision visibility.
* Corrections and withdrawals.
* Forgetting and reactivation.
* Learning and reinforcement.
* Interference and inhibition.
* Working-memory selection.
* Metamemory assessment.
* Tenant isolation and temporal boundaries.
Retain exact scores and fingerprints where the existing contract requires them.
Do not introduce broad numeric tolerances simply to facilitate optimisation.
Independent Bench
CogKuraBench must compare:
1. Released 0.19.6 baseline.
2. New 0.20 candidate.
3. In-memory implementation.
4. SQLite implementation.
5. SQLite close/reopen.
6. PostgreSQL implementation.
Use existing workload and source-diagnostic fixtures, supplemented by targeted performance regression cases.
Any intentional behavioural deviation requires a separate design review and explicit approval, not merely a performance justification.
Regression isolation
Every optimisation should be measurable independently before being combined with additional changes.
Where possible, preserve before/after profiles, output comparisons and the exact source commit.
11. Workstream G — Performance targets and acceptance
Proposed release: 0.20.2–0.20.3
Performance requirements must be defined before accepting implementation changes.
The initial 0.20.0 investigation will establish a reference host, realistic workload and repeatable baseline.
Proposed targets for ratification
Measure	Initial target
Small interactive memory	Approximately 100 observations: subsecond warm recall
Medium interactive memory	Approximately 250 observations: around 1 second or less
Growing Agent memory	Approximately 1,000 observations: p95 under 2 seconds on an agreed modern reference system
Scaling	Demonstrable improvement in the observed growth curve
Correctness	No unapproved cognitive deviations
Memory	Bounded, measured growth without unexplained allocation spikes
These are proposed usability targets, not existing performance claims or automatically approved release gates.
The 0.20.0 profiling findings must either ratify them with a defined workload and host or propose evidence-backed alternatives before the optimisation release is accepted.
A significant speedup on the original 2018 test machine is useful comparison evidence, but hardware-specific speedups alone do not establish acceptable production latency.
Performance validation
Use sufficiently repeated, uninstrumented executions for primary latency measurements.
Keep profiler overhead separate.
Test both warm and fresh-process behaviour, and report cold-device limitations honestly where page-cache control is unavailable.
Performance improvements must persist as memory size increases; optimising one benchmark query without improving the underlying scaling problem is insufficient.
12. Proposed release sequence
0.20.0 — Retrieval profiling and root-cause analysis
Objective: Explain the scaling problem.
Deliver:
* Deterministic retrieval workload matrix.
* End-to-end timing baselines.
* Stage-level instrumentation.
* CPU and allocation profiles.
* Candidate, revision and relationship counters.
* Confirmed computational hotspots.
* Root-cause findings.
* Prioritised optimisation proposals.
* Ratified performance acceptance targets.
Release gate: The dominant performance problem is understood and supported by repeatable evidence.
This milestone should not require changing cognitive behaviour.
0.20.1 — Shared cognitive processing optimisation
Objective: Remove confirmed inefficient computation.
Potential work:
* Feature reuse.
* Association and support indexing.
* Repeated-scan elimination.
* Improved algorithmic data structures.
* Historical projection improvements where justified.
Implement only changes supported by 0.20.0 profiling.
Release gate: Significant measured improvement with no unapproved frozen-output deviations.
0.20.2 — Retrieval scaling and storage efficiency
Objective: Address remaining growth-related costs.
Conditional work:
* Selective data access.
* Reduced semantic materialisation.
* Improved evidence and revision loading.
* Backend query optimisation.
* Safe candidate-lifecycle improvements.
A selective candidate discovery design requires its own review because it may affect global cognitive ranking.
Release gate: Agreed growing-memory performance targets are met, or outstanding targets have an explicit release-blocking disposition.
0.20.3 — Performance architecture freeze
Objective: Establish a validated retrieval implementation for Agent integration.
Deliver:
* Consolidated regression tests.
* Independent Bench validation.
* Cross-backend parity.
* Restart and historical correctness.
* Repeatable performance baselines.
* Resource and scaling limits.
* Architecture and operational documentation.
* Final findings and release acceptance.
Release gate: The optimised retrieval engine is correct, reproducible and suitable for the agreed interactive workloads.
The exact 0.20.x release boundaries may be revised after the root-cause investigation; the gates remain applicable.
13. Required test and CI coverage
Retain the established Core compatibility matrix:
* CPython 3.11–3.15.
* Linux, macOS and Windows.
* SQLite and in-memory.
* PostgreSQL-enabled integration tests.
* Dependency-free wheel installation and restart checks.
Run the full Core gates:
uv sync --all-extras --dev --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
Retain the independent Bench strict and reviewed comparisons.
Performance tests should primarily be separately reproducible benchmarks rather than timing assertions in ordinary shared-host CI.
Deterministic output and complexity-regression checks may run in CI; host-dependent latency acceptance should use a controlled benchmark environment.
14. Deliverables
Core
* docs/design-retrieval-performance-0.20.0.md
* docs/findings/0.20.0-retrieval-profiling.md
* Reproducible retrieval profiling scripts.
* Fixed performance workload definitions.
* Stage-level diagnostic counters and optional profiling support.
* Optimisation design notes for subsequent releases.
* Updated architecture, performance and configuration documentation.
* Final 0.20.x performance-freeze findings.
Bench
* Reproducible 0.19.6-to-0.20 comparison.
* Retrieval workload scaling fixtures.
* Exact cognitive output comparisons.
* Candidate-specific approval manifests where required.
* CPU, memory and latency reporting.
* Cross-backend performance and correctness findings.
Transient raw profiler outputs may remain ignored. Retain only the reviewed data and reproducible evidence required to substantiate release conclusions.
15. Non-goals
This work does not include:
* New cognitive memory capabilities.
* Changes to activation, forgetting, competition or inhibition formulas merely for speed.
* Automatic replacement of SQLite with PostgreSQL.
* Mandatory vector databases or embeddings.
* Cloud memory synchronisation.
* MCP transport changes.
* Agent session management.
* Agent-side repeated-context caching.
* Commercial platform functionality.
* Removal of supported Python versions.
* Loosening frozen benchmark contracts.
Any intentionally changed cognitive behaviour must be separately designed and approved.
16. Agent integration boundary
The results of this work will support two deployment environments.
Local Agent: CogKura Core with embedded SQLite, supporting Cursor, Claude Code and Codex without external database infrastructure.
Hosted Platform: The same CogKura Core algorithms backed by PostgreSQL, exposed through a managed service.
Performance optimisations should benefit both environments wherever possible.
Agent-level context reuse and repeated preparation policies will be addressed after the Core retrieval cost is understood.
The production Agent may add its own lifecycle and caching capabilities, but these must not be used to conceal an unacceptably expensive underlying Core retrieval path.
17. Final acceptance
The 0.20.x retrieval-performance programme is complete when:
* The observed 0.19.6 scaling limitation has a demonstrated root cause.

* The principal expensive paths have been profiled and quantified.

* Proposed optimisations are supported by measurements.

* Retrieval scaling has materially improved on representative workloads.

* Ratified interactive performance targets are satisfied.

* SQLite and PostgreSQL remain supported without separate cognitive engines.

* Historical knowledge, corrections and provenance remain correct.

* Activation, ranking, learning, forgetting, interference and inhibition retain approved semantics.

* Independent Bench validation passes with no unapproved differences.

* Python 3.11–3.15 compatibility remains intact.

* Release artifacts and exact candidate identities are verified.

* Remaining performance limitations are documented.

18. Decision
Proceed with 0.20.0 as a dedicated retrieval-performance investigation.
Do not implement speculative candidate pruning, storage replacement or cognitive algorithm changes before the root cause has been established.
Prioritise improvements to shared computation and data structures that preserve established behaviour.
Only introduce selective retrieval or changed ranking semantics through a separate, explicitly approved design.
The desired outcome is a CogKura engine that retains the cognitive fidelity developed through versions 0.15–0.19 while becoming efficient enough for interactive Agent workloads.
Performance must improve without sacrificing the properties that distinguish CogKura from simpler memory systems.
