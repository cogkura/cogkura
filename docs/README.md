# Documentation

This index separates guides for using Core from design history and validation evidence. Core 0.19.5 froze the embedded architecture; 0.19.6 focuses on stabilisation and compatibility. Historical design notes describe decisions at the time and do not by themselves establish the current release status.

## Start here

- [Project README](../README.md): installation, quick start and embedded example.
- [Architecture](architecture.md): public API, storage protocols, deployment models and cognitive boundaries.
- [SQLite storage](sqlite-storage.md): embedded lifecycle, compatibility, backups, concurrency and security limits.
- [Application integration](application-integration.md): prepare and record the use of memory context.
- [Configuration](configuration.md): cognitive configuration and defaults.
- [Roadmap](roadmap.md): release history, current candidate and later work.
- [Vision](vision.md): the project's direction and scope.
- [Contributing](../CONTRIBUTING.md): development setup and validation.

## Cognitive memory guides

- [Declarative activation](declarative-activation.md) and [spreading activation](spreading-activation.md).
- [Forgetting](forgetting.md), [reconsolidation](reconsolidation.md) and [learning](learning.md).
- [Working memory](working-memory.md) and [metamemory](metamemory.md).
- [Source-aware metamemory and semantic inspection](source-metamemory.md).

## Current validation and operational findings

- [0.19.6 embedded stabilisation findings](findings/0.19.6-embedded-stabilisation.md): candidate matrix, artifact identity, independent Bench approvals and measured retrieval limits.
- [0.19.5 embedded architecture freeze](findings/0.19.5-embedded-architecture-freeze.md) and [0.19.4 reliability](findings/0.19.4-embedded-reliability.md).
- [0.18.4 source-provenance freeze](findings/0.18.4-source-provenance-architecture-freeze.md) and [0.17.7 interference/inhibition freeze](findings/0.17.7-interference-inhibition-architecture-freeze.md).
- [Independent CogKuraBench candidate findings](https://github.com/cogkura/cogkura-bench/blob/main/docs/findings/embedded-core-0.19.6-788444d-candidate.md): raw and reviewed 0.19.6 comparisons.

Files in `findings/data/` are retained measurement and comparison evidence. They are not general usage guides. Raw benchmark results, gold and separately approved reviews have distinct roles and should be preserved.

## Design history

### Embedded storage and stabilisation (0.19)

- [Embedded memory architecture freeze (0.19.5)](design-embedded-architecture-freeze-0.19.5.md)
- [Embedded reliability and performance hardening (0.19.4) — design](design-embedded-reliability-0.19.4.md)
- [Embedded runtime (0.19.3) — design](design-embedded-runtime-0.19.3.md)
- [Embedded runtime stabilisation and Python compatibility (0.19.6)](design-embedded-stabilisation-0.19.6.md)
- [SQLite cognitive state (0.19.2) — design](design-sqlite-cognitive-state-0.19.2.md)
- [SQLite derived memory (0.19.1) — design proposal](design-sqlite-derived-memory-0.19.1.md)
- [SQLite foundation (0.19.0) — design proposal](design-sqlite-foundation-0.19.0.md)

### Source provenance and authority (0.18)

- [Cogkura 0.18.2 — Authoritative corrections and withdrawals](design-authoritative-corrections-0.18.2.md)
- [Cogkura 0.18.1 — Source authority and single-observation consolidation](design-source-authority-0.18.1.md)
- [CogKura Core 0.18.3 — Source-aware metamemory and conflict observability](design-source-aware-metamemory-0.18.3.md)
- [Source identity and provenance monitoring (0.18.0)](design-source-provenance-0.18.0.md)
- [CogKura Core 0.18.4 — Source provenance architecture freeze](design-source-provenance-architecture-freeze-0.18.4.md)
- [Corrective design: withdrawn readmission in current context](design-withdrawn-readmission-context-correction.md)

### Interference and inhibition (0.17)

- [Co-consumption inhibition hardening (0.17.6.1)](design-co-consumption-inhibition-hardening-0.17.6.1.md)
- [Behavioural competition eligibility and retrieval-scope isolation (0.17.2)](design-competition-behavioral-eligibility-0.17.2.md)
- [Competition hardening and transient interference (0.17.1)](design-competition-hardening-interference-0.17.1.md)
- [Competition representation and diagnostics (0.17.0)](design-competition-representation-0.17.0.md)
- [Persistent inhibition recovery and retrieval application (0.17.4)](design-inhibition-application-0.17.4.md)
- [Interference and inhibition architecture freeze (0.17.7)](design-interference-inhibition-architecture-freeze-0.17.7.md)
- [Interference observability and metamemory (0.17.5)](design-interference-observability-0.17.5.md)
- [Retrieval-induced inhibitory traces (0.17.3)](design-retrieval-inhibition-0.17.3.md)
- [Transient interference reachability (0.17.6)](design-transient-interference-reachability-0.17.6.md)

### Contextual memory (0.16)

- [Cogkura Core 0.16.5 — Context Hardening and Architecture Freeze](design-context-hardening-0.16.5.md)
- [Context observability and metamemory (0.16.4)](design-context-observability-0.16.4.md)
- [Context reinstatement (0.16.2)](design-context-reinstatement-0.16.2.md)
- [Encoding context — 0.16.0](design-encoding-context-0.16.0.md)
- [Retrieval context and context matching — 0.16.1](design-retrieval-context-0.16.1.md)
- [Semantic support-context propagation (0.16.3)](design-semantic-support-context-0.16.3.md)

### Earlier design references

- [CogKura 0.15.0 - Application integration and memory context](design-application-integration-memory-context-0.15.0.md)
- [Conjunctive slot matching and positive structured ranking](design-conjunctive-slot-ranking-0.14.3.md)
- [Gated slot admission, association, and metamemory design reference](design-gated-slot-admission.md)
- [Ranking, time, and current-state design reference](design-ranking-time-current-state.md)
- [Retrieval corrections and ranking separation design reference](design-retrieval-corrections-0.14.1.md)
- [CogKura 0.14.4 - Retrieval diagnostics and SUPPORT provenance](design-retrieval-diagnostics-support-provenance-0.14.4.md)
- [Retrieval eligibility, global ranking, and temporal relevance design reference](design-retrieval-eligibility-ranking.md)
- [Semantic consolidation design reference](design-semantic-consolidation.md)
- [String cues, candidate generation, and current-state design reference](design-string-cues-current-state.md)
- [Temporal mode, structured slot fit, and metamemory answerability](design-temporal-slot-answerability-0.14.2.md)

## Reviews and historical findings

- [0.19.5 candidate review](review-0.19.5-f289-candidate.md) and [support-evidence ordering review](review-0.19.5-support-evidence-order.md).
- [0.18.3 source-inspection ordering review](review-source-inspection-ordering-0.18.3.md).
- The [findings directory](findings/) contains versioned investigations, historical performance measurements and freeze records. Use each finding's date, candidate and stated limits when interpreting it.
