# Scoring and candidate-population contract

Status: Proposed for review; not approved and not implemented as a new policy.  
Date: 20 November 2026  
Candidate investigated: Core `d5c009faae3e90d63c44196b2e91c6c77412ef9b` (0.20.1).

## Purpose and decision

Specify which scoring properties can be expected when the visible candidate population changes. The correction-history fix correctly reveals provenance and can remove previously eligible stale evidence. The resulting change in the IDF population affects scores even when retrieved identities and ranks are unchanged. The three frozen-query differences remain unapproved; this proposal does not change the release baseline or approve them.

Retain the current default candidate-dependent IDF for the 0.20.1 candidate while investigating these differences. Disabling candidate IDF is already an opt-in configuration, but is not a behavior-preserving repair: the controlled comparison changed 361 of 393 query signatures, 25 retrieved-item sequences, eight query metric outputs and six recorded-use runs. See the [Bench findings](../../cogkura-bench/docs/findings/scoring-population-experiment.md) for exact scope and artifacts. No new scoring policy or frozen result is accepted by this document.

## Proposed contract

* A scoring run is identified by the Core version, activation configuration, cue, tenant, relevant time parameters, candidate and its available evidence, and the eligible candidate population passed to activation. Given identical inputs and state, its deterministic components should be reproducible. Record configuration and version with comparisons.
* The default lexical weights are population-dependent. IDF is computed from the candidate corpus presented to the activator: for a token, `log((N + 1) / (df + 1)) + 1`, subject to the implementation's cue-specific scaling. A visibility or correction change can therefore alter the lexical component for an otherwise unchanged candidate. Such changes are observable behavior and require review; identical retrieved ranks do not make changed scores equivalent.
* Candidate eligibility must respect tenant isolation, cognitive knowledge time (`as_of`), factual validity (`valid_at`), correction and withdrawal history, forgetting, and the configured candidate selection or cap. Evidence unavailable at the requested knowledge time must not affect that historical run. The investigated `as_of`-only error in the earlier candidate violates the intended temporal behavior; the local fix and its limits are described below.
* `enable_candidate_idf=False` removes candidate-derived IDF weighting from lexical overlap. For fixed cue and candidate text, that lexical calculation is independent of changes to the candidate corpus. This is a narrow property: it does not guarantee unchanged total activation, admission, interference, inhibition, rank, context selection, or subsequent learned state. It also removes the associated distinctive-token scaling and can change retrieval quality.
* Cross-backend agreement must be evaluated on full, ordered outputs, source assessments, stateful effects and metrics. Any explicitly normalized comparison is diagnostic; it does not approve order differences. Tests must cover additions and removals of eligible candidates, other tenants, both time dimensions, corrections, withdrawals, caps and persistent state.

A future fixed, versioned weighting prior could offer corpus-independent lexical weights while retaining token discrimination, but requires a separately designed and evaluated policy. A prior derived dynamically from the current candidate set would not satisfy that stability property. No prior or default change is proposed here.

## Historical recall defect

In 0.20.0 and 0.20.1, after a same-tenant later episode is encoded, an earlier `as_of`-only recall raises `ValidationError: Memory reference cannot occur after the activation timestamp.` The probe reproduces this with both IDF settings on memory, SQLite and PostgreSQL. In that particular fixture, providing the earlier `valid_at` avoids the error; the two time dimensions are not generally interchangeable.

Investigate visibility and trace handling independently of IDF, including late-arriving observations, corrections and withdrawals, and records whose factual time differs from when they became known. Do not filter solely by episode start time without establishing the historical semantics. Add a regression at the correct ownership layer and verify all backends and both time axes before determining its release impact. The Core historical-candidate tests cover the fix, including the conservative treatment of mixed-time episodes described below.

## Evaluation required for any alternative

Compare exact score components, ranking and selected context, source/provenance, query and aggregate quality metrics, and recorded-use/inhibition outcomes against a specified baseline. Include representative datasets and time-boundary cases, all storage backends and candidate-limit effects. Review any expected differences explicitly for the exact candidate and preserve raw evidence. Performance measurements must use the agreed reference Mac, workload sizes, sample counts and targets; the small exploratory IDF timing comparison is not a release performance gate.

The existing three 0.20.1 frozen-query differences still need resolution or explicit approval after further investigation. This design does not authorize release, a rebaseline, or a change to provenance reporting.

## Fix under validation — 20 November 2026

The local fix excludes a candidate from historical activation when its current episode or semantic support traces include evidence after `as_of`. It applies before candidate-based IDF, forgetting and inspection limits. `valid_at` remains an independent factual-time filter, and known facts with future validity can still be queried. Since the stored current episode and aggregate semantic rows are not complete historical snapshots, a mixed-time episode or semantic candidate is omitted as a whole rather than partially reconstructed. This avoids using its future content, but can omit earlier known evidence; full historical reconstruction remains separate work. This scoped fix does not establish historical isolation of every other learned or relational state.
