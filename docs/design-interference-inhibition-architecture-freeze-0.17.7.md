# Interference and inhibition architecture freeze (0.17.7)

Shipped in Cogkura `0.17.7`.

This release freezes the `0.17` interference and inhibition line. It does not add a mechanism, and it does not change formulas, weights, eligibility, defaults, or schema. Retrieval behaviour matches `0.17.6.1`. `apply_interference` and `apply_to_recall` stay `false`.

## Frozen flow

1. Context reinstatement adjusts accessibility.
2. Persistent inhibition, when `apply_to_recall` is true, applies a scope-matched penalty before transient interference.
3. Competition identifies rivals. Behavioural eligibility is the gate that decides which rivals may affect retrieval. Diagnostic competition can exist while `scope_eligible` is false.
4. Transient interference, when `apply_interference` is true, applies a retrieval-local, negative-only proactive or retroactive penalty. Co-temporal pairs can be scope-eligible and still penalty-free.
5. The retrieval threshold and working-memory selection run on ordinary recall results.

`recall`, `inspect_recall`, `assess_memory`, and `prepare_context` write no traces. Explicit use writes reinforcement and inhibitory traces as separate records. Consumed memories and reinforced memories stay distinct under burst limiting and `min_score`. Same-slot membership is not co-consumption. Co-consumption follows active semantic lineage.

`valid_at` is fact time. `as_of` is the inhibition clock. Forgetting lifecycle is not written by interference. `ActivationComponents.inhibition` stays separate from the transient penalty, and both are `<= 0`.

## Contracts

[`tests/test_interference_inhibition_architecture_freeze.py`](../tests/test_interference_inhibition_architecture_freeze.py) locks those boundaries in memory. [`tests/test_postgres_inhibition_store.py`](../tests/test_postgres_inhibition_store.py) checks that a stored trace changes recall only when `apply_to_recall` is true. Co-consumption remains an induction rule, not a storage rule. No new migration.

Weights, floors, and the recovery half-life are engineering parameters, not biological constants. Documented defaults match `CompetitionConfig()` and `InhibitionConfig()`.

## Deferred

Trace compaction, automatic deletion, cancellation, default-on application, and adaptive or model-based competition.

A behavioural change after this release needs a new design note.

## Bench

CogKuraBench was not rerun from this repository. The recorded evidence is in [`findings/0.17.7-interference-inhibition-architecture-freeze.md`](findings/0.17.7-interference-inhibition-architecture-freeze.md). `transient_interference_v1` is historical and is not a freeze failure.
