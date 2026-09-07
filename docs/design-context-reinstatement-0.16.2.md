# Context reinstatement (0.16.2)

**Status:** Shipped

## Summary

Cogkura `0.16.2` converts frozen `0.16.1` `ContextMatch` evidence into a bounded episodic activation contribution during declarative scoring. Reinstatement adjusts **accessibility**, not relevance: it does not generate candidates, filter storage, or bypass admission gates.

## Formula

\[
R_i = M_i \times V_i \qquad C_i = \lambda_{ctx} \times R_i \qquad A_i' = A_i + C_i
\]

| Symbol | Source |
|--------|--------|
| \(M_i\) | `ContextMatch.score` (`0` when `None`) |
| \(V_i\) | `ContextMatch.cue_coverage` |
| \(\lambda_{ctx}\) | `ActivationConfig.context_reinstatement_weight` |
| \(C_i\) | `ContextReinstatement.activation_contribution` |

## Policy reasons

| Reason | Contribution |
|--------|--------------|
| `applied` | `weight × strength` |
| `disabled` | `0` (diagnostics retain strength) |
| `not_episodic` | `0` |
| `no_retrieval_context` | `0` |
| `no_comparable_context` | `0` |
| `zero_match` | `0` |

## Invariants

- `ContextMatch` semantics from `0.16.1` unchanged (`tests/test_context_matching.py`).
- Episodes only; semantic candidates keep `context_match=None`.
- Positive-only: mismatch/missing never penalise.
- No storage migration; matches and reinstatement are not persisted.
- No-context and `weight=0` reproduce `0.16.1` retrieval behaviour.

## Public types

- `ContextReinstatement`, `ContextReinstatementReason`
- `DeterministicContextReinstatementPolicy`
- `ActivationConfig.context_reinstatement_weight`
- `ActivationComponents.context_reinstatement`
- `RetrievalDiagnostics.context_reinstatement`, `activation_before_context`

## Tests

- `tests/test_context_reinstatement.py` — policy unit tests
- `tests/test_016_2_context_reinstatement.py` — behaviour and regression
- `tests/test_016_1_retrieval_context_neutrality.py` — frozen match diagnostics

## Calibration

See [`findings/0.16.2-context-reinstatement-calibration.md`](findings/0.16.2-context-reinstatement-calibration.md).
