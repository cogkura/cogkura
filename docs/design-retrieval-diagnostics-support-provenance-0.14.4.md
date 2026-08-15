# CogKura 0.14.4 - Retrieval diagnostics and SUPPORT provenance

## Summary

CogKura `0.14.4` is a diagnostics-first retrieval patch. The release keeps scoring behavior from `0.14.3` and adds structured visibility into why candidates were eligible, how they ranked, and which semantic revisions SUPPORT episodes derive from.

## Added

- Optional `RecallResult.diagnostics` payload with:
  - accessibility activation vs final rank activation;
  - accessibility and ranking partial terms;
  - text coverage and cue fit;
  - temporal mode;
  - structured slot fit and rank adjustment;
  - threshold vs soft-admission eligibility;
  - semantic slot/status and observation evidence ids;
  - derivation-backed SUPPORT provenance and selected supporting revision.
- `build_episode_support_provenance_index(...)` for per-episode semantic derivation context without storage changes.

## Behavioral constraints

- `RecallResult.activation` stays the accessibility value.
- `RecallResult.score` stays the presentation score over accessibility vs threshold.
- Admission, threshold semantics, stale-support filtering, and global ranking behavior are preserved.
- No new ranking weights, support boosts, fast lanes, migrations, or benchmark-specific rules.

## Notes

- Diagnostics are optional and backward compatible.
- Existing reason strings are still emitted for callers that read `RecallResult.reason`.
