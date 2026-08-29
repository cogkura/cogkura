"""Unit tests for canonical retrieval feature extraction."""

from __future__ import annotations

from cogkura.algorithms.retrieval_features import (
    canonical_content_features,
    distinctive_content_features,
    feature_overlap,
    normalize_and_tokenize,
    predicate_content_features,
)


def test_stopwords_removed_from_content_features() -> None:
    features = canonical_content_features("Could you recommend a backpack for me?")
    assert "a" not in features
    assert "for" not in features
    assert "you" not in features
    assert "me" not in features
    assert "recommend" in features
    assert "backpack" in features


def test_contractions_do_not_produce_i_and_m() -> None:
    features = canonical_content_features("I'm looking for a jacket.")
    assert "i" not in features
    assert "m" not in features
    assert "jacket" in features


def test_jacket_jackets_normalize_to_same_feature() -> None:
    singular = canonical_content_features("recommend a jacket")
    plural = canonical_content_features("prefers neutral jackets")
    assert "jacket" in singular
    assert "jacket" in plural
    _, _, _, matched = feature_overlap(singular, plural)
    assert "jacket" in matched


def test_punctuation_variants_normalize_consistently() -> None:
    hyphen = canonical_content_features("waterproof-shell")
    spaced = canonical_content_features("waterproof shell")
    _, _, _, matched = feature_overlap(hyphen, spaced)
    assert "waterproof" in matched
    assert "shell" in matched


def test_colour_color_orthographic_normalization() -> None:
    british = canonical_content_features("bright colours")
    american = canonical_content_features("bright colors")
    _, _, _, matched = feature_overlap(british, american)
    assert "color" in matched


def test_predicate_underscore_splits() -> None:
    features = predicate_content_features("outerwear_weight_preference")
    assert "outerwear" in features
    assert "weight" in features
    assert "preference" in features


def test_distinctive_excludes_current_state_tokens() -> None:
    features = distinctive_content_features(
        "What is the current jacket size?",
        exclude_tokens=frozenset({"current"}),
    )
    assert "current" not in features
    assert "jacket" in features


def test_normalize_and_tokenize_preserves_temporal_intent_tokens() -> None:
    tokens = normalize_and_tokenize("What happened before yesterday?")
    assert "before" in tokens
    assert "yesterday" in tokens
    assert "happened" in tokens


def test_structured_object_value_splits_on_colon() -> None:
    features = canonical_content_features("northpeak-alpine-shell:sleeves_too_short")
    assert "northpeak" in features
    assert "alpine" in features
    assert "shell" in features
    assert "sleeve" in features
