"""Unit tests for observation hashing."""

from cogkura.observations.hashing import content_hash, normalize_content


def test_normalize_content_strips_trailing_whitespace() -> None:
    assert normalize_content("hello  \n") == "hello"


def test_content_hash_is_deterministic() -> None:
    assert content_hash("PostgreSQL") == content_hash("PostgreSQL")


def test_content_hash_differs_for_different_content() -> None:
    assert content_hash("alpha") != content_hash("beta")
