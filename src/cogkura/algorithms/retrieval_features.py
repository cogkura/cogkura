"""Deterministic canonical retrieval features for lexical matching."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
_SEPARATOR_PATTERN = re.compile(r"[\s\-_:./]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "for",
        "to",
        "of",
        "in",
        "on",
        "with",
        "i",
        "me",
        "my",
        "we",
        "you",
        "your",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "what",
        "would",
        "could",
        "should",
        "do",
        "does",
        "did",
    }
)

_CONTRACTIONS: dict[str, str] = {
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "it's": "it is",
    "it'll": "it will",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
}

_ORTHographic_MAP: dict[str, str] = {
    "colour": "color",
    "colours": "color",
    "color": "color",
    "colors": "color",
    "grey": "gray",
    "gray": "gray",
}

_EXPLICIT_INFLECTIONS: dict[str, str] = {
    "jackets": "jacket",
    "jacket": "jacket",
    "colours": "color",
    "colour": "color",
    "colors": "color",
    "color": "color",
    "preferences": "preference",
    "preference": "preference",
    "sleeves": "sleeve",
    "sleeve": "sleeve",
    "returned": "return",
    "returning": "return",
    "return": "return",
    "returns": "return",
    "prefers": "prefer",
    "preferred": "prefer",
    "prefer": "prefer",
    "avoided": "avoid",
    "avoiding": "avoid",
    "avoid": "avoid",
    "avoids": "avoid",
    "compared": "compare",
    "comparing": "compare",
    "compare": "compare",
    "compares": "compare",
    "purchased": "purchase",
    "purchasing": "purchase",
    "purchase": "purchase",
    "purchases": "purchase",
}


def normalize_and_tokenize(text: str) -> frozenset[str]:
    """Tokenise text after Unicode, contraction, and separator normalisation."""
    if not text or not text.strip():
        return frozenset()
    normalised = unicodedata.normalize("NFKC", text)
    normalised = _WHITESPACE_PATTERN.sub(" ", normalised).strip().casefold()
    normalised = _expand_contractions(normalised)
    normalised = _SEPARATOR_PATTERN.sub(" ", normalised)
    tokens = {token for token in _TOKEN_PATTERN.findall(normalised) if token}
    return frozenset(tokens)


def canonical_content_features(text: str) -> frozenset[str]:
    """Return retrieval content features with stopword removal and morphology."""
    tokens = normalize_and_tokenize(text)
    if not tokens:
        return frozenset()
    features: set[str] = set()
    for token in tokens:
        if token in _RETRIEVAL_STOPWORDS:
            continue
        features.add(_canonicalise_token(token))
    return frozenset(features)


def canonical_features_from_parts(*parts: str | None) -> frozenset[str]:
    """Union canonical features from multiple text fragments."""
    combined: set[str] = set()
    for part in parts:
        if part and part.strip():
            combined.update(canonical_content_features(part))
    return frozenset(combined)


def predicate_content_features(predicate: str | None) -> frozenset[str]:
    """Canonical features for a structured predicate (underscores become separators)."""
    if predicate is None or not predicate.strip():
        return frozenset()
    normalised = predicate.strip().replace("_", " ")
    return canonical_content_features(normalised)


def feature_overlap(
    query_features: frozenset[str],
    target_features: frozenset[str],
) -> tuple[float, float, float, frozenset[str]]:
    """Return query coverage, target coverage, dice, and matched features."""
    if not query_features or not target_features:
        return 0.0, 0.0, 0.0, frozenset()
    matched = query_features.intersection(target_features)
    if not matched:
        return 0.0, 0.0, 0.0, frozenset()
    query_coverage = len(matched) / len(query_features)
    target_coverage = len(matched) / len(target_features)
    dice = (2.0 * len(matched)) / (len(query_features) + len(target_features))
    return query_coverage, target_coverage, dice, frozenset(matched)


def distinctive_content_features(
    text: str,
    *,
    exclude_tokens: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Canonical query content features minus configured temporal/current tokens."""
    features = canonical_content_features(text)
    if not features or not exclude_tokens:
        return features
    return frozenset(token for token in features if token not in exclude_tokens)


def _expand_contractions(text: str) -> str:
    words = text.split()
    expanded: list[str] = []
    for word in words:
        stripped = word.strip(".,!?;:'\"()[]{}")
        replacement = _CONTRACTIONS.get(stripped)
        if replacement is not None:
            expanded.append(replacement)
        else:
            expanded.append(word)
    return " ".join(expanded)


def _canonicalise_token(token: str) -> str:
    if token in _EXPLICIT_INFLECTIONS:
        token = _EXPLICIT_INFLECTIONS[token]
    return _ORTHographic_MAP.get(token, token)
