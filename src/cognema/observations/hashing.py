"""Content hashing for observations."""

from __future__ import annotations

import hashlib
import unicodedata


def normalize_content(content: str) -> str:
    """Normalize content for hashing: NFC + strip trailing whitespace."""
    return unicodedata.normalize("NFC", content).rstrip()


def content_hash(content: str) -> str:
    """Return sha256 hex digest of normalized UTF-8 content."""
    normalized = normalize_content(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
