"""Package-specific exceptions for Cognema."""


class CognemaError(Exception):
    """Base exception for Cognema."""


class ValidationError(CognemaError):
    """Raised when user input or model data is invalid."""


class StorageError(CognemaError):
    """Raised when a storage operation fails."""


class CandidateSetTooLargeError(CognemaError):
    """Raised when recall candidate set exceeds configured maximum."""
