"""Package-specific exceptions for Cogkura."""


class CogkuraError(Exception):
    """Base exception for Cogkura."""


class ValidationError(CogkuraError):
    """Raised when user input or model data is invalid."""


class StorageError(CogkuraError):
    """Raised when a storage operation fails."""


class CandidateSetTooLargeError(CogkuraError):
    """Raised when recall candidate set exceeds configured maximum."""


class RecallInspectionUnsupportedError(CogkuraError):
    """Raised when the configured declarative activator does not support inspection."""
