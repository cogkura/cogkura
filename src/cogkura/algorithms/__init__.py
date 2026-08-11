"""Cognitive algorithm modules."""

from cogkura.algorithms.activation import ACTRDeclarativeActivator, DeclarativeActivator
from cogkura.algorithms.episodic import DeterministicEpisodicEncoder, EpisodicEncoder
from cogkura.algorithms.forgetting import (
    EbbinghausForgettingEvaluator,
    ForgettingEvaluator,
)
from cogkura.algorithms.semantic import (
    ComplementaryLearningSemanticConsolidator,
    MetadataSemanticExtractor,
    SemanticConsolidator,
    SemanticExtractor,
)
from cogkura.algorithms.spreading import (
    DeterministicSpreadingActivator,
    SpreadingActivator,
)
from cogkura.algorithms.working_memory import (
    ApproximateTokenEstimator,
    DeterministicWorkingMemorySelector,
    TokenEstimator,
    WorkingMemorySelector,
)

__all__ = [
    "ACTRDeclarativeActivator",
    "ApproximateTokenEstimator",
    "ComplementaryLearningSemanticConsolidator",
    "DeclarativeActivator",
    "DeterministicEpisodicEncoder",
    "DeterministicSpreadingActivator",
    "DeterministicWorkingMemorySelector",
    "EbbinghausForgettingEvaluator",
    "EpisodicEncoder",
    "ForgettingEvaluator",
    "MetadataSemanticExtractor",
    "SemanticConsolidator",
    "SemanticExtractor",
    "SpreadingActivator",
    "TokenEstimator",
    "WorkingMemorySelector",
]
