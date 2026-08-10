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

__all__ = [
    "ACTRDeclarativeActivator",
    "ComplementaryLearningSemanticConsolidator",
    "DeclarativeActivator",
    "DeterministicEpisodicEncoder",
    "DeterministicSpreadingActivator",
    "EbbinghausForgettingEvaluator",
    "EpisodicEncoder",
    "ForgettingEvaluator",
    "MetadataSemanticExtractor",
    "SemanticConsolidator",
    "SemanticExtractor",
    "SpreadingActivator",
]
