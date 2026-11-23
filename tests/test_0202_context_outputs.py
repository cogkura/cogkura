"""Check context outputs independently on the mixed project fixture."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    LearningConfig,
    Memory,
    MemoryAssessmentFlag,
    RetrievalContext,
    RetrievalCue,
    SemanticCorrectionConfig,
    WorkingMemoryConfig,
)

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_sqlite_workload import _T, _run  # noqa: E402


async def _outputs(memory: Memory) -> list[dict[str, Any]]:
    await _run(memory, 100, 5, 1, mixed=True, trace_sql=False)
    at = _T + timedelta(days=2)
    relevant = "service-0006 deployment environment env-0006"
    missing = RetrievalCue(
        text="what is the primary lunar crater?",
        entity_ids=("moon",),
        predicate="primary_crater",
    )
    results = []
    for query, budget, structured, use_previous in (
        (relevant, 256, False, False),
        (relevant, 2048, True, True),
        (missing, 512, True, False),
    ):
        kwargs: dict[str, Any] = {
            "tenant_id": "benchmark",
            "subject_id": "service-0006",
            "as_of": at,
        }
        if structured:
            kwargs["goal"] = "Identify deployment environment"
            kwargs["retrieval_context"] = RetrievalContext(
                domain="deployment", activity="configuration review"
            )
        previous = await memory.select_working_memory(query, **kwargs) if use_previous else None
        context = await memory.prepare_context(
            query, **kwargs, previous=previous, prompt_budget_tokens=budget
        )
        repeated = await memory.prepare_context(
            query, **kwargs, previous=previous, prompt_budget_tokens=budget
        )
        assert repeated == context
        assert context.estimated_tokens <= budget
        assert context.assessment == await memory.assess_memory(query, **kwargs)
        if query == missing:
            assert MemoryAssessmentFlag.MISSING_KNOWLEDGE in context.assessment.flags
        else:
            assert any(
                "service-0006 deployment environment env-0006" in item.memory.statement
                for item in context.recall_results
            )
        results.append(
            {
                "rendered": context.render(),
                "recall": [
                    (item.memory_kind.value, item.memory.statement, round(item.score, 8))
                    for item in context.recall_results
                ],
                "flags": context.assessment.flags,
                "signals": context.assessment.signals,
                "counts": (
                    context.assessment.retrieved_count,
                    context.assessment.episode_count,
                    context.assessment.semantic_count,
                    context.working_memory.candidate_count,
                    context.working_memory.selected_count,
                    context.estimated_tokens,
                ),
            }
        )
    return results


@pytest.mark.asyncio
@pytest.mark.parametrize("learning", [False, True])
async def test_mixed_fixture_context_outputs_match_sqlite(tmp_path: Path, learning: bool) -> None:
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
        "working_memory_config": WorkingMemoryConfig(candidate_pool_size=20),
        "learning_config": LearningConfig(enabled=learning),
    }
    expected = await _outputs(Memory(**options))
    async with await EmbeddedMemory.open(tmp_path / "context-output.db", **options) as embedded:
        assert await _outputs(embedded) == expected
