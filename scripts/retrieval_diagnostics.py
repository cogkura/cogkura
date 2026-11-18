"""Opt-in retrieval timings for the external profiling script.

Timings overlap, particularly during asyncio.gather; they are not additive.
Patches exist only for the duration of a single measured recall.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import cogkura.memory as memory_module
from cogkura import Memory
from cogkura.algorithms import activation


@contextmanager
def stage_diagnostics(memory: Memory) -> Iterator[dict[str, dict[str, Any]]]:
    """Measure coarse stages and returned populations without changing Core APIs."""
    stages: dict[str, dict[str, Any]] = {}

    def install(stack: ExitStack, target: Any, name: str, label: str) -> None:
        original = getattr(target, name)
        row = stages.setdefault(label, {"calls": 0, "wall_seconds": 0.0, "returned_items": 0})

        def finish(
            start: float, result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            row["calls"] += 1
            row["wall_seconds"] += time.perf_counter() - start
            if isinstance(result, (list, tuple, dict, set, frozenset)):
                row["returned_items"] += len(result)
            if "candidates" in kwargs and isinstance(kwargs["candidates"], (list, tuple)):
                row["input_candidates"] = row.get("input_candidates", 0) + len(kwargs["candidates"])
            if label == "historical_projection" and len(args) > 2:
                row["input_revisions"] = row.get("input_revisions", 0) + len(args[2])
            if label == "competition_and_inhibition" and isinstance(result, tuple):
                evaluation = result[2]
                row["competition_enabled"] = evaluation is not None
                if evaluation is not None:
                    row["potential_pairs"] = evaluation.run.potential_competitor_pairs
                    row["evaluated_pairs"] = evaluation.run.evaluated_competitor_pairs
                    row["accepted_pairs"] = evaluation.run.accepted_competition_pairs

        if inspect.iscoroutinefunction(original):

            async def asynchronous(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = await original(*args, **kwargs)
                finish(start, result, args, kwargs)
                return result

            stack.enter_context(patch.object(target, name, asynchronous))
        else:

            def synchronous(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = original(*args, **kwargs)
                finish(start, result, args, kwargs)
                return result

            stack.enter_context(patch.object(target, name, synchronous))

    with ExitStack() as stack:
        for store, method, label in (
            (memory._episode_store, "list", "episode_loading"),
            (memory._semantic_store, "list", "semantic_loading"),
            (memory._semantic_store, "list_revisions", "revision_loading"),
            (memory._entity_relationship_store, "list", "relationship_loading"),
        ):
            install(stack, type(store), method, label)
        for name, label in (
            ("_rank_declarative_results", "recall_pipeline"),
            ("_filter_recallable_candidates", "forgetting_eligibility"),
            ("_list_activation_traces", "activation_trace_loading"),
            ("_load_learned_associations", "learned_association_loading"),
        ):
            install(stack, Memory, name, label)
        for name, label in (
            ("project_known_semantics", "historical_projection"),
            ("selectively_filter_stale_episode_content", "stale_episode_filtering"),
            ("build_activation_candidates", "candidate_construction"),
        ):
            install(stack, memory_module, name, label)
        install(stack, activation.ACTRDeclarativeActivator, "rank_evaluated", "scoring_and_ranking")
        for name, label in (
            ("_authoritative_semantic_current_admission_identities", "semantic_admission"),
            ("apply_competition_pipeline", "competition_and_inhibition"),
        ):
            install(stack, activation, name, label)
        yield stages
