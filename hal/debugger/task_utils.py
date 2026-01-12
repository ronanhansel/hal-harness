from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

from datasets import load_dataset

LOGGER = logging.getLogger(__name__)

DEFAULT_BENCHMARK = "swebench_verified"

_BENCHMARK_DATASETS: Dict[str, Tuple[str, str]] = {
    "swebench_verified": ("princeton-nlp/SWE-bench_Verified", "test"),
    "swebench_verified_mini": ("princeton-nlp/SWE-bench_Verified", "test"),
}


def _normalize_benchmark_name(benchmark_name: str | None) -> str:
    """Map user provided benchmark names to the internal keys used here."""
    if not benchmark_name:
        return DEFAULT_BENCHMARK

    normalized = benchmark_name.lower()
    if "mini" in normalized:
        return "swebench_verified_mini"
    if "verified" in normalized:
        return "swebench_verified"
    return normalized


@lru_cache(maxsize=1)
def _load_mini_instance_ids() -> set[str]:
    """Read the curated list of SWE-Bench mini ids if it exists."""
    ids_file = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "benchmarks"
        / "swebench_verified_mini_task_ids.txt"
    )
    if not ids_file.exists():
        return set()

    with ids_file.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


@lru_cache(maxsize=None)
def _load_dataset_index(benchmark_name: str) -> Dict[str, Dict[str, Any]]:
    """Load the requested benchmark split and build an index by task id."""
    if benchmark_name not in _BENCHMARK_DATASETS:
        raise ValueError(f"Unsupported benchmark '{benchmark_name}'")

    dataset_name, split = _BENCHMARK_DATASETS[benchmark_name]
    LOGGER.debug("Loading dataset %s (%s)", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)

    if benchmark_name == "swebench_verified_mini":
        mini_ids = _load_mini_instance_ids()
    else:
        mini_ids = None

    index: Dict[str, Dict[str, Any]] = {}
    for row in dataset:
        instance_id = row.get("instance_id")
        if not instance_id:
            continue
        if mini_ids and instance_id not in mini_ids:
            continue
        index[instance_id] = dict(row)
    return index


def get_task_data(task_id: str, benchmark_name: str | None) -> Dict[str, Any]:
    """
    Fetch a specific SWE-Bench task by id.

    Returns a dictionary that contains at least the keys:
    problem_statement, repo, base_commit, and instance_id.
    """
    normalized_benchmark = _normalize_benchmark_name(benchmark_name)
    dataset_index = _load_dataset_index(normalized_benchmark)

    if task_id not in dataset_index:
        raise KeyError(
            f"Task id '{task_id}' not found in benchmark '{normalized_benchmark}'"
        )

    task = dataset_index[task_id]
    problem_statement = task.get("problem_statement") or task.get("prompt")
    base_commit = task.get("base_commit") or task.get("environment_setup_commit")

    task_data: Dict[str, Any] = {
        "instance_id": task.get("instance_id", task_id),
        "problem_statement": problem_statement,
        "repo": task.get("repo"),
        "base_commit": base_commit,
    }

    if "environment_setup_commit" in task:
        task_data["environment_setup_commit"] = task["environment_setup_commit"]

    return task_data
