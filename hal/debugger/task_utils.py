from __future__ import annotations

import logging
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

LOGGER = logging.getLogger(__name__)

DEFAULT_BENCHMARK = "swebench_verified"

_BENCHMARK_DATASETS: Dict[str, Tuple[str, str]] = {
    "swebench_verified": ("princeton-nlp/SWE-bench_Verified", "test"),
    "swebench_verified_mini": ("princeton-nlp/SWE-bench_Verified", "test"),
}

COREBENCH_VARIANTS = {"corebench_easy", "corebench_medium", "corebench_hard"}
COREBENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "corebench"
COREBENCH_JSON = COREBENCH_DIR / "core_test.json"
COREBENCH_GPG = COREBENCH_DIR / "core_test.json.gpg"


def _normalize_benchmark_name(benchmark_name: str | None) -> str:
    """Map user provided benchmark names to the internal keys used here."""
    if not benchmark_name:
        return DEFAULT_BENCHMARK

    normalized = benchmark_name.lower()
    if normalized.startswith("corebench"):
        if normalized in COREBENCH_VARIANTS:
            return normalized
        if normalized == "corebench":
            return "corebench_hard"
        if normalized.startswith("corebench_hard"):
            return "corebench_hard"
        if normalized.startswith("corebench_medium"):
            return "corebench_medium"
        if normalized.startswith("corebench_easy"):
            return "corebench_easy"
        return "corebench_hard"

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
    if benchmark_name in COREBENCH_VARIANTS:
        return _load_corebench_index(benchmark_name)

    if benchmark_name not in _BENCHMARK_DATASETS:
        raise ValueError(f"Unsupported benchmark '{benchmark_name}'")

    from datasets import load_dataset  # type: ignore

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


@lru_cache(maxsize=None)
def _load_corebench_json() -> Any:
    if not COREBENCH_JSON.exists():
        raise FileNotFoundError(
            f"CoreBench metadata not found at {COREBENCH_JSON}. "
            f"Decrypt {COREBENCH_GPG} via "
            f"`gpg --output {COREBENCH_JSON} --decrypt {COREBENCH_GPG}` "
            'using password "reproducibility".'
        )
    with COREBENCH_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def _load_corebench_index(benchmark_name: str) -> Dict[str, Dict[str, Any]]:
    tasks = _load_corebench_json()
    index: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        capsule_id = task.get("capsule_id")
        if not capsule_id:
            continue
        prompt = _build_corebench_prompt(benchmark_name, task)
        results_schema = _extract_result_keys(task)
        index[capsule_id] = {
            "instance_id": capsule_id,
            "problem_statement": prompt,
            "repo": task.get("repo") or task.get("repository_url"),
            "base_commit": task.get("commit"),
            "capsule_id": capsule_id,
            "task_prompt": task.get("task_prompt"),
            "results_schema": results_schema,
        }
    return index


def _extract_result_keys(task: Dict[str, Any]) -> list[str]:
    results = task.get("results") or []
    if not results:
        return []
    first = results[0]
    if isinstance(first, dict):
        return list(first.keys())
    return []


def _build_corebench_prompt(benchmark_name: str, task: Dict[str, Any]) -> str:
    questions = _extract_result_keys(task)
    if questions:
        questions_text = ", ".join(questions)
    else:
        questions_text = "the questions defined in the capsule results schema"

    task_prompt = task.get("task_prompt") or "follow the capsule instructions"
    if benchmark_name == "corebench_easy":
        return (
            "Task: codeocean_easy\n\n"
            "Your goal is to answer questions about the output of scientific code. "
            "Inspect the files in the `environment/results` directory to answer the following questions: "
            f"{questions_text}. "
            "Return a Python dictionary whose keys exactly match these questions and whose values are the answers. "
            "Do not execute any code; everything you need is already in the results directory."
        )

    if benchmark_name == "corebench_medium":
        return (
            "Task: codeocean_medium\n\n"
            "Your goal is to test the computational reproducibility of the repository cloned to your workspace. "
            f"Specifically, {task_prompt} to answer the following questions: {questions_text}. "
            "Return a Python dictionary whose keys exactly match these questions and whose values are the answers. "
            "Consult REPRODUCING.md inside the capsule for detailed run instructions."
        )

    # Default to hard instructions.
    return (
        "Task: codeocean_hard\n\n"
        "Your goal is to test the computational reproducibility of the repository cloned to your workspace. "
        f"Specifically, {task_prompt} to answer the following questions: {questions_text}. "
        "Return a Python dictionary whose keys exactly match these questions and whose values are the answers. "
        "Install the requirements documented in the repository and run the necessary commands to gather the answers."
    )


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
