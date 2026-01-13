from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hal.debugger.inspector_pipeline import InspectorPipeline
from hal.debugger.log_reader import LogIngester
from hal.debugger.pipeline_logger import PipelineRunLogger
from hal.debugger.runner_pipeline import RunnerPipeline

LOGGER = logging.getLogger("hal.auto_debug")
HAL_HARNESS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HAL_HARNESS_DIR.parent.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the automated trace-to-fix debugger over a batch of failures.",
    )
    parser.add_argument("--rubrics-csv", help="Path to a single rubric CSV (legacy mode)")
    parser.add_argument(
        "--rubrics-output-dir",
        default="../rubrics_output",
        help="Directory containing rubric CSV files (process all). Default: ../rubrics_output",
    )
    parser.add_argument(
        "--fixable-output-dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--traces-dir", required=True, help="Directory that stores trace folders for each model run")
    parser.add_argument("--agent-dir", required=True, help="Directory that contains the agent entrypoint (main.py)")
    parser.add_argument("--agent-args", help="Optional JSON string or path to a JSON file with agent kwargs")
    parser.add_argument("--agent-function", help="Override the agent function path (default: main.run)")
    parser.add_argument("--benchmark-name", help="Override target benchmark (default: swebench_verified)")
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Limit processing to a specific task_id (may be provided multiple times)",
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "run"],
        default="inspect",
        help="Choose 'inspect' to generate AutoInspector reports or 'run' to execute fix packages.",
    )
    parser.add_argument(
        "--trace-output-dir",
        default="../traces/debug_runs",
        help="Directory where synthetic rerun traces will be written (mode=run).",
    )
    parser.add_argument(
        "--fixed-output",
        action="store_true",
        help="Store inspection/runner outputs in deterministic paths (overwrites previous results).",
    )
    parser.add_argument(
        "--run-label",
        default="latest",
        help="Folder name used when --fixed-output is enabled (default: latest).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="Set reasoning_effort when calling the inspector's LLM.",
    )
    parser.add_argument(
        "--inspector-model",
        help="Override the AutoInspector LLM model (defaults to HAL_AUTOFIX_MODEL env).",
    )
    return parser.parse_args()


def load_agent_args(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    candidate = Path(payload)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(payload)


def load_failures(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[str]]:
    failures: List[Dict[str, Any]] = []
    sources: List[str] = []
    traces_dir = args.traces_dir

    rubric_dir = args.rubrics_output_dir or args.fixable_output_dir
    if rubric_dir:
        root = Path(rubric_dir)
        if not root.exists():
            raise FileNotFoundError(f"Rubric output directory not found: {root}")
        csv_files = sorted(root.rglob("*.csv"))
        if not csv_files:
            LOGGER.warning("No CSV files found in %s", root)
        for csv_file in csv_files:
            LOGGER.info("Loading failures from %s", csv_file)
            ingester = LogIngester(csv_file, traces_dir, allowed_criteria=None)
            items = ingester.get_failing_tasks()
            for item in items:
                item["_source_csv"] = str(csv_file)
            failures.extend(items)
            sources.append(str(csv_file))
        return _filter_failures(failures, args.task_ids), sources

    if args.rubrics_csv:
        LOGGER.info("Loading failures from %s", args.rubrics_csv)
        ingester = LogIngester(args.rubrics_csv, traces_dir)
        items = ingester.get_failing_tasks()
        for item in items:
            item["_source_csv"] = str(args.rubrics_csv)
        return _filter_failures(items, args.task_ids), [str(args.rubrics_csv)]

    raise ValueError("Provide either --rubrics-csv or --rubrics-output-dir")


def _filter_failures(
    failures: List[Dict[str, Any]], task_ids: Optional[List[str]]
) -> List[Dict[str, Any]]:
    if not task_ids:
        return failures
    desired = {tid.strip() for tid in task_ids if tid and tid.strip()}
    filtered = [item for item in failures if item.get("task_id") in desired]
    missing = desired - {item.get("task_id") for item in filtered}
    if missing:
        LOGGER.warning(
            "Requested --task-id entries not present in the rubric data: %s",
            ", ".join(sorted(missing)),
        )
    LOGGER.info(
        "Filtered rubric entries to %d task(s) (requested %d).",
        len(filtered),
        len(desired),
    )
    return filtered


def _log_failure_summary(items: List[Dict[str, Any]], sources: List[Path | str]) -> None:
    if not items:
        LOGGER.info("No rubric entries found across %d source file(s).", len(sources))
        return
    criteria_counts: Dict[str, int] = {}
    for item in items:
        criteria = (item.get("criteria") or "unknown").lower()
        criteria_counts[criteria] = criteria_counts.get(criteria, 0) + 1
    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(criteria_counts.items()))
    LOGGER.info(
        "Loaded %d rubric task(s) across %d source file(s). Breakdown -> %s",
        len(items),
        len(sources),
        breakdown or "n/a",
    )
    grade_one = [
        item.get("task_id")
        for item in items
        if float(item.get("grade_numeric") or item.get("grade") or 0) >= 1.0
    ]
    if grade_one:
        LOGGER.info("Grade=1.0 task_ids: %s", ", ".join(sorted(set(filter(None, grade_one)))))

def _resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (HAL_HARNESS_DIR / path).resolve()


def _build_pipeline(args: argparse.Namespace, agent_args: Optional[Dict[str, Any]], rerun_command: str):
    output_root = (REPO_ROOT / "debug").resolve()
    fixes_root = (REPO_ROOT / "fixes").resolve()
    trace_output_root = _resolve_path(args.trace_output_dir)
    common_kwargs = {
        "agent_dir": args.agent_dir,
        "agent_args": agent_args,
        "agent_function": args.agent_function,
        "benchmark_name": args.benchmark_name,
        "rerun_command": rerun_command,
        "output_root": output_root,
        "fixes_root": fixes_root,
        "fixed_layout": args.fixed_output,
        "run_label": args.run_label,
    }
    if args.mode == "inspect":
        return InspectorPipeline(
            **common_kwargs,
            reasoning_effort=args.reasoning_effort,
            inspector_model=getattr(args, "inspector_model", None),
        )
    return RunnerPipeline(
        **common_kwargs,
        trace_output_root=trace_output_root,
    )


async def _run_pipeline(args: argparse.Namespace, rerun_command: str) -> None:
    failures, sources = load_failures(args)
    pipeline_logger = PipelineRunLogger()
    pipeline_logger.log(f"Mode={args.mode}, requested_tasks={len(args.task_ids or [])}, loaded_tasks={len(failures)}")

    agent_args = load_agent_args(args.agent_args)
    pipeline = _build_pipeline(args, agent_args, rerun_command)

    if args.fixed_output:
        summary_root = pipeline.results_root / pipeline.benchmark_name
    else:
        summary_root = pipeline.results_root
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "rubric_summary.json"
    write_rubric_summary(failures, summary_path)
    _log_failure_summary(failures, sources)
    LOGGER.info("Wrote rubric summary to %s", summary_path)
    pipeline_logger.log(f"Rubric summary updated at {summary_path}")

    if not failures:
        LOGGER.info("No rubric entries detected. Nothing to process.")
        pipeline_logger.log("No rubric entries detected; exiting.")
        pipeline_logger.finish()
        print(f"Pipeline logs: {pipeline_logger.root}")
        return

    LOGGER.info("Starting %s pipeline for %d task(s)", args.mode, len(failures))
    pipeline_logger.log(f"Starting {args.mode} pipeline for {len(failures)} task(s).")
    await pipeline.run_batch(failures)
    LOGGER.info("Debug run complete")
    pipeline_logger.log("Pipeline run complete.")
    pipeline_logger.finish()
    print(f"Pipeline logs: {pipeline_logger.root}")


def write_rubric_summary(failures: List[Dict[str, Any]], output_path: Path) -> None:
    payload_tasks: List[Dict[str, Any]] = []
    for failure in failures:
        grade_numeric = failure.get("grade_numeric")
        if grade_numeric is None:
            try:
                grade_numeric = float(failure.get("grade") or 0)
            except (TypeError, ValueError):
                grade_numeric = 0.0

        payload_tasks.append(
            {
                "task_id": failure.get("task_id"),
                "criteria": failure.get("criteria"),
                "grade": grade_numeric,
                "grade_raw": failure.get("grade"),
                "explanation": failure.get("explanation"),
                "source_csv": failure.get("_source_csv"),
                "model_run": failure.get("model_run"),
            }
        )

    payload_tasks.sort(key=lambda item: (item.get("task_id") or "", item.get("criteria") or ""))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_tasks": len(failures),
        "grade_one_tasks": [
            task["task_id"] for task in payload_tasks if task.get("grade") == 1.0 and task.get("task_id")
        ],
        "tasks": payload_tasks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    rerun_command = "python scripts/auto_debug_batch.py " + " ".join(sys.argv[1:])
    asyncio.run(_run_pipeline(args, rerun_command))


if __name__ == "__main__":
    main()
