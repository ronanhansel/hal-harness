from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hal.debugger.log_reader import LogIngester
from hal.debugger.pipeline import DebugPipeline

LOGGER = logging.getLogger("hal.auto_debug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the automated trace-to-fix debugger over a batch of failures.",
    )
    parser.add_argument("--rubrics-csv", help="Path to a single rubric CSV (legacy mode)")
    parser.add_argument("--fixable-output-dir", help="Directory containing fixable rubric CSV files (process all)")
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

    if args.fixable_output_dir:
        root = Path(args.fixable_output_dir)
        if not root.exists():
            raise FileNotFoundError(f"Fixable output directory not found: {root}")
        csv_files = sorted(root.glob("*.csv"))
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

    raise ValueError("Provide either --rubrics-csv or --fixable-output-dir")


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
        LOGGER.info("No fixable rubric entries found across %d source file(s).", len(sources))
        return
    criteria_counts: Dict[str, int] = {}
    for item in items:
        criteria = (item.get("criteria") or "unknown").lower()
        criteria_counts[criteria] = criteria_counts.get(criteria, 0) + 1
    breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(criteria_counts.items()))
    LOGGER.info(
        "Loaded %d fixable task(s) across %d source file(s). Breakdown -> %s",
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


async def _run_pipeline(args: argparse.Namespace, rerun_command: str) -> None:
    failures, sources = load_failures(args)
    agent_args = load_agent_args(args.agent_args)
    pipeline = DebugPipeline(
        agent_dir=args.agent_dir,
        agent_args=agent_args,
        agent_function=args.agent_function,
        benchmark_name=args.benchmark_name,
        rerun_command=rerun_command,
    )
    summary_path = pipeline.results_root / "fixable_summary.json"
    write_fixable_summary(failures, summary_path)
    _log_failure_summary(failures, sources)
    LOGGER.info("Wrote fixable summary to %s", summary_path)

    if not failures:
        LOGGER.info("No fixable rubric entries detected. Nothing to inspect.")
        return

    LOGGER.info("Starting debug run for %d task(s)", len(failures))
    await pipeline.run_batch_debug(failures)
    LOGGER.info("Debug run complete")


def write_fixable_summary(failures: List[Dict[str, Any]], output_path: Path) -> None:
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
