from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from hal.debugger.log_reader import LogIngester
from hal.debugger.pipeline import DebugPipeline

LOGGER = logging.getLogger("hal.auto_debug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the automated trace-to-fix debugger over a batch of failures.",
    )
    parser.add_argument("--rubrics-csv", required=True, help="Path to environmental_barrier_rubrics.csv")
    parser.add_argument("--traces-dir", required=True, help="Directory that stores trace folders for each model run")
    parser.add_argument("--agent-dir", required=True, help="Directory that contains the agent entrypoint (main.py)")
    parser.add_argument("--agent-args", help="Optional JSON string or path to a JSON file with agent kwargs")
    parser.add_argument("--agent-function", help="Override the agent function path (default: main.run)")
    parser.add_argument("--benchmark-name", help="Override target benchmark (default: swebench_verified)")
    return parser.parse_args()


def load_agent_args(payload: Optional[str]) -> Optional[Dict[str, Any]]:
    if not payload:
        return None
    candidate = Path(payload)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(payload)


async def _run_pipeline(args: argparse.Namespace, rerun_command: str) -> None:
    LOGGER.info("Loading failures from %s", args.rubrics_csv)
    ingester = LogIngester(args.rubrics_csv, args.traces_dir)
    failures = ingester.get_failing_tasks()
    if not failures:
        LOGGER.info("No environmental barriers detected. Nothing to do.")
        return

    agent_args = load_agent_args(args.agent_args)
    pipeline = DebugPipeline(
        agent_dir=args.agent_dir,
        agent_args=agent_args,
        agent_function=args.agent_function,
        benchmark_name=args.benchmark_name,
        rerun_command=rerun_command,
    )
    LOGGER.info("Starting debug run for %d task(s)", len(failures))
    await pipeline.run_batch_debug(failures)
    LOGGER.info("Debug run complete")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    rerun_command = "python scripts/auto_debug_batch.py " + " ".join(sys.argv[1:])
    asyncio.run(_run_pipeline(args, rerun_command))


if __name__ == "__main__":
    main()
