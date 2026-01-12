from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils.docker_runner import DockerRunner
from .auto_fixer import AutoFixer, FixSuggestion
from .task_utils import get_task_data

LOGGER = logging.getLogger(__name__)


class DebugPipeline:
    """Coordinate the log ingestion, fix proposal, and re-run workflow."""

    def __init__(
        self,
        agent_dir: str | os.PathLike[str],
        *,
        agent_args: Optional[Dict[str, Any]] = None,
        agent_function: str | None = None,
        benchmark_name: str | None = None,
    ) -> None:
        self.agent_dir = Path(agent_dir)
        if not self.agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {self.agent_dir}")

        self.benchmark_name = benchmark_name or os.getenv(
            "HAL_AUTOFIX_BENCHMARK",
            "swebench_verified",
        )
        self.results_root = Path("results") / "debug_fixes"
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.autofixer = AutoFixer(agent_dir=self.agent_dir)
        self.agent_function = (
            agent_function
            or os.getenv("HAL_AUTOFIX_AGENT_FUNCTION")
            or "main.run"
        )

        self.agent_args = self._build_agent_args(agent_args)
        self.runner = DockerRunner(
            log_dir=str(self.results_root),
            max_concurrent=1,
            benchmark=None,
        )
        self.runner.verbose = True

    async def run_batch_debug(self, failures_list: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Iterate over rubric failures sequentially."""
        results: List[Dict[str, Any]] = []
        for failure in failures_list:
            try:
                result = await self.debug_single_task(failure)
                results.append(result)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Pipeline failed for task %s: %s", failure.get("task_id"), exc)
        return results

    async def debug_single_task(self, failure_item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full trace-to-fix loop for a single failed task."""
        task_id = failure_item.get("task_id")
        if not task_id:
            raise ValueError("failure_item missing task_id")

        LOGGER.info("Processing task %s", task_id)
        suggestion = self.autofixer.generate_fix(failure_item)
        LOGGER.info(
            "AutoFixer chose %s fix for %s", suggestion.fix_type.upper(), task_id
        )

        task_payload = get_task_data(task_id, self.benchmark_name)
        if suggestion.fix_type == "input":
            task_payload["problem_statement"] = suggestion.content
            agent_dir = self.agent_dir
            temp_dir: Optional[Path] = None
        else:
            temp_dir = self._create_temp_agent_dir(task_id, suggestion.content)
            agent_dir = temp_dir

        dataset = {task_id: task_payload}
        run_id = f"debug_{self._sanitize_task_id(task_id)}_{int(time.time())}"
        task_run_dir = self._get_task_run_dir(task_id, run_id)
        self._log_task_event(
            task_run_dir,
            f"Starting rerun with run_id={run_id}, fix_type={suggestion.fix_type}",
        )

        LOGGER.info(
            "Launching DockerRunner for %s (run_id=%s). Logs: %s",
            task_id,
            run_id,
            task_run_dir,
        )
        run_result = await self.runner.run_agent(
            dataset=dataset,
            agent_function=self.agent_function,
            agent_dir=str(agent_dir),
            agent_args=self.agent_args,
            run_id=run_id,
        )
        LOGGER.info("Completed rerun for %s (run_id=%s)", task_id, run_id)
        self._log_task_event(task_run_dir, "DockerRunner completed for this task.")

        summary = {
            "task_id": task_id,
            "run_id": run_id,
            "fix": suggestion.__dict__,
            "results": run_result,
        }
        self._write_summary(task_run_dir, summary)

        if temp_dir:
            self._cleanup_temp_agent(temp_dir)

        return summary

    def _build_agent_args(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge CLI/env overrides with mandatory defaults."""
        agent_args: Dict[str, Any] = {}
        env_args = os.getenv("HAL_AUTOFIX_AGENT_ARGS")
        if env_args:
            try:
                agent_args.update(json.loads(env_args))
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse HAL_AUTOFIX_AGENT_ARGS; ignoring value")

        if override:
            agent_args.update(override)

        agent_args.setdefault("benchmark_name", self.benchmark_name)
        return agent_args

    def _create_temp_agent_dir(self, task_id: str, new_main_contents: str) -> Path:
        """Create agents/temp_<task_id> and overwrite main.py with new code."""
        sanitized = self._sanitize_task_id(task_id)
        temp_dir = self.agent_dir.parent / f"temp_{sanitized}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(self.agent_dir, temp_dir, dirs_exist_ok=True)

        entrypoint = temp_dir / self.autofixer.entrypoint.name
        entrypoint.write_text(new_main_contents, encoding="utf-8")
        return temp_dir

    def _cleanup_temp_agent(self, temp_dir: Path) -> None:
        try:
            shutil.rmtree(temp_dir)
        except FileNotFoundError:
            return

    def _write_summary(self, output_dir: Path, payload: Dict[str, Any]) -> None:
        summary_path = output_dir / "debug_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log_task_event(output_dir, f"Wrote summary to {summary_path.name}")

    @staticmethod
    def _sanitize_task_id(task_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in task_id)

    def _get_task_run_dir(self, task_id: str, run_id: Optional[str] = None) -> Path:
        base = self.results_root / self._sanitize_task_id(task_id)
        if run_id:
            base = base / run_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _log_task_event(self, output_dir: Path, message: str) -> None:
        log_path = output_dir / "pipeline.log"
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
