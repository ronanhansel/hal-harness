from __future__ import annotations

import copy
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils.docker_runner import DockerRunner
from .fix_loader import FixPackage, apply_agent_overlay, apply_agent_patch, load_fix_package
from .pipeline_base import PipelineBase
from .trace_writer import RunnerTraceWriter

LOGGER = logging.getLogger(__name__)


class RunnerPipeline(PipelineBase):
    """Apply fix packages, run the agent in Docker, and archive synthetic traces."""

    def __init__(
        self,
        agent_dir: str | Path,
        *,
        agent_args: Dict[str, Any] | None = None,
        agent_function: str | None = None,
        benchmark_name: str | None = None,
        rerun_command: str,
        output_root: str | Path = "debug",
        fixes_root: str | Path = "fixes",
        fixed_layout: bool = False,
        run_label: str = "latest",
        trace_output_root: str | Path = "traces/debug_runs",
    ) -> None:
        super().__init__(
            agent_dir,
            agent_args=agent_args,
            agent_function=agent_function,
            benchmark_name=benchmark_name,
            rerun_command=rerun_command,
            output_root=output_root,
            fixes_root=fixes_root,
            fixed_layout=fixed_layout,
            run_label=run_label,
        )
        self.runner = DockerRunner(
            log_dir=str(self.results_root),
            max_concurrent=1,
            benchmark=None,
        )
        self.runner.verbose = True
        self.trace_writer = RunnerTraceWriter(trace_output_root)

    async def run_batch(self, failures_list: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for failure in failures_list:
            try:
                results.append(await self.run_single_task(failure))
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Runner failed for task %s: %s", failure.get("task_id"), exc)
        return results

    async def run_single_task(self, failure_item: Dict[str, Any]) -> Dict[str, Any]:
        task_id = failure_item.get("task_id")
        if not task_id:
            raise ValueError("failure_item missing task_id")

        task_payload = self._get_task_payload(task_id)
        sanitized_id = self._sanitize_task_id(task_id)
        if self.fixed_layout:
            run_id = self.run_label
            task_run_dir = self._get_task_run_dir(task_id)
        else:
            run_id = f"debug_{sanitized_id}_{int(time.time())}"
            task_run_dir = self._get_task_run_dir(task_id, run_id)
        trace_label = self._derive_trace_label(failure_item, task_id)

        fix_package = load_fix_package(sanitized_id, self.fixes_root, self.benchmark_name)
        if not fix_package:
            self._log_task_event(
                task_run_dir,
                f"No fix package detected for {task_id} in {self._fix_dir_for_task(task_id)}.",
            )
            return {"task_id": task_id, "run_id": run_id, "status": "missing_fix"}

        self._log_task_event(task_run_dir, "Launching debugger rerun with fixes applied.")
        rerun_output = await self._execute_fix_rerun(
            task_id=task_id,
            run_id=run_id,
            task_run_dir=task_run_dir,
            base_task_payload=task_payload,
            fix_package=fix_package,
        )

        if rerun_output is None:
            self._log_task_event(task_run_dir, "Debugger rerun produced no output.")
            return {"task_id": task_id, "run_id": run_id, "status": "no_output"}

        self._write_json(task_run_dir / "rerun_results.json", rerun_output, task_run_dir)
        success = self._rerun_successful(task_id, rerun_output)
        status = "passed" if success else "failed"
        self._log_task_event(task_run_dir, f"Debugger rerun completed with status={status}.")

        trace_path = self.trace_writer.record(
            run_id=run_id,
            task_id=task_id,
            success=success,
            pipeline_log_path=task_run_dir / "pipeline.log",
            rerun_output=rerun_output,
            agent_dir=self.agent_dir,
            agent_args=self.agent_args,
            benchmark_name=self.benchmark_name,
            trace_label=trace_label,
        )
        pointer_path = task_run_dir / "trace_pointer.txt"
        pointer_path.write_text(str(trace_path), encoding="utf-8")
        self._log_task_event(task_run_dir, f"Synthetic trace recorded at {trace_path}")

        return {
            "task_id": task_id,
            "run_id": run_id,
            "rerun_results": rerun_output,
            "status": status,
        }

    async def _execute_fix_rerun(
        self,
        task_id: str,
        run_id: str,
        task_run_dir: Path,
        base_task_payload: Dict[str, Any],
        fix_package: FixPackage,
    ) -> Optional[Dict[str, Any]]:
        task_payload = copy.deepcopy(base_task_payload)
        if fix_package.input_override:
            task_payload.update(fix_package.input_override)

        dataset = {task_id: task_payload}
        temp_agent_dir: Optional[Path] = None

        if fix_package.has_agent_changes:
            temp_agent_dir = self._prepare_agent_dir(task_id, fix_package)
            agent_dir = temp_agent_dir
        else:
            agent_dir = self.agent_dir

        env_overrides = {task_id: fix_package.env_override} if fix_package.env_override else None

        try:
            run_result = await self.runner.run_agent(
                dataset=dataset,
                agent_function=self.agent_function,
                agent_dir=str(agent_dir),
                agent_args=self.agent_args,
                run_id=run_id,
                task_env_overrides=env_overrides,
            )
            self._log_task_event(task_run_dir, "Docker rerun finished.")
        finally:
            if temp_agent_dir and temp_agent_dir.exists():
                shutil.rmtree(temp_agent_dir, ignore_errors=True)

        return run_result

    def _prepare_agent_dir(self, task_id: str, fix_package: FixPackage) -> Path:
        sanitized = self._sanitize_task_id(task_id)
        benchmark_slug = self._sanitize_task_id(self.benchmark_name)
        temp_dir = self.agent_dir.parent / f"temp_{benchmark_slug}_{sanitized}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(self.agent_dir, temp_dir)

        if fix_package.agent_overlay:
            apply_agent_overlay(temp_dir, fix_package.agent_overlay)
        if fix_package.agent_patch:
            apply_agent_patch(temp_dir, fix_package.agent_patch)

        return temp_dir

    def _derive_trace_label(self, failure_item: Dict[str, Any], task_id: str) -> str:
        source = failure_item.get("_source_csv")
        if source:
            return Path(source).stem
        benchmark_slug = self._sanitize_task_id(self.benchmark_name)
        task_slug = self._sanitize_task_id(task_id or "task")
        model_name = str(self.agent_args.get("model_name") or "model")
        model_slug = self._sanitize_task_id(model_name)
        return f"debug_{benchmark_slug}_{task_slug}_{model_slug}"

    @staticmethod
    def _rerun_successful(task_id: str, rerun_output: Dict[str, Any]) -> bool:
        if not rerun_output:
            return False

        candidate = rerun_output.get(task_id) if isinstance(rerun_output, dict) else rerun_output
        if isinstance(candidate, str):
            lowered = candidate.lower()
            if lowered.startswith("error") or lowered.startswith("timeout"):
                return False
            return True

        if isinstance(candidate, dict):
            status = str(
                candidate.get("status")
                or candidate.get("result")
                or candidate.get("state")
                or ""
            ).lower()
            if status in {"success", "succeeded", "passed", "pass", "ok", "done"}:
                return True
            if status in {"error", "failed", "failure", "timeout"}:
                return False

            if candidate.get("success") is True:
                return True
            if candidate.get("success") is False:
                return False

            grade = candidate.get("grade")
            try:
                if grade is not None and float(grade) > 0:
                    return True
            except (TypeError, ValueError):
                pass

        return False
