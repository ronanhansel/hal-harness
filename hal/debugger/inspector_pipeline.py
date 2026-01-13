from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .auto_inspector import AutoInspector
from .fix_loader import load_fix_package
from .pipeline_base import PipelineBase

LOGGER = logging.getLogger(__name__)


class InspectorPipeline(PipelineBase):
    """LLM-based inspector that produces guidance without applying fixes."""

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
        reasoning_effort: str | None = None,
        inspector_model: str | None = None,
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
        self.auto_inspector = AutoInspector(
            agent_dir=self.agent_dir,
            llm_model=inspector_model,
            reasoning_effort=reasoning_effort,
        )

    async def run_batch(self, failures_list: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for failure in failures_list:
            try:
                results.append(await self.inspect_single_task(failure))
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Inspector failed for task %s: %s", failure.get("task_id"), exc)
        return results

    async def inspect_single_task(self, failure_item: Dict[str, Any]) -> Dict[str, Any]:
        task_id = failure_item.get("task_id")
        if not task_id:
            raise ValueError("failure_item missing task_id")

        LOGGER.info("Inspecting task %s", task_id)
        task_payload = self._get_task_payload(task_id)
        sanitized_id = self._sanitize_task_id(task_id)
        if self.fixed_layout:
            run_id = self.run_label
            task_run_dir = self._get_task_run_dir(task_id)
        else:
            run_id = f"inspect_{sanitized_id}_{int(time.time())}"
            task_run_dir = self._get_task_run_dir(task_id, run_id)
        single_task_command = self._build_single_task_command(self.rerun_command, task_id)

        fix_package = load_fix_package(sanitized_id, self.fixes_root, self.benchmark_name)
        context_blocks = self._build_context_blocks(
            failure_item,
            task_payload,
            fix_package,
            None,
            task_run_dir,
        )

        report = self.auto_inspector.generate_report(
            failure_item,
            log_dir=task_run_dir,
            context_blocks=context_blocks,
        )
        report_dict = report.to_dict()

        workspace_root = Path.cwd().resolve()
        next_steps = report_dict.get("next_steps", "")
        reminder_parts = [
            next_steps.strip(),
            f"Create or update {self._fix_dir_for_task(task_id, ensure_exists=True)} with overlays/patches/input/env overrides.",
            f"After packaging fixes, rerun `{self.rerun_command}` to validate queued tasks.",
            f"For a quick single-task check, rerun `{single_task_command}`.",
        ]
        report_dict["next_steps"] = " ".join(part for part in reminder_parts if part).strip()
        report_dict["coding_agent_context"] = self._build_coding_agent_context(
            workspace_root,
            task_id,
            self.rerun_command,
            single_task_command,
        )

        self._write_json(task_run_dir / "inspection_report.json", report_dict, task_run_dir)
        LOGGER.info("Inspection completed for %s", task_id)

        return {
            "task_id": task_id,
            "run_id": run_id,
            "report": report_dict,
        }
