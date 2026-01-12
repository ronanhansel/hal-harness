from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import dotenv_values

from ..utils.docker_runner import DockerRunner
from .auto_inspector import AutoInspector
from .fix_loader import (
    FixPackage,
    apply_agent_overlay,
    apply_agent_patch,
    load_fix_package,
)
from .task_utils import get_task_data

LOGGER = logging.getLogger(__name__)


class DebugPipeline:
    """Coordinate inspection, fix packaging, and reruns."""

    def __init__(
        self,
        agent_dir: str | os.PathLike[str],
        *,
        agent_args: Optional[Dict[str, Any]] = None,
        agent_function: str | None = None,
        benchmark_name: str | None = None,
        rerun_command: str,
        output_root: str | os.PathLike[str] = "debug",
        fixes_root: str | os.PathLike[str] = "fixes",
    ) -> None:
        self.agent_dir = Path(agent_dir)
        if not self.agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {self.agent_dir}")

        self.benchmark_name = benchmark_name or os.getenv(
            "HAL_AUTOFIX_BENCHMARK", "swebench_verified"
        )
        self.results_root = Path(output_root)
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.auto_inspector = AutoInspector(agent_dir=self.agent_dir)
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

        self.fixes_root = Path(fixes_root)
        self.fixes_root.mkdir(parents=True, exist_ok=True)
        self.rerun_command = rerun_command

        self._agent_tree_snapshot = self._collect_agent_tree()
        self._env_summary = self._collect_env_summary()
        self._tools_description = self._describe_agent_tools()

    async def run_batch_debug(self, failures_list: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for failure in failures_list:
            try:
                results.append(await self.debug_single_task(failure))
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Pipeline failed for task %s: %s", failure.get("task_id"), exc)
        return results

    async def debug_single_task(self, failure_item: Dict[str, Any]) -> Dict[str, Any]:
        task_id = failure_item.get("task_id")
        if not task_id:
            raise ValueError("failure_item missing task_id")

        LOGGER.info("Inspecting task %s", task_id)
        task_payload = get_task_data(task_id, self.benchmark_name)
        sanitized_id = self._sanitize_task_id(task_id)
        run_id = f"debug_{sanitized_id}_{int(time.time())}"
        task_run_dir = self._get_task_run_dir(task_id, run_id)
        single_task_command = self._build_single_task_command(self.rerun_command, task_id)

        context_blocks = self._build_context_blocks(
            failure_item, task_payload, None, None, task_run_dir
        )
        report = self.auto_inspector.generate_report(
            failure_item,
            log_dir=task_run_dir,
            context_blocks=context_blocks,
        )
        report_dict = report.to_dict()

        workspace_root = Path.cwd().resolve()
        fix_dir = self.fixes_root / sanitized_id
        next_steps = report_dict.get("next_steps", "")
        reminder_parts = [
            next_steps.strip(),
            f"Create or update {fix_dir} with overlays/patches/input/env overrides.",
            f"After packaging fixes, rerun `{self.rerun_command}` to validate queued tasks.",
            f"For a quick single-task check, rerun `{single_task_command}`.",
        ]
        report_dict["next_steps"] = " ".join(part for part in reminder_parts if part).strip()
        report_dict["coding_agent_context"] = self._build_coding_agent_context(
            workspace_root, fix_dir, self.rerun_command, single_task_command
        )

        LOGGER.info("Inspection completed for %s", task_id)
        self._write_json(task_run_dir / "inspection_report.json", report_dict, task_run_dir)

        rerun_output: Optional[Dict[str, Any]] = None
        followup_report: Optional[Dict[str, Any]] = None
        fix_package = load_fix_package(sanitized_id, self.fixes_root)

        if fix_package:
            LOGGER.info("Fix package found for %s; rerunning debugger", task_id)
            rerun_output = await self._execute_fix_rerun(
                task_id=task_id,
                run_id=run_id,
                task_run_dir=task_run_dir,
                base_task_payload=task_payload,
                fix_package=fix_package,
            )
            if rerun_output is not None:
                self._write_json(task_run_dir / "rerun_results.json", rerun_output, task_run_dir)
                followup_contexts = self._build_context_blocks(
                    failure_item, task_payload, fix_package, rerun_output, task_run_dir
                )
                followup = self.auto_inspector.generate_report(
                    failure_item,
                    log_dir=task_run_dir,
                    context_blocks=followup_contexts,
                )
                followup_report = followup.to_dict()
                followup_report["coding_agent_context"] = self._build_coding_agent_context(
                    workspace_root, fix_dir, self.rerun_command, single_task_command
                )
                self._write_json(
                    task_run_dir / "post_rerun_report.json",
                    followup_report,
                    task_run_dir,
                )
        else:
            self._log_task_event(task_run_dir, f"No fix package detected in {fix_dir}.")

        summary = {
            "task_id": task_id,
            "run_id": run_id,
            "report": report_dict,
        }
        if rerun_output is not None:
            summary["rerun_results"] = rerun_output
        if followup_report is not None:
            summary["post_rerun_report"] = followup_report
        return summary

    def _build_agent_args(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

        self._log_task_event(task_run_dir, "Launching debugger rerun with fixes applied.")

        env_overrides = (
            {task_id: fix_package.env_override}
            if fix_package.env_override
            else None
        )

        try:
            run_result = await self.runner.run_agent(
                dataset=dataset,
                agent_function=self.agent_function,
                agent_dir=str(agent_dir),
                agent_args=self.agent_args,
                run_id=run_id,
                task_env_overrides=env_overrides,
            )
            self._log_task_event(task_run_dir, "Debugger rerun completed.")
        finally:
            if temp_agent_dir and temp_agent_dir.exists():
                shutil.rmtree(temp_agent_dir, ignore_errors=True)

        return run_result

    def _prepare_agent_dir(self, task_id: str, fix_package: FixPackage) -> Path:
        sanitized = self._sanitize_task_id(task_id)
        temp_dir = self.agent_dir.parent / f"temp_{sanitized}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        shutil.copytree(self.agent_dir, temp_dir)

        if fix_package.agent_overlay:
            apply_agent_overlay(temp_dir, fix_package.agent_overlay)
        if fix_package.agent_patch:
            apply_agent_patch(temp_dir, fix_package.agent_patch)

        return temp_dir

    def _write_json(self, path: Path, payload: Dict[str, Any], log_dir: Path) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log_task_event(log_dir, f"Wrote {path.name}")

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

    def _build_context_blocks(
        self,
        failure_item: Dict[str, Any],
        task_payload: Dict[str, Any],
        fix_package: Optional[FixPackage],
        rerun_output: Optional[Dict[str, Any]],
        task_run_dir: Path,
    ) -> List[Tuple[str, str]]:
        blocks: List[Tuple[str, str]] = []
        source_csv = failure_item.get("_source_csv")
        if source_csv:
            blocks.append(("rubric_source", source_csv))
        blocks.append(("task_input", self._truncate(json.dumps(task_payload, indent=2))))
        blocks.append(("agent_args", self._truncate(json.dumps(self.agent_args, indent=2))))
        blocks.append(("env_summary", self._env_summary))
        blocks.append(("agent_file_tree", self._agent_tree_snapshot))
        blocks.append(("available_tools", self._tools_description))
        if fix_package:
            blocks.append(("fix_package", self._summarize_fix_package(fix_package)))
        if rerun_output is not None:
            blocks.append(("rerun_results", self._truncate(json.dumps(rerun_output, indent=2))))
        log_path = task_run_dir / "pipeline.log"
        blocks.append(("log_location", str(log_path)))
        return blocks

    def _build_coding_agent_context(
        self,
        workspace_root: Path,
        fix_dir: Path,
        rerun_command: str,
        single_task_command: str,
    ) -> Dict[str, Any]:
        instructions = [
            f"cd {workspace_root}",
            "Inspect inspection_report.json for analysis and rationale.",
            "Do NOT modify repository files directly; place overlays or patches under fixes/<task_id>/.",
            "Use input_override.json / problem_statement.txt / env_override.json if inputs or env vars must change.",
            f"After preparing the fix package, rerun the debugger using `{rerun_command}` for the full backlog.",
            f"To iterate quickly on this task only, run `{single_task_command}` (leverages --task-id filtering).",
        ]
        return {
            "working_directory": str(workspace_root),
            "fix_folder": str(fix_dir),
            "rerun_command": rerun_command,
            "rerun_single_task_command": single_task_command,
            "system_prompt": (
                "You are a coding agent operating inside hal-harness. Follow the inspection report guidance, "
                "prepare self-contained fixes under fixes/<task_id>/, and rerun the debugger command to validate. "
                "Never modify repository files directly."
            ),
            "instructions": instructions,
        }

    def _collect_agent_tree(self, limit: int = 400) -> str:
        entries: List[str] = []
        for path in sorted(self.agent_dir.rglob("*")):
            rel = path.relative_to(self.agent_dir)
            suffix = "/" if path.is_dir() else ""
            entries.append(str(rel) + suffix)
            if len(entries) >= limit:
                entries.append("... (truncated)")
                break
        if not entries:
            return "(agent directory empty)"
        return "\n".join(entries)

    def _collect_env_summary(self) -> str:
        try:
            env = dotenv_values(".env")
        except Exception:
            env = {}
        if not env:
            return "No .env file detected or it is empty."
        lines = []
        for key in sorted(env.keys()):
            value = env[key]
            masked = "***" if value else ""
            lines.append(f"{key}={masked}")
        return "\n".join(lines)

    def _describe_agent_tools(self) -> str:
        return (
            "HAL generalist agent uses CORE_TOOLS: GoogleSearchTool(provider='serpapi'), "
            "VisitWebpageTool, PythonInterpreterTool, execute_bash (sandboxed), "
            "TextInspectorTool, edit_file, file_content_search, and query_vision_language_model. "
            "Ensure necessary API keys (e.g., SERPAPI_API_KEY) are supplied via env overrides."
        )

    def _summarize_fix_package(self, fix_package: FixPackage) -> str:
        parts = []
        if fix_package.agent_overlay:
            parts.append(f"overlay: {fix_package.agent_overlay}")
        if fix_package.agent_patch:
            parts.append(f"patch: {fix_package.agent_patch.name}")
        if fix_package.input_override:
            parts.append(f"input_override keys: {list(fix_package.input_override.keys())}")
        if fix_package.env_override:
            parts.append(f"env_override keys: {list(fix_package.env_override.keys())}")
        if not parts:
            return "Fix folder present but empty."
        return "; ".join(parts)

    @staticmethod
    def _truncate(text: str, limit: int = 4000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n... (truncated)"

    @staticmethod
    def _build_single_task_command(base_command: str, task_id: str) -> str:
        flag = f"--task-id {task_id}"
        if flag in base_command:
            return base_command
        return f"{base_command} {flag}"
