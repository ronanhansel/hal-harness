from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional dependency
    def dotenv_values(_: str) -> Dict[str, str]:  # type: ignore[override]
        return {}

from .fix_loader import FixPackage
from .task_utils import get_task_data

LOGGER = logging.getLogger(__name__)


class PipelineBase:
    """Shared helpers for inspector and runner pipelines."""

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
        fixed_layout: bool = False,
        run_label: str = "latest",
    ) -> None:
        self.agent_dir = Path(agent_dir)
        if not self.agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {self.agent_dir}")

        self.benchmark_name = benchmark_name or os.getenv("HAL_AUTOFIX_BENCHMARK", "swebench_verified")
        self.results_root = Path(output_root).expanduser().resolve()
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.agent_function = agent_function or os.getenv("HAL_AUTOFIX_AGENT_FUNCTION") or "main.run"
        self.agent_args = self._build_agent_args(agent_args)

        self.fixes_root = Path(fixes_root).expanduser().resolve()
        self.fixes_root.mkdir(parents=True, exist_ok=True)
        self.fix_benchmark_root = self.fixes_root / self._sanitize_task_id(self.benchmark_name)
        self.fix_benchmark_root.mkdir(parents=True, exist_ok=True)

        self.rerun_command = rerun_command
        self.fixed_layout = fixed_layout
        self.run_label = run_label if run_label else "latest"

        self._agent_tree_snapshot = self._collect_agent_tree()
        self._env_summary = self._collect_env_summary()
        self._tools_description = self._describe_agent_tools()

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

    def _sanitize_task_id(self, task_id: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in task_id)

    def _get_task_run_dir(self, task_id: str, run_id: Optional[str] = None) -> Path:
        if self.fixed_layout:
            base = self.results_root / self.benchmark_name / self._sanitize_task_id(task_id) / self.run_label
        else:
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

    def _write_json(self, path: Path, payload: Dict[str, Any], log_dir: Path) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log_task_event(log_dir, f"Wrote {path.name}")
        print(f"[pipeline] wrote {path}")

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
        task_id: str,
        rerun_command: str,
        single_task_command: str,
    ) -> Dict[str, Any]:
        fix_dir = self._fix_dir_for_task(task_id, ensure_exists=True)
        instructions = [
            f"cd {workspace_root}",
            "Inspect inspection_report.json for analysis and rationale.",
            f"Do NOT modify repository files directly; place overlays or patches under {fix_dir}.",
            f"Prefer dropping fully-edited files under {fix_dir / 'agent'} (patch.diff is only a fallback when overlays are impractical).",
            "Use input_override.json / problem_statement.txt / env_override.json if inputs or env vars must change (store them beside the fix).",
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

    def _get_task_payload(self, task_id: str) -> Dict[str, Any]:
        return get_task_data(task_id, self.benchmark_name)

    def _fix_dir_for_task(self, task_id: str, ensure_exists: bool = False) -> Path:
        path = self.fix_benchmark_root / self._sanitize_task_id(task_id)
        if ensure_exists:
            path.mkdir(parents=True, exist_ok=True)
        return path
