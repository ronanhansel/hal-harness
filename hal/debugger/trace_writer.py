from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class RunnerTraceWriter:
    """Write synthetic trace files for debugger reruns."""

    def __init__(self, root: str | Path = "traces/debug_runs") -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        run_id: str,
        task_id: str,
        success: bool,
        pipeline_log_path: Path,
        rerun_output: Dict[str, Any] | None,
        agent_dir: Path,
        agent_args: Dict[str, Any],
        benchmark_name: str,
        trace_label: str | None = None,
    ) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trace_id = f"{trace_label}_debug_{timestamp}" if trace_label else run_id
        trace_path = self.root / f"{trace_id}.json"
        data = self._load_existing(trace_path, run_id, trace_id, agent_dir, agent_args, benchmark_name)

        results = data.setdefault("results", {"successful_tasks": [], "failed_tasks": []})
        if success:
            if task_id not in results["successful_tasks"]:
                results["successful_tasks"].append(task_id)
            if task_id in results["failed_tasks"]:
                results["failed_tasks"].remove(task_id)
        else:
            if task_id not in results["failed_tasks"]:
                results["failed_tasks"].append(task_id)
            if task_id in results["successful_tasks"]:
                results["successful_tasks"].remove(task_id)

        pipeline_log = pipeline_log_path.read_text(encoding="utf-8") if pipeline_log_path.exists() else ""
        rerun_text = json.dumps(rerun_output, indent=2) if isinstance(rerun_output, dict) else str(rerun_output or "")

        entry = {
            "id": f"{task_id}-{datetime.utcnow().timestamp()}",
            "attributes": {"weave_task_id": task_id},
            "inputs": {
                "messages": [
                    {
                        "role": "system",
                        "content": pipeline_log or "Debugger pipeline log unavailable.",
                    }
                ]
            },
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": rerun_text or "Debugger rerun produced no textual output.",
                        }
                    }
                ]
            },
            "created_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        data.setdefault("raw_logging_results", []).append(entry)

        trace_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[pipeline] wrote synthetic trace {trace_path}")
        self._write_verbose_log(trace_id, task_id, pipeline_log, rerun_text)
        return trace_path

    def _load_existing(
        self,
        trace_path: Path,
        run_id: str,
        trace_id: str,
        agent_dir: Path,
        agent_args: Dict[str, Any],
        benchmark_name: str,
    ) -> Dict[str, Any]:
        if trace_path.exists():
            try:
                return json.loads(trace_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        return {
            "config": {
                "agent_name": agent_dir.name,
                "benchmark_name": benchmark_name,
                "agent_args": agent_args,
                "run_id": run_id,
                "trace_id": trace_id,
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
            "results": {"successful_tasks": [], "failed_tasks": []},
            "raw_logging_results": [],
        }

    def _write_verbose_log(self, trace_id: str, task_id: str, pipeline_log: str, rerun_text: str) -> None:
        task_dir = self.root / trace_id / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        verbose_path = task_dir / "verbose.log"
        combined = []
        if pipeline_log:
            combined.append(pipeline_log.strip())
        if rerun_text:
            combined.append("RERUN OUTPUT:\n" + rerun_text.strip())
        verbose_path.write_text("\n\n".join(combined) + "\n", encoding="utf-8")
        print(f"[pipeline] wrote {verbose_path}")
