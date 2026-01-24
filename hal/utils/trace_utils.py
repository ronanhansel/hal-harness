import json
import os
from pathlib import Path
from typing import Any, Dict, List


def get_trace_mode() -> str:
    """Return trace mode: local, weave, or both."""
    mode = (os.getenv("HAL_TRACE_MODE") or "").strip().lower()
    if mode in ("local", "weave", "both"):
        return mode
    if os.getenv("WANDB_API_KEY"):
        return "weave"
    return "local"


def trace_output_dir() -> Path:
    """Resolve the local trace output directory."""
    configured = os.getenv("HAL_TRACE_OUTPUT_DIR", "traces")
    path = Path(configured)
    if path.is_absolute():
        return path
    repo_root = os.getenv("HAL_REPO_ROOT")
    if repo_root:
        return Path(repo_root) / path
    # hal-harness/hal/utils -> repo root is 3 levels up
    return Path(__file__).resolve().parents[3] / path


def collect_local_trace_entries(run_dir: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Load local trace JSONL files from a run directory."""
    entries: List[Dict[str, Any]] = []
    run_path = Path(run_dir)
    if not run_path.exists():
        return entries
    for trace_path in run_path.rglob("local_trace.jsonl"):
        try:
            with trace_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    return entries
