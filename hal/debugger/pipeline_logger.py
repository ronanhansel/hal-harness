from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class PipelineRunLogger:
    """Simple file logger that records pipeline events per invocation."""

    def __init__(self, root: Optional[str | Path] = None) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if root is None:
            repo_root = Path(__file__).resolve().parents[3]
            base_dir = repo_root / "log"
        else:
            base_dir = Path(root).expanduser().resolve()
        self.root = base_dir / f"pipeline-run-{timestamp}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "pipeline.log"
        self.log("Pipeline run started.")

    def log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def attach_stdout(self) -> Path:
        """Return a path where callers can optionally copy stdout/stderr captures."""
        return self.root / "stdout.log"

    def finish(self) -> None:
        self.log("Pipeline run completed.")

    def __str__(self) -> str:  # pragma: no cover - convenience
        return str(self.log_path)
