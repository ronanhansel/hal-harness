from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)
TAIL_BYTES = 4096


class LogIngester:
    """Load HAL rubric rows and associated trace snippets."""

    def __init__(self, csv_path: str | os.PathLike[str], traces_root_dir: str | os.PathLike[str]):
        self.csv_path = Path(csv_path)
        self.traces_root_dir = Path(traces_root_dir)

    def get_failing_tasks(self) -> List[Dict[str, Any]]:
        """Return every rubric entry that maps to an environmental barrier."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Rubrics CSV not found: {self.csv_path}")

        failures: List[Dict[str, Any]] = []
        with self.csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not self._is_environmental_barrier(row):
                    continue

                task_id = (row.get("task_id") or "").strip()
                if not task_id:
                    LOGGER.debug("Skipping row without task_id: %s", row)
                    continue

                model_run = (row.get("model_run") or "").strip()
                trace_content = self._load_trace_tail(model_run, task_id)

                failures.append(
                    {
                        "task_id": task_id,
                        "explanation": (row.get("explanation") or "").strip(),
                        "trace_content": trace_content,
                        "model_run": model_run,
                    }
                )
        return failures

    @staticmethod
    def _is_environmental_barrier(row: Dict[str, Any]) -> bool:
        criteria = (row.get("criteria") or "").strip().lower()
        if criteria != "environmentalbarrier":
            return False

        grade_str = (row.get("grade") or "0").strip()
        try:
            grade = float(grade_str)
        except ValueError:
            LOGGER.debug("Unable to parse grade '%s'; treating as 0", grade_str)
            return False
        return grade > 0

    def _load_trace_tail(self, model_run: str, task_id: str) -> str:
        trace_path = self.traces_root_dir / model_run / task_id / "verbose.log"
        if not trace_path.exists():
            LOGGER.warning("Trace file missing for run=%s task=%s (%s)", model_run, task_id, trace_path)
            return ""

        with trace_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            offset = max(file_size - TAIL_BYTES, 0)
            handle.seek(offset, os.SEEK_SET)
            snippet = handle.read().decode("utf-8", errors="replace")

        if offset > 0:
            newline_idx = snippet.find("\n")
            if newline_idx != -1:
                snippet = snippet[newline_idx + 1 :]
        return snippet.strip()
