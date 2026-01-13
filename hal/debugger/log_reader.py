from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)
TAIL_BYTES = 4096


class LogIngester:
    """Load HAL rubric rows and associated trace snippets."""

    def __init__(
        self,
        csv_path: str | os.PathLike[str],
        traces_root_dir: str | os.PathLike[str],
        allowed_criteria: Optional[List[str]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.traces_root_dir = Path(traces_root_dir)
        self.allowed_criteria = [c.lower() for c in allowed_criteria] if allowed_criteria else None

    def get_failing_tasks(self) -> List[Dict[str, Any]]:
        """Return every rubric entry that maps to an environmental barrier."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Rubrics CSV not found: {self.csv_path}")

        failures: List[Dict[str, Any]] = []
        with self.csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not self._include_row(row):
                    continue

                task_id = (row.get("task_id") or "").strip()
                if not task_id:
                    LOGGER.debug("Skipping row without task_id: %s", row)
                    continue

                model_run = (row.get("model_run") or "").strip()
                criteria = (row.get("criteria") or "").strip()
                grade_str = (row.get("grade") or "0").strip()
                grade_value = self._parse_grade(grade_str)
                trace_content = self._load_trace_tail(model_run, task_id)

                failures.append(
                    {
                        "task_id": task_id,
                        "explanation": (row.get("explanation") or "").strip(),
                        "trace_content": trace_content,
                        "model_run": model_run,
                        "criteria": criteria,
                        "grade": grade_str,
                        "grade_numeric": grade_value,
                        "raw_row": row,
                    }
                )
        return failures

    def _include_row(self, row: Dict[str, Any]) -> bool:
        criteria = (row.get("criteria") or "").strip().lower()
        if self.allowed_criteria is not None and criteria not in self.allowed_criteria:
            return False

        grade = self._parse_grade(row.get("grade"))
        return grade > 0

    def _load_trace_tail(self, model_run: str, task_id: str) -> str:
        candidates = [
            self.traces_root_dir / model_run / task_id / "verbose.log",
            self.traces_root_dir / "debug_runs" / model_run / task_id / "verbose.log",
        ]
        trace_path = None
        for candidate in candidates:
            if candidate.exists():
                trace_path = candidate
                break
        if trace_path is None:
            fallback = self._load_from_trace_json(model_run, task_id)
            if fallback:
                return fallback
            LOGGER.warning(
                "Trace file missing for run=%s task=%s (searched %s)",
                model_run,
                task_id,
                candidates[0].parent,
            )
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

    def _load_from_trace_json(self, model_run: str, task_id: str) -> str:
        trace_json = self.traces_root_dir / f"{model_run}.json"
        if not trace_json.exists():
            trace_json = self.traces_root_dir / "debug_runs" / f"{model_run}.json"
        if not trace_json.exists():
            return ""
        try:
            data = json.loads(trace_json.read_text(encoding="utf-8"))
        except Exception:
            return ""

        raw_entries = data.get("raw_logging_results") or []
        filtered = [
            entry for entry in raw_entries
            if self._entry_matches_task(entry, task_id)
        ]
        if not filtered:
            return ""
        latest = filtered[-1]
        message = latest.get("output", {}).get("choices") or []
        if message:
            choice = message[-1]
            content = choice.get("message") or {}
            text = content.get("content")
            if isinstance(text, str):
                return text.strip()
        output_text = latest.get("output_text")
        if isinstance(output_text, str):
            return output_text.strip()
        return ""

    @staticmethod
    def _entry_matches_task(entry: Dict[str, Any], task_id: str) -> bool:
        attributes = entry.get("attributes") or {}
        return (
            attributes.get("weave_task_id") == task_id
            or entry.get("weave_task_id") == task_id
            or entry.get("inputs", {}).get("task_id") == task_id
        )

    def _parse_grade(self, grade_raw: Optional[str]) -> float:
        grade_str = (grade_raw or "0").strip()
        try:
            return float(grade_str)
        except ValueError:
            LOGGER.debug("Unable to parse grade '%s'; treating as 0", grade_str)
            return 0.0
