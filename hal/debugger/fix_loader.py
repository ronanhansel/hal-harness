from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class FixPackage:
    root: Path
    agent_overlay: Optional[Path]
    agent_patch: Optional[Path]
    input_override: Optional[Dict[str, object]]
    env_override: Optional[Dict[str, str]]

    @property
    def has_agent_changes(self) -> bool:
        return bool(self.agent_overlay or self.agent_patch)

    @property
    def has_input_override(self) -> bool:
        return bool(self.input_override)

    @property
    def has_env_override(self) -> bool:
        return bool(self.env_override)

    @property
    def is_empty(self) -> bool:
        return not (self.has_agent_changes or self.has_input_override or self.has_env_override)


def load_fix_package(
    task_id: str,
    fixes_root: Path | str = "fixes",
    benchmark: str | None = None,
) -> Optional[FixPackage]:
    fixes_root = Path(fixes_root).expanduser().resolve()
    candidates = []
    if benchmark:
        benchmark_slug = _sanitize_component(benchmark)
        candidates.append(fixes_root / benchmark_slug / task_id)
    candidates.append(fixes_root / task_id)

    for task_root in candidates:
        task_root = task_root.resolve()
        if task_root.exists():
            package = _build_package(task_root)
            if package is not None:
                return package
    return None

def _build_package(task_root: Path) -> Optional[FixPackage]:
    agent_overlay = task_root / "agent"
    if not agent_overlay.exists():
        agent_overlay = None

    agent_patch = task_root / "patch.diff"
    if not agent_patch.exists():
        agent_patch = None

    input_override = _load_json(task_root / "input_override.json")
    problem_statement_txt = task_root / "problem_statement.txt"
    if problem_statement_txt.exists():
        text = problem_statement_txt.read_text(encoding="utf-8")
        if input_override is None:
            input_override = {}
        input_override["problem_statement"] = text

    env_override = _load_json(task_root / "env_override.json")

    package = FixPackage(
        root=task_root,
        agent_overlay=agent_overlay,
        agent_patch=agent_patch,
        input_override=input_override,
        env_override=env_override,
    )

    if package.is_empty:
        return None
    return package


def apply_agent_overlay(base_agent_dir: Path, overlay_dir: Path) -> None:
    for item in overlay_dir.iterdir():
        dest = base_agent_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def apply_agent_patch(base_agent_dir: Path, patch_file: Path) -> None:
    proc = subprocess.run(
        ["patch", "-p0", "--batch", "-i", str(patch_file)],
        cwd=base_agent_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to apply patch {patch_file}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
