from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import litellm  # type: ignore
except ImportError:  # pragma: no cover
    litellm = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class InspectionReport:
    analysis: str
    rationale: str
    recommended_files: List[str]
    recommended_actions: List[str]
    next_steps: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoInspector:
    """LLM-based analyst that produces guidance instead of patches."""

    def __init__(
        self,
        agent_dir: str | os.PathLike[str],
        llm_model: str | None = None,
        entrypoint: str | None = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.agent_dir = Path(agent_dir)
        if not self.agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {self.agent_dir}")

        self.model = llm_model or os.getenv("HAL_AUTOFIX_MODEL", "gpt-4o-mini-2024-07-18")
        entrypoint_name = entrypoint or os.getenv("HAL_AUTOFIX_ENTRYPOINT", "main.py")
        self.entrypoint = self.agent_dir / entrypoint_name
        if not self.entrypoint.exists():
            raise FileNotFoundError(f"Agent entrypoint not found: {self.entrypoint}")
        if reasoning_effort:
            normalized_effort = reasoning_effort.lower()
            if normalized_effort not in {"low", "medium", "high"}:
                raise ValueError("reasoning_effort must be one of: low, medium, high.")
            self.reasoning_effort = normalized_effort
        else:
            self.reasoning_effort = None

        self.system_prompt = (
            "You are the HAL AutoInspector. Given a root-cause explanation and the "
            "tail of a failing trace, examine the HAL agent source code and produce "
            "a precise diagnostic for a coding agent to act on. You MUST NOT attempt "
            "to fix the code yourself. Instead, output guidance that includes:\n"
            "1) concise analysis of why the run failed,\n"
            "2) rationale referencing the evidence provided,\n"
            "3) concrete files or components the coding agent should inspect,\n"
            "4) recommended actions the coding agent should perform before re-running "
            "the debugger, and\n"
            "5) explicit reminder to re-run the debugger after applying changes.\n"
            "Respond strictly as JSON with the schema provided."
        )

        self._openai_client = self._init_openai_client()

    def generate_report(
        self,
        failure_context: Dict[str, Any],
        log_dir: Optional[Path] = None,
        context_blocks: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> InspectionReport:
        prompt = self._build_prompt(failure_context, context_blocks or [])
        response_text = self._call_model(prompt)
        try:
            report = self._parse_response(response_text)
        except ValueError:
            LOGGER.warning(
                "AutoInspector produced invalid JSON for task %s; retrying once",
                failure_context.get("task_id"),
            )
            self._write_raw_response(log_dir, response_text, attempt=0)
            retry_prompt = self._build_retry_prompt(prompt, response_text)
            response_text = self._call_model(retry_prompt)
            report = self._parse_response(response_text)

        return report

    def _init_openai_client(self):
        if OpenAI is None:
            return None
        try:
            return OpenAI()
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to initialize OpenAI client: %s", exc)
            return None

    def _load_agent_source(self) -> str:
        with self.entrypoint.open("r", encoding="utf-8") as handle:
            return handle.read()

    def _build_prompt(
        self,
        failure_context: Dict[str, Any],
        context_blocks: Sequence[Tuple[str, str]],
    ) -> str:
        agent_source = self._load_agent_source()
        explanation = failure_context.get("explanation", "").strip()
        trace_content = failure_context.get("trace_content", "").strip()

        extra_sections = "\n\n".join(
            f"[{title.upper()}]\n{body.strip()}" for title, body in context_blocks if body.strip()
        )

        return f"""
Task ID: {failure_context.get('task_id') or 'unknown'}

[ROOT CAUSE ANALYSIS]
{explanation or 'No explanation provided.'}

[ERROR TRACE TAIL]
{trace_content or 'Trace missing or empty.'}

[AGENT SOURCE CODE EXCERPT]
{agent_source}

{extra_sections}

Produce a JSON object with this exact schema:
{{
  "analysis": "<concise description of the failure>",
  "rationale": "<cite the evidence from trace/explanation>",
  "recommended_files": ["relative/path.py", "..."],
  "recommended_actions": [
    "Step-by-step actions a coding agent should perform to fix the issue"
  ],
  "next_steps": "Explicit reminder that the coding agent must re-run the HAL debugger after applying fixes."
}}

Do NOT include code patches. Focus on investigative guidance only.
""".strip()

    def _call_model(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if litellm is not None:
            temperature = 1 if self._requires_unity_temperature(self.model) else 0
            completion = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                reasoning_effort=self.reasoning_effort,
            )
            message = completion.choices[0].message
            if isinstance(message, dict):
                return message.get("content", "")
            return getattr(message, "content", "")

        if self._openai_client is not None:
            temperature = 1 if self._requires_unity_temperature(self.model) else 0
            kwargs = {
                "model": self.model,
                "input": messages,
                "temperature": temperature,
                "max_output_tokens": 2048,
            }
            if self.reasoning_effort:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            response = self._openai_client.responses.create(
                **kwargs,
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text
            collected: List[str] = []
            for item in getattr(response, "output", []):
                content = getattr(item, "content", [])
                for fragment in content:
                    text = getattr(fragment, "text", None)
                    if text:
                        collected.append(text)
            return "".join(collected)

        raise RuntimeError(
            "No LLM backend available. Install `litellm` or `openai`, "
            "or set HAL_AUTOFIX_MODEL to a supported provider."
        )

    def _parse_response(self, raw_response: str) -> InspectionReport:
        raw_response = raw_response.strip()
        if not raw_response:
            raise ValueError("AutoInspector returned an empty response")

        payload = self._extract_json(raw_response)
        analysis = (payload.get("analysis") or "").strip()
        rationale = (payload.get("rationale") or "").strip()
        recommended_files = payload.get("recommended_files") or []
        recommended_actions = payload.get("recommended_actions") or []
        next_steps = (payload.get("next_steps") or "").strip()

        if not analysis or not rationale:
            raise ValueError("Missing analysis or rationale in AutoInspector response")

        return InspectionReport(
            analysis=analysis,
            rationale=rationale,
            recommended_files=list(recommended_files),
            recommended_actions=list(recommended_actions),
            next_steps=next_steps,
        )

    @staticmethod
    def _extract_json(response_text: str) -> Dict[str, Any]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _build_retry_prompt(original_prompt: str, bad_response: str) -> str:
        return (
            f"{original_prompt}\n\n"
            "The previous answer was not valid JSON and could not be parsed. "
            "Here is what you returned:\n"
            "```\n"
            f"{bad_response.strip()}\n"
            "```\n"
            "Send the response again using STRICT JSON that matches the schema."
        )

    @staticmethod
    def _write_raw_response(
        log_dir: Optional[Path],
        content: str,
        attempt: int,
    ) -> None:
        if not log_dir:
            return
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            filename = log_dir / f"autoinspector_response_attempt_{attempt}.txt"
            filename.write_text(content, encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to write autoinspector raw response: %s", exc)

    @staticmethod
    def _requires_unity_temperature(model_name: str) -> bool:
        normalized = model_name.split("/")[-1]
        return normalized.startswith("o3") or normalized.startswith("o4") or "o-" in normalized
