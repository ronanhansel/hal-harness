from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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
class FixSuggestion:
    fix_type: str  # "code" or "input"
    content: str
    rationale: str


class AutoFixer:
    """Use an LLM to propose an automated remediation for a failed task."""

    def __init__(
        self,
        agent_dir: str | os.PathLike[str],
        llm_model: str | None = None,
        entrypoint: str | None = None,
    ) -> None:
        self.agent_dir = Path(agent_dir)
        if not self.agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {self.agent_dir}")
        self.model = llm_model or os.getenv("HAL_AUTOFIX_MODEL", "gpt-4o-mini-2024-07-18")
        entrypoint_name = entrypoint or os.getenv("HAL_AUTOFIX_ENTRYPOINT", "main.py")
        self.entrypoint = self.agent_dir / entrypoint_name
        if not self.entrypoint.exists():
            raise FileNotFoundError(f"Agent entrypoint not found: {self.entrypoint}")

        self.system_prompt = (
            "You are the HAL AutoFixer. "
            "Given the root-cause analysis from a rubric and the raw error trace "
            "for a HAL evaluation run, produce a JSON plan that either updates the agent "
            "code or rewrites the benchmark problem statement. "
            "ALWAYS treat the rubric explanation as the authoritative diagnosis."
        )
        self._openai_client = self._init_openai_client()

    def generate_fix(self, failure_context: Dict[str, Any], log_dir: Optional[Path] = None) -> FixSuggestion:
        """Call the LLM and parse the structured fix recommendation."""
        prompt = self._build_prompt(failure_context)
        response_text = self._call_model(prompt)
        try:
            suggestion = self._parse_response(response_text)
        except ValueError:
            LOGGER.warning(
                "AutoFixer produced invalid JSON for task %s; retrying once",
                failure_context.get("task_id"),
            )
            self._write_raw_response(log_dir, response_text, attempt=0)
            retry_prompt = self._build_retry_prompt(prompt, response_text)
            response_text = self._call_model(retry_prompt)
            suggestion = self._parse_response(response_text)

        LOGGER.info(
            "AutoFixer selected %s fix for task %s",
            suggestion.fix_type,
            failure_context.get("task_id"),
        )
        return suggestion

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

    def _build_prompt(self, failure_context: Dict[str, Any]) -> str:
        agent_source = self._load_agent_source()
        explanation = failure_context.get("explanation", "").strip()
        trace_content = failure_context.get("trace_content", "").strip()

        prompt = f"""
Task ID: {failure_context.get('task_id') or 'unknown'}

[ROOT CAUSE ANALYSIS]
{explanation or 'No explanation provided.'}

[ERROR TRACE TAIL]
{trace_content or 'Trace missing or empty.'}

[AGENT SOURCE CODE: {self.entrypoint.name}]
{agent_source}

Your job is to determine whether the failure can be resolved by modifying the agent code
or by updating the input/problem statement that will be fed into the agent. Output STRICT JSON
with this exact schema:
{{
  "fix_type": "code" | "input",
  "content": "<updated full main.py file or revised problem statement text>",
  "rationale": "<short justification that cites the rubric explanation>"
}}

Rules:
1. If the rubric explanation points to tooling limits, sandbox setup, or missing repo files, prefer "code".
2. If the explanation blames ambiguous task instructions or missing context, prefer "input".
3. Provide the COMPLETE updated Python file when fix_type is "code".
4. NEVER include markdown fences or commentary outside the JSON object.
"""
        return prompt.strip()

    def _call_model(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if litellm is not None:
            completion = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=1,
                max_tokens=6000,
            )
            message = completion.choices[0].message
            if isinstance(message, dict):
                return message.get("content", "")
            return getattr(message, "content", "")

        if self._openai_client is not None:
            response = self._openai_client.responses.create(
                model=self.model,
                input=messages,
                temperature=0,
                max_output_tokens=2048,
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text
            # Fallback to concatenating all output segments.
            collected = []
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

    def _parse_response(self, raw_response: str) -> FixSuggestion:
        raw_response = raw_response.strip()
        if not raw_response:
            raise ValueError("AutoFixer returned an empty response")

        payload = self._extract_json(raw_response)
        fix_type = (payload.get("fix_type") or "input").strip().lower()
        if fix_type not in {"code", "input"}:
            LOGGER.warning("Unknown fix_type '%s'; defaulting to 'input'", fix_type)
            fix_type = "input"

        content = payload.get("content")
        rationale = payload.get("rationale", "").strip()
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Content missing from AutoFixer response")

        return FixSuggestion(fix_type=fix_type, content=content, rationale=rationale)

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
    def _write_raw_response(log_dir: Optional[Path], content: str, attempt: int) -> None:
        if not log_dir:
            return
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            filename = log_dir / f"autofixer_response_attempt_{attempt}.txt"
            filename.write_text(content, encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Failed to write autofixer raw response: %s", exc)
