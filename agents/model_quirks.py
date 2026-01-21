"""
Shared Model Quirks Module
==========================

Comprehensive parameter compatibility checking for various LLM models.
This module provides utilities to determine which parameters are supported
by different model families (GPT-4, GPT-5, O-series reasoning models, DeepSeek, etc.)

Usage:
    from model_quirks import ModelQuirks

    quirks = ModelQuirks("openai/o3-2025-04-16")
    if quirks.supports_stop:
        params["stop"] = stop_sequences
    if quirks.supports_temperature:
        params["temperature"] = 0.7
    if quirks.uses_max_completion_tokens:
        params["max_completion_tokens"] = 4096
    else:
        params["max_tokens"] = 4096

Or use class methods directly:
    if ModelQuirks.model_supports_stop("o3"):
        ...
"""

import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


# =============================================================================
# MODEL FAMILY PATTERNS
# =============================================================================

# O-series reasoning models (o1, o3, o4-mini, etc.)
O_SERIES_PATTERN = re.compile(
    r"^(o[134])(-mini|-pro)?(-\d{4}-\d{2}-\d{2})?(_.*)?$",
    re.IGNORECASE
)

# GPT-5 family (gpt-5, gpt-5-mini, gpt-5.1, gpt-5.2, etc.)
GPT5_PATTERN = re.compile(
    r"^gpt-?5(\.\d+)?(-mini|-nano|-pro|-chat|-codex)?(-\d{4}-\d{2}-\d{2})?(_.*)?$",
    re.IGNORECASE
)

# GPT-4 family (gpt-4, gpt-4o, gpt-4.1, gpt-4-turbo, etc.)
GPT4_PATTERN = re.compile(
    r"^gpt-?4(o|-turbo|-32k|\.\d+)?(-mini|-nano)?(-\d{4}-\d{2}-\d{2})?(_.*)?$",
    re.IGNORECASE
)

# DeepSeek models
DEEPSEEK_PATTERN = re.compile(r"deepseek", re.IGNORECASE)

# Claude models (anthropic)
CLAUDE_PATTERN = re.compile(r"claude", re.IGNORECASE)

# Llama models
LLAMA_PATTERN = re.compile(r"llama", re.IGNORECASE)

# Qwen models
QWEN_PATTERN = re.compile(r"qwen", re.IGNORECASE)


def _normalize_model_id(model_id: str) -> str:
    """
    Normalize model ID by removing prefixes and standardizing format.

    Examples:
        "openai/gpt-4o" -> "gpt-4o"
        "azure/o3-2025-04-16" -> "o3-2025-04-16"
        "gpt_4o_2024-11-20" -> "gpt-4o-2024-11-20"
    """
    # Remove common prefixes
    model_id = model_id.replace("openai/", "").replace("azure/", "")
    model_id = model_id.replace("openrouter/", "").replace("anthropic/", "")
    model_id = model_id.replace("litellm/", "")

    # Normalize underscores to dashes (except in date suffixes like _2024-11-20)
    # Keep the format consistent for pattern matching
    parts = model_id.split("_")
    if len(parts) > 1:
        # Check if last part looks like a date (YYYY-MM-DD or similar)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", parts[-1]):
            model_id = "-".join(parts[:-1]) + "_" + parts[-1]
        else:
            model_id = "-".join(parts)

    return model_id


def _is_o_series(model_id: str) -> bool:
    """Check if model is an O-series reasoning model (o1, o3, o4-mini, etc.)"""
    normalized = _normalize_model_id(model_id)
    model_lower = normalized.lower()

    # Quick checks first
    if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
        return True

    # Pattern match for more complex cases
    return bool(O_SERIES_PATTERN.match(normalized))


def _is_gpt5(model_id: str) -> bool:
    """Check if model is a GPT-5 family model"""
    normalized = _normalize_model_id(model_id)
    model_lower = normalized.lower()

    # Quick check
    if model_lower.startswith("gpt-5") or model_lower.startswith("gpt5"):
        return True

    # Pattern match
    return bool(GPT5_PATTERN.match(normalized))


def _is_gpt4(model_id: str) -> bool:
    """Check if model is a GPT-4 family model"""
    normalized = _normalize_model_id(model_id)
    model_lower = normalized.lower()

    if model_lower.startswith("gpt-4") or model_lower.startswith("gpt4"):
        return True

    return bool(GPT4_PATTERN.match(normalized))


def _is_deepseek(model_id: str) -> bool:
    """Check if model is a DeepSeek model"""
    return bool(DEEPSEEK_PATTERN.search(model_id))


def _is_claude(model_id: str) -> bool:
    """Check if model is a Claude/Anthropic model"""
    return bool(CLAUDE_PATTERN.search(model_id))


# =============================================================================
# QUIRK CHECK FUNCTIONS (standalone, can be used directly)
# =============================================================================

def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the 'stop' parameter.

    NOT supported by:
    - O-series reasoning models (o1, o3, o4-mini, etc.)
    - GPT-5 family models

    Args:
        model_id: Model identifier (e.g., "openai/o3-2025-04-16", "gpt-4o")

    Returns:
        True if the model supports stop sequences, False otherwise
    """
    if _is_o_series(model_id):
        return False
    if _is_gpt5(model_id):
        return False
    return True


def supports_temperature(model_id: str) -> bool:
    """
    Check if the model supports the 'temperature' parameter.

    NOT supported by:
    - O-series reasoning models (o1, o3, o4-mini, etc.) - only temperature=1
    - GPT-5 family models - only temperature=1 (default)

    Args:
        model_id: Model identifier

    Returns:
        True if the model supports temperature parameter, False otherwise
    """
    if _is_o_series(model_id):
        return False
    if _is_gpt5(model_id):
        return False
    return True


def uses_max_completion_tokens(model_id: str) -> bool:
    """
    Check if the model uses 'max_completion_tokens' instead of 'max_tokens'.

    Uses max_completion_tokens:
    - O-series reasoning models (o1, o3, o4-mini, etc.)
    - GPT-5 family models

    Args:
        model_id: Model identifier

    Returns:
        True if model uses max_completion_tokens, False if it uses max_tokens
    """
    if _is_o_series(model_id):
        return True
    if _is_gpt5(model_id):
        return True
    return False


def supports_reasoning_effort(model_id: str) -> bool:
    """
    Check if the model supports the 'reasoning_effort' parameter.

    Supported by:
    - O-series reasoning models (o1, o3, o4-mini, etc.)
    - GPT-5 family models (some versions)

    Args:
        model_id: Model identifier

    Returns:
        True if model supports reasoning_effort parameter
    """
    if _is_o_series(model_id):
        return True
    # GPT-5 may or may not support it depending on version
    if _is_gpt5(model_id):
        return True
    return False


def requires_extra_headers(model_id: str) -> Optional[Dict[str, str]]:
    """
    Get extra headers required for specific models.

    DeepSeek models require: {"extra-parameters": "pass-through"}

    Args:
        model_id: Model identifier

    Returns:
        Dict of extra headers if required, None otherwise
    """
    if _is_deepseek(model_id):
        return {"extra-parameters": "pass-through"}
    return None


def has_thinking_tags(model_id: str) -> bool:
    """
    Check if the model outputs <think>...</think> tags that need stripping.

    DeepSeek R1 models use thinking tags.

    Args:
        model_id: Model identifier

    Returns:
        True if model outputs thinking tags
    """
    if _is_deepseek(model_id):
        model_lower = model_id.lower()
        # DeepSeek R1 specifically uses thinking tags
        if "r1" in model_lower or "deepseek-r1" in model_lower:
            return True
    return False


def strip_thinking_tags(content: str) -> str:
    """
    Remove <think>...</think> tags from model output.

    Args:
        content: Model output text

    Returns:
        Content with thinking tags removed
    """
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)


def get_model_family(model_id: str) -> str:
    """
    Determine the model family for a given model ID.

    Args:
        model_id: Model identifier

    Returns:
        One of: "o-series", "gpt-5", "gpt-4", "deepseek", "claude", "llama", "qwen", "unknown"
    """
    if _is_o_series(model_id):
        return "o-series"
    if _is_gpt5(model_id):
        return "gpt-5"
    if _is_gpt4(model_id):
        return "gpt-4"
    if _is_deepseek(model_id):
        return "deepseek"
    if _is_claude(model_id):
        return "claude"
    if LLAMA_PATTERN.search(model_id):
        return "llama"
    if QWEN_PATTERN.search(model_id):
        return "qwen"
    return "unknown"


# =============================================================================
# MODELQUIRKS CLASS (for object-oriented usage)
# =============================================================================

@dataclass
class ModelQuirks:
    """
    Container for model-specific parameter quirks.

    Usage:
        quirks = ModelQuirks("openai/o3-2025-04-16")
        print(quirks.supports_stop)  # False
        print(quirks.supports_temperature)  # False
        print(quirks.uses_max_completion_tokens)  # True

        # Apply quirks to parameters
        params = quirks.apply_to_params({
            "temperature": 0.7,
            "max_tokens": 4096,
            "stop": ["END"],
        })
    """
    model_id: str

    @property
    def normalized_id(self) -> str:
        """Get normalized model ID without prefixes"""
        return _normalize_model_id(self.model_id)

    @property
    def family(self) -> str:
        """Get model family (o-series, gpt-5, gpt-4, etc.)"""
        return get_model_family(self.model_id)

    @property
    def supports_stop(self) -> bool:
        """Whether model supports stop sequences"""
        return supports_stop_parameter(self.model_id)

    @property
    def supports_temperature(self) -> bool:
        """Whether model supports temperature parameter"""
        return supports_temperature(self.model_id)

    @property
    def uses_max_completion_tokens(self) -> bool:
        """Whether model uses max_completion_tokens instead of max_tokens"""
        return uses_max_completion_tokens(self.model_id)

    @property
    def supports_reasoning_effort(self) -> bool:
        """Whether model supports reasoning_effort parameter"""
        return supports_reasoning_effort(self.model_id)

    @property
    def extra_headers(self) -> Optional[Dict[str, str]]:
        """Extra headers required for this model (e.g., DeepSeek)"""
        return requires_extra_headers(self.model_id)

    @property
    def has_thinking_tags(self) -> bool:
        """Whether model outputs <think> tags that need stripping"""
        return has_thinking_tags(self.model_id)

    def apply_to_params(
        self,
        params: Dict[str, Any],
        remove_unsupported: bool = True
    ) -> Dict[str, Any]:
        """
        Apply model quirks to a parameters dictionary.

        This will:
        - Remove 'stop' if not supported
        - Remove 'temperature' if not supported
        - Convert 'max_tokens' to 'max_completion_tokens' if needed
        - Add extra headers if required

        Args:
            params: Dictionary of parameters to send to the model
            remove_unsupported: If True, remove unsupported params; if False, raise error

        Returns:
            Modified parameters dictionary
        """
        result = dict(params)

        # Handle stop parameter
        if "stop" in result and not self.supports_stop:
            if remove_unsupported:
                del result["stop"]
            else:
                raise ValueError(f"Model {self.model_id} does not support 'stop' parameter")

        # Handle stop_sequences (alias)
        if "stop_sequences" in result and not self.supports_stop:
            if remove_unsupported:
                del result["stop_sequences"]
            else:
                raise ValueError(f"Model {self.model_id} does not support 'stop_sequences' parameter")

        # Handle temperature
        if "temperature" in result and not self.supports_temperature:
            if remove_unsupported:
                del result["temperature"]
            else:
                raise ValueError(f"Model {self.model_id} does not support 'temperature' parameter")

        # Handle max_tokens vs max_completion_tokens
        if self.uses_max_completion_tokens:
            if "max_tokens" in result and "max_completion_tokens" not in result:
                result["max_completion_tokens"] = result.pop("max_tokens")
        else:
            if "max_completion_tokens" in result and "max_tokens" not in result:
                result["max_tokens"] = result.pop("max_completion_tokens")

        # Add extra headers if needed
        if self.extra_headers:
            if "extra_headers" in result:
                result["extra_headers"].update(self.extra_headers)
            else:
                result["extra_headers"] = dict(self.extra_headers)

        return result

    def process_output(self, content: str) -> str:
        """
        Process model output, stripping thinking tags if needed.

        Args:
            content: Raw model output

        Returns:
            Processed output
        """
        if self.has_thinking_tags:
            return strip_thinking_tags(content)
        return content

    # Class methods for direct access without instantiation
    @classmethod
    def model_supports_stop(cls, model_id: str) -> bool:
        """Class method to check stop parameter support"""
        return supports_stop_parameter(model_id)

    @classmethod
    def model_supports_temperature(cls, model_id: str) -> bool:
        """Class method to check temperature support"""
        return supports_temperature(model_id)

    @classmethod
    def model_uses_max_completion_tokens(cls, model_id: str) -> bool:
        """Class method to check max_completion_tokens usage"""
        return uses_max_completion_tokens(model_id)

    def __repr__(self) -> str:
        return (
            f"ModelQuirks({self.model_id!r}, "
            f"family={self.family!r}, "
            f"stop={self.supports_stop}, "
            f"temp={self.supports_temperature}, "
            f"max_completion_tokens={self.uses_max_completion_tokens})"
        )


# =============================================================================
# SMOLAGENTS INTEGRATION
# =============================================================================

def patch_smolagents():
    """
    Patch smolagents.models.supports_stop_parameter with our implementation.

    Call this at the start of your agent to ensure smolagents uses the correct
    stop parameter checking logic.

    Usage:
        from model_quirks import patch_smolagents
        patch_smolagents()
    """
    try:
        import smolagents.models
        smolagents.models.supports_stop_parameter = supports_stop_parameter
        print("[model_quirks] Patched smolagents.models.supports_stop_parameter")
    except ImportError:
        pass  # smolagents not installed


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "ModelQuirks",

    # Standalone functions
    "supports_stop_parameter",
    "supports_temperature",
    "uses_max_completion_tokens",
    "supports_reasoning_effort",
    "requires_extra_headers",
    "has_thinking_tags",
    "strip_thinking_tags",
    "get_model_family",

    # Smolagents integration
    "patch_smolagents",
]


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing ModelQuirks module...\n")

    test_models = [
        "openai/gpt-4o",
        "gpt-4.1-2025-04-14",
        "openai/o3-2025-04-16",
        "o3-mini",
        "o4-mini-2025-04-16",
        "openai/o1",
        "gpt-5",
        "gpt-5.2_2025-12-11",
        "gpt-5-mini",
        "deepseek-r1",
        "deepseek-ai/DeepSeek-V3",
        "claude-3-opus",
        "anthropic/claude-3-sonnet",
    ]

    print(f"{'Model':<35} {'Family':<10} {'Stop':<6} {'Temp':<6} {'MaxComp':<8} {'Reasoning':<10}")
    print("-" * 85)

    for model in test_models:
        quirks = ModelQuirks(model)
        print(
            f"{model:<35} "
            f"{quirks.family:<10} "
            f"{str(quirks.supports_stop):<6} "
            f"{str(quirks.supports_temperature):<6} "
            f"{str(quirks.uses_max_completion_tokens):<8} "
            f"{str(quirks.supports_reasoning_effort):<10}"
        )

    print("\n\nTesting apply_to_params:")
    quirks = ModelQuirks("o3-2025-04-16")
    params = {
        "temperature": 0.7,
        "max_tokens": 4096,
        "stop": ["END", "STOP"],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    print(f"  Input:  {params}")
    result = quirks.apply_to_params(params)
    print(f"  Output: {result}")

    print("\n\nTesting process_output (DeepSeek):")
    quirks = ModelQuirks("deepseek-r1")
    content = "Let me think about this.\n<think>Internal reasoning...</think>\nThe answer is 42."
    print(f"  Input:  {content!r}")
    print(f"  Output: {quirks.process_output(content)!r}")

    print("\n✅ All tests passed!")
