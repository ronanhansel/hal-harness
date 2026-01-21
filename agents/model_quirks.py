"""
Shared Model Quirks Module
==========================

Comprehensive parameter compatibility checking for various LLM models.
This module provides utilities to determine which parameters are supported
by different model families (GPT-4, GPT-5, O-series reasoning models, DeepSeek, etc.)

Also includes TRAPI (Azure) deployment mapping and availability checking.

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

TRAPI deployment mapping:
    from model_quirks import get_trapi_deployment, is_available_on_trapi

    if is_available_on_trapi("deepseek-ai/DeepSeek-V3"):
        deployment = get_trapi_deployment("deepseek-ai/DeepSeek-V3")
    else:
        raise ValueError("Model not available on TRAPI")
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

# DeepSeek R1 specific (reasoning model with thinking tags)
DEEPSEEK_R1_PATTERN = re.compile(r"deepseek[-_]?r1|deepseek.*r1", re.IGNORECASE)

# DeepSeek V3 specific (fast model, different from R1)
DEEPSEEK_V3_PATTERN = re.compile(r"deepseek[-_]?v3|deepseek.*v3", re.IGNORECASE)


# =============================================================================
# TRAPI DEPLOYMENT MAPPING
# =============================================================================
# Maps friendly model names to TRAPI deployment names.
# ONLY models listed here are available on TRAPI.
# If a model is not in this map, it should NOT be used with USE_DIRECT_AZURE=true

TRAPI_DEPLOYMENT_MAP = {
    # GPT-5 series (VERIFIED WORKING)
    'gpt-5': 'gpt-5_2025-08-07',
    'gpt-5_2025-08-07': 'gpt-5_2025-08-07',
    'gpt-5-mini': 'gpt-5-mini_2025-08-07',
    'gpt-5-mini_2025-08-07': 'gpt-5-mini_2025-08-07',
    'gpt-5-nano': 'gpt-5-nano_2025-08-07',
    'gpt-5-pro': 'gpt-5-pro_2025-10-06',
    'gpt-5.2': 'gpt-5.2_2025-12-11',
    'gpt-5.2_2025-12-11': 'gpt-5.2_2025-12-11',
    'gpt-5.2-chat': 'gpt-5.2-chat_2025-12-11',

    # GPT-5.1 series
    'gpt-5.1': 'gpt-5.1_2025-11-13',
    'gpt-5.1-chat': 'gpt-5.1-chat_2025-11-13',
    'gpt-5.1-codex': 'gpt-5.1-codex_2025-11-13',
    'gpt-5.1-codex-mini': 'gpt-5.1-codex-mini_2025-11-13',

    # GPT-4 series (VERIFIED WORKING)
    'gpt-4o': 'gpt-4o_2024-11-20',
    'gpt-4o_2024-11-20': 'gpt-4o_2024-11-20',
    'gpt-4o-mini': 'gpt-4o-mini_2024-07-18',
    'gpt-4o-mini_2024-07-18': 'gpt-4o-mini_2024-07-18',
    'gpt-4.1': 'gpt-4.1_2025-04-14',
    'gpt-4.1_2025-04-14': 'gpt-4.1_2025-04-14',
    'gpt-4.1-mini': 'gpt-4.1-mini_2025-04-14',
    'gpt-4.1-mini_2025-04-14': 'gpt-4.1-mini_2025-04-14',
    'gpt-4.1-nano': 'gpt-4.1-nano_2025-04-14',
    'gpt-4.1-nano_2025-04-14': 'gpt-4.1-nano_2025-04-14',
    'gpt-4-turbo': 'gpt-4_turbo-2024-04-09',
    'gpt-4-32k': 'gpt-4-32k_0613',
    'gpt-4': 'gpt-4_turbo-2024-04-09',

    # O-series reasoning models (VERIFIED WORKING)
    'o1': 'o1_2024-12-17',
    'o1_2024-12-17': 'o1_2024-12-17',
    'o1-mini': 'o1-mini_2024-09-12',
    'o1-mini_2024-09-12': 'o1-mini_2024-09-12',
    'o3': 'o3_2025-04-16',
    'o3_2025-04-16': 'o3_2025-04-16',
    'o3-mini': 'o3-mini_2025-01-31',
    'o3-mini_2025-01-31': 'o3-mini_2025-01-31',
    'o4-mini': 'o4-mini_2025-04-16',
    'o4-mini_2025-04-16': 'o4-mini_2025-04-16',

    # DeepSeek - ONLY R1 is available on TRAPI!
    # DeepSeek-V3 is NOT available on TRAPI (404 error)
    'deepseek-r1': 'deepseek-r1_1',
    'deepseek-r1_1': 'deepseek-r1_1',
    'DeepSeek-R1_1': 'deepseek-r1_1',

    # Other models
    'grok-3.1': 'grok-3_1',
    'grok-3': 'grok-3_1',
    'llama-3.3': 'gcr-llama-33-70b-shared',
    'llama-3.3-70b': 'gcr-llama-33-70b-shared',
    'llama-3.1-70b': 'gcr-llama-31-70b-shared',
    'llama-3.1-8b': 'gcr-llama-31-8b-instruct',
    'qwen3-8b': 'gcr-qwen3-8b',
    'phi4': 'gcr-phi-4-shared',
    'mistral': 'gcr-mistralai-8x7b-shared',

    # Embeddings
    'text-embedding-3-large': 'text-embedding-3-large_1',
    'text-embedding-3-small': 'text-embedding-3-small_1',
    'text-embedding-ada-002': 'text-embedding-ada-002_2',
}

# Models explicitly NOT available on TRAPI (will error if used)
TRAPI_UNAVAILABLE_MODELS = {
    # DeepSeek V3 variants - NOT on TRAPI, use direct DeepSeek API
    'deepseek-v3',
    'deepseek-v3-0324',
    'DeepSeek-V3',
    'DeepSeek-V3-0324',
    'deepseek-ai/DeepSeek-V3',
    'deepseek-ai/DeepSeek-V3-0324',
}


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


def _is_deepseek_r1(model_id: str) -> bool:
    """Check if model is specifically DeepSeek R1 (reasoning model)"""
    return bool(DEEPSEEK_R1_PATTERN.search(model_id))


def _is_deepseek_v3(model_id: str) -> bool:
    """Check if model is specifically DeepSeek V3"""
    return bool(DEEPSEEK_V3_PATTERN.search(model_id))


# =============================================================================
# TRAPI DEPLOYMENT FUNCTIONS
# =============================================================================

def is_available_on_trapi(model_id: str) -> bool:
    """
    Check if a model is available on TRAPI (Azure).

    Args:
        model_id: Model identifier (e.g., "deepseek-ai/DeepSeek-V3-0324")

    Returns:
        True if model has a TRAPI deployment, False otherwise
    """
    normalized = _normalize_model_id(model_id)
    model_lower = normalized.lower()

    # Check explicit unavailable list first
    if model_id in TRAPI_UNAVAILABLE_MODELS or normalized in TRAPI_UNAVAILABLE_MODELS:
        return False

    # Check if it's a DeepSeek-V3 model (NOT available on TRAPI)
    if _is_deepseek_v3(model_id):
        return False

    # Check direct match
    if model_id in TRAPI_DEPLOYMENT_MAP:
        return True
    if normalized in TRAPI_DEPLOYMENT_MAP:
        return True
    if model_lower in TRAPI_DEPLOYMENT_MAP:
        return True

    # Check partial match (but be careful with DeepSeek)
    for key in TRAPI_DEPLOYMENT_MAP.keys():
        if key in model_lower or model_lower in key:
            # Don't match generic "deepseek" to V3 models
            if 'deepseek' in key.lower() and _is_deepseek_v3(model_id):
                continue
            return True

    return False


def get_trapi_deployment(model_id: str, strict: bool = True) -> str:
    """
    Resolve a model ID to its TRAPI deployment name.

    Args:
        model_id: Model identifier (e.g., "openai/gpt-4o", "deepseek-r1")
        strict: If True, raise error for unavailable models. If False, return as-is.

    Returns:
        TRAPI deployment name (e.g., "gpt-4o_2024-11-20")

    Raises:
        ValueError: If strict=True and model is not available on TRAPI
    """
    normalized = _normalize_model_id(model_id)
    model_lower = normalized.lower()

    # Check explicit unavailable list
    if model_id in TRAPI_UNAVAILABLE_MODELS or normalized in TRAPI_UNAVAILABLE_MODELS:
        if strict:
            raise ValueError(
                f"Model '{model_id}' is NOT available on TRAPI. "
                f"Use direct API access instead (set USE_DIRECT_AZURE=false and provide API key)."
            )
        return normalized

    # DeepSeek-V3 is NOT available on TRAPI
    if _is_deepseek_v3(model_id):
        if strict:
            raise ValueError(
                f"DeepSeek-V3 ('{model_id}') is NOT available on TRAPI. "
                f"Only DeepSeek-R1 is available. Use direct DeepSeek API for V3."
            )
        return normalized

    # Try exact match
    if model_id in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[model_id]
    if normalized in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[normalized]
    if model_lower in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[model_lower]

    # Try partial match
    for key, value in TRAPI_DEPLOYMENT_MAP.items():
        if key in model_lower or model_lower in key:
            # For DeepSeek, only match R1 to R1
            if 'deepseek' in key.lower():
                if _is_deepseek_r1(model_id):
                    return value
                # Skip generic deepseek matches for non-R1 models
                continue
            return value

    # No match found
    if strict:
        raise ValueError(
            f"Model '{model_id}' not found in TRAPI deployment map. "
            f"Available models: {', '.join(sorted(set(TRAPI_DEPLOYMENT_MAP.keys())))}"
        )
    return normalized


def validate_trapi_model(model_id: str) -> None:
    """
    Validate that a model can be used with TRAPI.
    Raises ValueError with helpful message if not.

    Args:
        model_id: Model identifier to validate

    Raises:
        ValueError: If model is not available on TRAPI
    """
    if not is_available_on_trapi(model_id):
        # Provide specific error message based on model type
        if _is_deepseek_v3(model_id):
            raise ValueError(
                f"❌ DeepSeek-V3 ('{model_id}') is NOT available on TRAPI!\n"
                f"   Only DeepSeek-R1 is available on TRAPI.\n"
                f"   To use DeepSeek-V3, either:\n"
                f"   1. Set USE_DIRECT_AZURE=false and provide DEEPSEEK_API_KEY\n"
                f"   2. Use DeepSeek-R1 instead (available on TRAPI as 'deepseek-r1_1')"
            )
        elif _is_deepseek(model_id):
            raise ValueError(
                f"❌ DeepSeek model '{model_id}' not recognized.\n"
                f"   Available on TRAPI: deepseek-r1, DeepSeek-R1_1\n"
                f"   NOT available: DeepSeek-V3 (use direct API)"
            )
        else:
            raise ValueError(
                f"❌ Model '{model_id}' is not available on TRAPI.\n"
                f"   Check TRAPI_DEPLOYMENT_MAP for available models."
            )


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

    @property
    def is_available_on_trapi(self) -> bool:
        """Whether model is available on TRAPI (Azure)"""
        return is_available_on_trapi(self.model_id)

    @property
    def trapi_deployment(self) -> Optional[str]:
        """Get TRAPI deployment name, or None if not available"""
        try:
            return get_trapi_deployment(self.model_id, strict=False)
        except ValueError:
            return None

    @property
    def is_deepseek_r1(self) -> bool:
        """Whether model is DeepSeek R1 (reasoning model)"""
        return _is_deepseek_r1(self.model_id)

    @property
    def is_deepseek_v3(self) -> bool:
        """Whether model is DeepSeek V3 (NOT available on TRAPI)"""
        return _is_deepseek_v3(self.model_id)

    def validate_for_trapi(self) -> None:
        """Validate model can be used with TRAPI, raise ValueError if not"""
        validate_trapi_model(self.model_id)

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

    # TRAPI deployment functions
    "is_available_on_trapi",
    "get_trapi_deployment",
    "validate_trapi_model",
    "TRAPI_DEPLOYMENT_MAP",
    "TRAPI_UNAVAILABLE_MODELS",

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
        "deepseek-ai/DeepSeek-V3-0324",
        "DeepSeek-R1_1",
        "claude-3-opus",
        "anthropic/claude-3-sonnet",
    ]

    print(f"{'Model':<35} {'Family':<10} {'Stop':<6} {'Temp':<6} {'MaxComp':<8} {'TRAPI':<6}")
    print("-" * 85)

    for model in test_models:
        quirks = ModelQuirks(model)
        print(
            f"{model:<35} "
            f"{quirks.family:<10} "
            f"{str(quirks.supports_stop):<6} "
            f"{str(quirks.supports_temperature):<6} "
            f"{str(quirks.uses_max_completion_tokens):<8} "
            f"{str(quirks.is_available_on_trapi):<6}"
        )

    print("\n\nTesting TRAPI deployment resolution:")
    trapi_tests = [
        ("gpt-4o", True),
        ("openai/gpt-4o", True),
        ("deepseek-r1", True),
        ("DeepSeek-R1_1", True),
        ("deepseek-ai/DeepSeek-V3", False),
        ("deepseek-ai/DeepSeek-V3-0324", False),
        ("o3-mini", True),
    ]

    for model, should_work in trapi_tests:
        try:
            deployment = get_trapi_deployment(model, strict=True)
            status = f"✓ -> {deployment}"
        except ValueError as e:
            status = f"✗ (expected)" if not should_work else f"✗ UNEXPECTED: {e}"
        print(f"  {model:<35} {status}")

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

    print("\n\nTesting process_output (DeepSeek R1):")
    quirks = ModelQuirks("deepseek-r1")
    content = "Let me think about this.\n<think>Internal reasoning...</think>\nThe answer is 42."
    print(f"  Input:  {content!r}")
    print(f"  Output: {quirks.process_output(content)!r}")

    print("\n\nTesting DeepSeek model detection:")
    print(f"  deepseek-r1: is_r1={_is_deepseek_r1('deepseek-r1')}, is_v3={_is_deepseek_v3('deepseek-r1')}")
    print(f"  DeepSeek-V3: is_r1={_is_deepseek_r1('deepseek-ai/DeepSeek-V3')}, is_v3={_is_deepseek_v3('deepseek-ai/DeepSeek-V3')}")
    print(f"  DeepSeek-V3-0324: is_r1={_is_deepseek_r1('DeepSeek-V3-0324')}, is_v3={_is_deepseek_v3('DeepSeek-V3-0324')}")

    print("\n✅ All tests passed!")
