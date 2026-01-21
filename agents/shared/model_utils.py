"""
Model Parameter Utilities for HAL Agents

This module provides utilities for handling model-specific parameters across
different model types (GPT-4, GPT-5, O-series, Claude, etc.).

Usage:
    from shared.model_utils import get_model_params, uses_max_completion_tokens

    params = get_model_params(
        model_id="openai/gpt-5_2025-08-07",
        reasoning_effort="medium",
        temperature=0.7,
        max_tokens=4096,
    )
"""

import re
from typing import Any, Dict, Optional


def normalize_model_id(model_id: str) -> str:
    """
    Normalize model ID to a consistent format.
    Removes common prefixes and normalizes separators.

    Args:
        model_id: Raw model identifier

    Returns:
        Normalized model identifier

    Examples:
        >>> normalize_model_id("openai/gpt-4.1_2025-04-14")
        'gpt-4.1_2025-04-14'
        >>> normalize_model_id("azure/o3-mini")
        'o3-mini'
    """
    # Remove provider prefixes
    for prefix in ['openai/', 'azure/', 'anthropic/', 'google/', 'deepseek/']:
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix):]
            break
    return model_id


def uses_max_completion_tokens(model_id: str) -> bool:
    """
    Check if model uses max_completion_tokens instead of max_tokens.
    O-series and GPT-5+ models use max_completion_tokens.

    Args:
        model_id: Model identifier (with or without provider prefix)

    Returns:
        True if model uses max_completion_tokens, False otherwise

    Examples:
        >>> uses_max_completion_tokens("o3-mini")
        True
        >>> uses_max_completion_tokens("gpt-5_2025-08-07")
        True
        >>> uses_max_completion_tokens("gpt-4o")
        False
    """
    model_lower = normalize_model_id(model_id).lower()

    # O-series models (o1, o3, o4-mini, etc.)
    if model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        return True

    # GPT-5+ series
    if model_lower.startswith('gpt-5') or 'gpt-5' in model_lower:
        return True

    return False


def supports_temperature(model_id: str) -> bool:
    """
    Check if model supports temperature parameter.
    O-series and GPT-5 models don't support temperature (only default=1).

    Args:
        model_id: Model identifier

    Returns:
        True if model supports temperature, False otherwise

    Examples:
        >>> supports_temperature("gpt-4o")
        True
        >>> supports_temperature("o3-mini")
        False
        >>> supports_temperature("gpt-5_2025-08-07")
        False
    """
    model_lower = normalize_model_id(model_id).lower()

    # O-series models don't support temperature
    if model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        return False

    # GPT-5 only supports temperature=1 (default)
    if model_lower.startswith('gpt-5') or 'gpt-5' in model_lower:
        return False

    # DeepSeek reasoning models (R1) don't support temperature
    if 'deepseek-r1' in model_lower or 'deepseek_r1' in model_lower:
        return False

    return True


def supports_top_p(model_id: str) -> bool:
    """
    Check if model supports top_p parameter.
    O-series and GPT-5 models don't support top_p.

    Args:
        model_id: Model identifier

    Returns:
        True if model supports top_p, False otherwise

    Examples:
        >>> supports_top_p("gpt-4o")
        True
        >>> supports_top_p("o3-mini")
        False
        >>> supports_top_p("gpt-5_2025-08-07")
        False
    """
    # top_p support follows the same rules as temperature
    return supports_temperature(model_id)


def supports_reasoning_effort(model_id: str) -> bool:
    """
    Check if model supports reasoning_effort parameter.

    Args:
        model_id: Model identifier

    Returns:
        True if model supports reasoning_effort, False otherwise
    """
    model_lower = normalize_model_id(model_id).lower()

    # O-series models support reasoning_effort
    if model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        return True

    # GPT-5 series supports reasoning_effort
    if model_lower.startswith('gpt-5') or 'gpt-5' in model_lower:
        return True

    return False


def supports_stop(model_id: str) -> bool:
    """
    Check if model supports stop parameter.
    Reasoning models have limited stop support.

    Args:
        model_id: Model identifier

    Returns:
        True if model supports stop parameter, False otherwise
    """
    model_lower = normalize_model_id(model_id).lower()

    # O-series models have limited stop support
    if model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        return False

    # GPT-5 may have limited stop support
    if model_lower.startswith('gpt-5') or 'gpt-5' in model_lower:
        return False

    return True


def is_reasoning_model(model_id: str) -> bool:
    """
    Check if model is a reasoning model (O-series, GPT-5, DeepSeek-R1).

    Args:
        model_id: Model identifier

    Returns:
        True if model is a reasoning model, False otherwise
    """
    model_lower = normalize_model_id(model_id).lower()

    # O-series
    if model_lower.startswith('o1') or model_lower.startswith('o3') or model_lower.startswith('o4'):
        return True

    # GPT-5 series
    if model_lower.startswith('gpt-5') or 'gpt-5' in model_lower:
        return True

    # DeepSeek R1
    if 'deepseek-r1' in model_lower or 'deepseek_r1' in model_lower:
        return True

    return False


def is_claude_model(model_id: str) -> bool:
    """Check if model is a Claude model."""
    model_lower = normalize_model_id(model_id).lower()
    return 'claude' in model_lower


def is_deepseek_model(model_id: str) -> bool:
    """Check if model is a DeepSeek model."""
    model_lower = normalize_model_id(model_id).lower()
    return 'deepseek' in model_lower


def is_gemini_model(model_id: str) -> bool:
    """Check if model is a Gemini model."""
    model_lower = normalize_model_id(model_id).lower()
    return 'gemini' in model_lower


def get_model_params(
    model_id: str,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build model parameters dict with appropriate settings for the model type.
    Automatically filters out unsupported parameters for reasoning models.

    Args:
        model_id: Model identifier
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
        temperature: Temperature setting (ignored for reasoning models)
        max_tokens: Max tokens for completion
        **kwargs: Additional parameters to include

    Returns:
        Dict of model parameters suitable for the model type

    Examples:
        >>> get_model_params("gpt-4o", temperature=0.7, max_tokens=4096)
        {'model_id': 'gpt-4o', 'temperature': 0.7, 'max_tokens': 4096}

        >>> get_model_params("o3-mini", reasoning_effort="high", max_tokens=4096)
        {'model_id': 'o3-mini', 'reasoning_effort': 'high', 'max_completion_tokens': 4096}

        >>> get_model_params("gpt-5_2025-08-07", reasoning_effort="medium", max_tokens=4096)
        {'model_id': 'gpt-5_2025-08-07', 'reasoning_effort': 'medium', 'max_completion_tokens': 4096}
    """
    params: Dict[str, Any] = {'model_id': model_id}

    # Add reasoning_effort if supported
    if reasoning_effort and supports_reasoning_effort(model_id):
        params['reasoning_effort'] = reasoning_effort

    # Add temperature only if supported
    if temperature is not None and supports_temperature(model_id):
        params['temperature'] = temperature

    # Use appropriate max tokens parameter
    if max_tokens is not None:
        if uses_max_completion_tokens(model_id):
            params['max_completion_tokens'] = max_tokens
        else:
            params['max_tokens'] = max_tokens

    # Add any additional kwargs
    params.update(kwargs)

    return params


def get_openai_request_params(
    model_id: str,
    messages: list,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 16384,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build request parameters for OpenAI API calls.
    Handles model-specific parameter requirements.

    Args:
        model_id: Model identifier (will be used as 'model' in request)
        messages: List of message dicts
        reasoning_effort: Reasoning effort level
        temperature: Temperature setting
        max_tokens: Max tokens for completion
        **kwargs: Additional parameters

    Returns:
        Dict ready to be passed to client.chat.completions.create()
    """
    params: Dict[str, Any] = {
        'model': model_id,
        'messages': messages,
    }

    # Add reasoning_effort if supported
    if reasoning_effort and supports_reasoning_effort(model_id):
        params['reasoning_effort'] = reasoning_effort

    # Add temperature only if supported
    if temperature is not None and supports_temperature(model_id):
        params['temperature'] = temperature

    # Use appropriate max tokens parameter
    if uses_max_completion_tokens(model_id):
        params['max_completion_tokens'] = max_tokens
    else:
        params['max_tokens'] = max_tokens

    # Add extra headers for DeepSeek models
    if is_deepseek_model(model_id):
        params['extra_headers'] = kwargs.pop('extra_headers', {})
        params['extra_headers']['extra-parameters'] = 'pass-through'

    # Add remaining kwargs (filtering out None values)
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value

    return params


def strip_thinking_tags(content: str, model_id: str) -> str:
    """
    Strip thinking tags from model output if applicable.
    DeepSeek R1 and O1 models may include <think>...</think> tags.

    Args:
        content: Model output content
        model_id: Model identifier

    Returns:
        Content with thinking tags removed
    """
    if not content:
        return content

    model_lower = normalize_model_id(model_id).lower()

    # Strip thinking tags for DeepSeek R1 and O1 models
    if 'deepseek' in model_lower or model_lower.startswith('o1'):
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    return content.strip()


# =============================================================================
# Agent argument helpers
# =============================================================================

def build_agent_args_from_config(
    model_config: Dict[str, Any],
    config_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Build agent_args dict from model configuration.

    Args:
        model_config: Full model configuration dict (from model_to_baseline_*.json)
        config_key: Key to look up in config

    Returns:
        Dict suitable for passing to agent run() function, or None if not found

    Example config entry:
        {
            "openai/gpt-4.1-2025-04-14": {
                "model_id": "openai/gpt-4.1_2025-04-14",
                "short_name": "gpt-4.1",
                "reasoning_effort": "high",
                "max_steps": 5
            }
        }

    Returns:
        {
            "model_name": "openai/gpt-4.1_2025-04-14",
            "reasoning_effort": "high",
            "max_steps": 5
        }
    """
    entry = model_config.get(config_key)
    if not entry:
        return None

    model_id = entry.get('model_id')
    if not model_id:
        return None

    args: Dict[str, Any] = {
        'model_name': model_id,
    }

    # Copy relevant parameters
    for key in ['reasoning_effort', 'temperature', 'max_steps', 'budget']:
        if key in entry:
            args[key] = entry[key]

    return args
