"""
Shared utilities for HAL agents.

This module provides:
- Azure/TRAPI client utilities for direct Azure access
- Model parameter handling utilities
- Configuration loading utilities
- Agent compatibility wrappers

Usage:
    from shared.azure_utils import get_trapi_client, resolve_deployment_name
    from shared.model_utils import get_model_params, uses_max_completion_tokens
    from shared.config_utils import load_benchmark_config, build_agent_args
"""

from .azure_utils import (
    get_trapi_client,
    get_azure_client,
    resolve_deployment_name,
    TRAPI_DEPLOYMENT_MAP,
    DEPLOYMENT_TO_FRIENDLY,
    is_trapi_enabled,
    is_direct_azure_enabled,
    setup_direct_azure_env,
)

from .model_utils import (
    uses_max_completion_tokens,
    supports_temperature,
    supports_reasoning_effort,
    get_model_params,
    get_openai_request_params,
    normalize_model_id,
    is_reasoning_model,
    is_claude_model,
    is_deepseek_model,
    strip_thinking_tags,
)

from .config_utils import (
    load_benchmark_config,
    get_model_entries,
    get_model_entry,
    get_agent_info,
    build_agent_args,
    list_models_by_agent,
    list_agents,
    resolve_agent_path,
)

__all__ = [
    # Azure utilities
    "get_trapi_client",
    "get_azure_client",
    "resolve_deployment_name",
    "TRAPI_DEPLOYMENT_MAP",
    "DEPLOYMENT_TO_FRIENDLY",
    "is_trapi_enabled",
    "is_direct_azure_enabled",
    "setup_direct_azure_env",
    # Model utilities
    "uses_max_completion_tokens",
    "supports_temperature",
    "supports_reasoning_effort",
    "get_model_params",
    "get_openai_request_params",
    "normalize_model_id",
    "is_reasoning_model",
    "is_claude_model",
    "is_deepseek_model",
    "strip_thinking_tags",
    # Config utilities
    "load_benchmark_config",
    "get_model_entries",
    "get_model_entry",
    "get_agent_info",
    "build_agent_args",
    "list_models_by_agent",
    "list_agents",
    "resolve_agent_path",
]
