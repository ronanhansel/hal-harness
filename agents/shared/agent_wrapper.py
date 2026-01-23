"""
Agent Wrapper for Azure Compatibility

This module provides a simple way to make any agent Azure-compatible by
wrapping the model creation and API call logic.

Usage in agent main.py:
    import sys
    from pathlib import Path

    # Add shared module to path
    _agents_dir = Path(__file__).resolve().parent.parent
    if str(_agents_dir) not in sys.path:
        sys.path.insert(0, str(_agents_dir))

    from shared.agent_wrapper import (
        create_azure_client,
        create_model_for_agent,
        setup_azure_env,
        make_chat_completion,
    )

    def run(input: dict, **kwargs) -> dict:
        setup_azure_env()  # Remove proxy URLs if using direct Azure

        # Option 1: Get OpenAI-compatible client
        client = create_azure_client(kwargs.get('model_name', 'gpt-4o'))
        response = make_chat_completion(
            client=client,
            model_name=kwargs['model_name'],
            messages=[...],
            reasoning_effort=kwargs.get('reasoning_effort'),
        )

        # Option 2: For smolagents - get a LiteLLMModel or AzureDirectModel
        model = create_model_for_agent(
            model_name=kwargs['model_name'],
            reasoning_effort=kwargs.get('reasoning_effort'),
            temperature=kwargs.get('temperature'),
        )
        # Use with CodeAgent, etc.
"""

import os
import re
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

# Import from sibling modules
from .azure_utils import (
    get_trapi_client,
    resolve_deployment_name,
    is_trapi_enabled,
    is_direct_azure_enabled,
    setup_direct_azure_env,
)
from .model_utils import (
    uses_max_completion_tokens,
    supports_temperature,
    supports_top_p,
    supports_reasoning_effort,
    is_deepseek_model,
    strip_thinking_tags,
)


def setup_azure_env() -> None:
    """
    Setup environment for Azure/TRAPI access.
    Call this at the start of your agent's run() function.
    """
    setup_direct_azure_env()


def create_azure_client(model_name: str = "gpt-4o") -> Union[OpenAI, Any]:
    """
    Create an OpenAI-compatible client for Azure/TRAPI or standard OpenAI.

    Args:
        model_name: Model name (used to determine if TRAPI is needed)

    Returns:
        OpenAI or AzureOpenAI client

    Usage:
        client = create_azure_client("gpt-4o")
        response = client.chat.completions.create(
            model=resolve_deployment_name("gpt-4o"),
            messages=[...]
        )
    """
    # Use TRAPI for OpenAI/Azure models
    if is_trapi_enabled() or is_direct_azure_enabled():
        model_lower = model_name.lower()
        # Check if it's an OpenAI/Azure model
        if any(x in model_lower for x in ['gpt', 'o1', 'o3', 'o4', 'deepseek']):
            return get_trapi_client()

    # Fall back to standard OpenAI client
    return OpenAI()


def create_model_for_agent(
    model_name: str,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> Any:
    """
    Create a model instance for use with smolagents CodeAgent.

    This function returns either:
    - AzureDirectModel if USE_DIRECT_AZURE is true
    - LiteLLMModel otherwise

    Args:
        model_name: Model name/ID
        reasoning_effort: Reasoning effort level
        temperature: Temperature setting
        **kwargs: Additional model parameters

    Returns:
        Model instance compatible with smolagents
    """
    model_params: Dict[str, Any] = {'model_id': model_name}

    # Add reasoning effort if supported
    if reasoning_effort and supports_reasoning_effort(model_name):
        model_params['reasoning_effort'] = reasoning_effort

    # Add temperature if supported
    if temperature is not None and supports_temperature(model_name):
        model_params['temperature'] = temperature

    # Add any additional kwargs
    model_params.update(kwargs)

    # Use AzureDirectModel if direct Azure mode is enabled
    if is_direct_azure_enabled():
        try:
            from .azure_direct_model import AzureDirectModel
            return AzureDirectModel(**model_params)
        except ImportError:
            # Fallback: Try importing from hal_generalist_agent
            try:
                import sys
                from pathlib import Path
                hal_agent_dir = Path(__file__).parent.parent / "hal_generalist_agent"
                if str(hal_agent_dir) not in sys.path:
                    sys.path.insert(0, str(hal_agent_dir))
                from azure_direct_model import AzureDirectModel
                return AzureDirectModel(**model_params)
            except ImportError:
                raise RuntimeError(
                    "Direct Azure is enabled but AzureDirectModel could not be imported. "
                    "Install dependencies or fix agent packaging."
                )

    # Use LiteLLMModel (default)
    try:
        from smolagents import LiteLLMModel
        return LiteLLMModel(**model_params)
    except ImportError:
        raise ImportError("smolagents not installed. Install with: pip install smolagents")


def make_chat_completion(
    client: Union[OpenAI, Any],
    model_name: str,
    messages: List[Dict[str, str]],
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: int = 16384,
    **kwargs,
) -> str:
    """
    Make a chat completion request with proper parameter handling.

    Automatically filters unsupported parameters based on model type:
    - O-series and GPT-5: No temperature, top_p; use max_completion_tokens
    - DeepSeek-R1: No temperature; add extra_headers
    - Standard models: All params supported

    Args:
        client: OpenAI or AzureOpenAI client
        model_name: Model name
        messages: List of message dicts
        reasoning_effort: Reasoning effort level
        temperature: Temperature setting (filtered for reasoning models)
        top_p: Top-p setting (filtered for reasoning models)
        max_tokens: Max tokens for completion
        **kwargs: Additional parameters

    Returns:
        Response content string
    """
    # Resolve deployment name for Azure
    if hasattr(client, 'azure_endpoint'):
        model = resolve_deployment_name(model_name)
    else:
        model = model_name

    # Build request parameters
    params: Dict[str, Any] = {
        'model': model,
        'messages': messages,
    }

    # Add max tokens with correct parameter name
    if uses_max_completion_tokens(model_name):
        params['max_completion_tokens'] = max_tokens
    else:
        params['max_tokens'] = max_tokens

    # Add temperature only if supported
    if temperature is not None and supports_temperature(model_name):
        params['temperature'] = temperature

    # Add top_p only if supported
    if top_p is not None and supports_top_p(model_name):
        params['top_p'] = top_p

    # Add reasoning effort if supported
    if reasoning_effort and supports_reasoning_effort(model_name):
        params['reasoning_effort'] = reasoning_effort

    # Add extra headers for DeepSeek models
    if is_deepseek_model(model_name):
        params['extra_headers'] = kwargs.pop('extra_headers', {})
        params['extra_headers']['extra-parameters'] = 'pass-through'

    # Filter out unsupported params from kwargs
    filtered_kwargs = {}
    for key, value in kwargs.items():
        # Skip params that were already handled or are known unsupported
        if key in ('temperature', 'top_p') and not supports_temperature(model_name):
            continue
        filtered_kwargs[key] = value

    # Add remaining kwargs
    params.update(filtered_kwargs)

    # Make request
    response = client.chat.completions.create(**params)
    content = response.choices[0].message.content or ""

    # Strip thinking tags if needed
    content = strip_thinking_tags(content, model_name)

    return content


def get_model_deployment(model_name: str) -> str:
    """
    Get the deployment name for a model (for Azure/TRAPI).

    Args:
        model_name: Model name

    Returns:
        Deployment name for Azure API calls
    """
    return resolve_deployment_name(model_name)


# =============================================================================
# Convenience class for agents that need more control
# =============================================================================

class AzureCompatibleAgent:
    """
    Base class for Azure-compatible agents.

    Usage:
        class MyAgent(AzureCompatibleAgent):
            def process_task(self, task_data):
                response = self.chat(
                    messages=[{"role": "user", "content": task_data["prompt"]}]
                )
                return response

        agent = MyAgent("gpt-4o", reasoning_effort="high")
        result = agent.process_task({"prompt": "Hello"})
    """

    def __init__(
        self,
        model_name: str,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        setup_azure_env()
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.client = create_azure_client(model_name)
        self.deployment = get_model_deployment(model_name)
        self.extra_params = kwargs

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Make a chat completion request."""
        params = {
            'reasoning_effort': self.reasoning_effort,
            'temperature': self.temperature,
            **self.extra_params,
            **kwargs,
        }
        return make_chat_completion(
            client=self.client,
            model_name=self.model_name,
            messages=messages,
            **params,
        )


# =============================================================================
# Test function
# =============================================================================
if __name__ == "__main__":
    print("Testing agent wrapper...")

    # Test environment setup
    setup_azure_env()
    print(f"TRAPI enabled: {is_trapi_enabled()}")
    print(f"Direct Azure enabled: {is_direct_azure_enabled()}")

    # Test client creation
    model = "gpt-4o"
    print(f"\nCreating client for {model}...")
    client = create_azure_client(model)
    print(f"Client type: {type(client).__name__}")

    # Test deployment resolution
    deployment = get_model_deployment(model)
    print(f"Deployment: {deployment}")

    # Test chat completion (if client is configured)
    try:
        response = make_chat_completion(
            client=client,
            model_name=model,
            messages=[{"role": "user", "content": "Say hello in one word"}],
            max_tokens=10,
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Chat test skipped: {e}")
