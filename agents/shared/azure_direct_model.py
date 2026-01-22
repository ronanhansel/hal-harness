"""
Direct Azure OpenAI Model for smolagents - with automatic token refresh
Bypasses litellm entirely for maximum stability and speed

Usage:
    from azure_direct_model import AzureDirectModel

    model = AzureDirectModel(model_id='gpt-4o')
    # Use with smolagents CodeAgent
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from openai import AzureOpenAI

# Add parent directory to path for model_quirks import
_agents_dir = Path(__file__).resolve().parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

# Import shared model quirks module
try:
    from model_quirks import (
        ModelQuirks,
        supports_stop_parameter,
        supports_temperature,
        uses_max_completion_tokens,
        requires_extra_headers,
        strip_thinking_tags,
        # TRAPI deployment functions
        is_available_on_trapi,
        get_trapi_deployment,
        validate_trapi_model,
        TRAPI_DEPLOYMENT_MAP,
    )
    MODEL_QUIRKS_AVAILABLE = True
except ImportError:
    MODEL_QUIRKS_AVAILABLE = False
    TRAPI_DEPLOYMENT_MAP = {}  # Fallback empty map

# Import smolagents Tool for type annotations
try:
    from smolagents import Tool
    from smolagents.utils import get_tool_json_schema
except ImportError:
    # Will be imported later if available
    Tool = Any
    get_tool_json_schema = None

# Try to import azure-identity (may not be available in all environments)
try:
    from azure.identity import (
        ChainedTokenCredential,
        AzureCliCredential,
        ManagedIdentityCredential,
        SharedTokenCacheCredential,
        get_bearer_token_provider
    )
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

# Try to import msal for direct token refresh (works without az CLI)
try:
    import msal
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False


# Azure CLI's public client ID (used for MSAL token refresh)
AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'


def get_msal_token(scope: str = 'api://trapi/.default') -> Optional[str]:
    """
    Get an access token using MSAL by refreshing from the Azure CLI token cache.
    This works in containers without az CLI installed, as long as ~/.azure is mounted.
    """
    if not MSAL_AVAILABLE:
        return None

    cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
    if not os.path.exists(cache_path):
        return None

    try:
        # Load the MSAL token cache
        cache = msal.SerializableTokenCache()
        with open(cache_path, 'r') as f:
            cache.deserialize(f.read())

        # Create public client app with the cache
        app = msal.PublicClientApplication(
            AZURE_CLI_CLIENT_ID,
            authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
            token_cache=cache
        )

        # Get accounts and try to acquire token silently
        accounts = app.get_accounts()
        if accounts:
            result = app.acquire_token_silent([scope], account=accounts[0])
            if result and 'access_token' in result:
                return result['access_token']
    except Exception as e:
        print(f"[AzureDirectModel] MSAL token refresh failed: {e}")

    return None


class MSALTokenProvider:
    """
    Token provider that uses MSAL to refresh tokens dynamically.

    This provider:
    1. Reloads the cache file before each token acquisition (handles external updates)
    2. Persists the cache after acquiring new tokens (keeps refresh tokens fresh)
    3. Handles token refresh failures gracefully with detailed logging
    """

    def __init__(self, scope: str = 'api://trapi/.default'):
        self.scope = scope
        self.cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
        self._last_token_time = None
        self._token_refresh_count = 0

    def _load_cache(self) -> Optional[Any]:
        """Load the MSAL cache from disk."""
        if not MSAL_AVAILABLE:
            return None
        if not os.path.exists(self.cache_path):
            return None
        try:
            cache = msal.SerializableTokenCache()
            with open(self.cache_path, 'r') as f:
                cache.deserialize(f.read())
            return cache
        except Exception as e:
            print(f"[MSALTokenProvider] Failed to load cache: {e}")
            return None

    def _save_cache(self, cache) -> None:
        """Save the MSAL cache to disk if it has changed."""
        if cache.has_state_changed:
            try:
                with open(self.cache_path, 'w') as f:
                    f.write(cache.serialize())
                print(f"[MSALTokenProvider] Cache persisted to disk")
            except Exception as e:
                print(f"[MSALTokenProvider] Failed to save cache: {e}")

    def __call__(self) -> str:
        """
        Get a fresh access token, refreshing if necessary.

        The MSAL library handles token refresh automatically when calling
        acquire_token_silent. If the access token is expired but refresh
        token is valid, MSAL will get a new access token.
        """
        import time

        if not MSAL_AVAILABLE:
            raise RuntimeError("MSAL not available - install msal package")

        # Reload cache from disk each time (handles external updates)
        cache = self._load_cache()
        if cache is None:
            raise RuntimeError(f"MSAL cache not found at {self.cache_path}")

        # Create app with fresh cache
        app = msal.PublicClientApplication(
            AZURE_CLI_CLIENT_ID,
            authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
            token_cache=cache,
        )

        accounts = app.get_accounts()
        if not accounts:
            raise RuntimeError("No accounts found in MSAL cache. Run 'az login' first.")

        # Try to acquire token silently (uses refresh token if access token expired)
        result = app.acquire_token_silent([self.scope], account=accounts[0])

        if result and 'access_token' in result:
            # Save cache to persist any refreshed tokens
            self._save_cache(cache)

            self._token_refresh_count += 1
            self._last_token_time = time.time()

            # Log occasional refresh stats
            if self._token_refresh_count % 100 == 0:
                print(f"[MSALTokenProvider] Token refresh count: {self._token_refresh_count}")

            return result['access_token']

        # Token acquisition failed
        error = result.get('error', 'unknown') if result else 'no result'
        error_desc = result.get('error_description', '') if result else ''
        raise RuntimeError(f"Token acquisition failed: {error} - {error_desc}")

# Import smolagents Model base class
try:
    from smolagents.models import Model, MessageRole, ChatMessage
except ImportError:
    # Fallback for standalone testing
    class Model:
        pass
    class MessageRole:
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
    class ChatMessage:
        def __init__(self, role: str, content: str = None, tool_calls=None, raw=None):
            self.role = role
            self.content = content
            self.tool_calls = tool_calls
            self.raw = raw


# TRAPI deployment name mapping - imported from model_quirks
# The TRAPI_DEPLOYMENT_MAP is now centralized in model_quirks.py
# This ensures consistent behavior across all agents


def resolve_deployment_name(model: str) -> str:
    """Resolve friendly model name to TRAPI deployment name.

    Uses the shared model_quirks module for consistency.
    Will raise ValueError for models not available on TRAPI (e.g., DeepSeek-V3).
    """
    if MODEL_QUIRKS_AVAILABLE:
        # Use the shared function which properly validates model availability
        try:
            return get_trapi_deployment(model, strict=True)
        except ValueError as e:
            # Re-raise with clear error message
            raise ValueError(f"Cannot use model '{model}' with TRAPI: {e}")

    # Fallback for when model_quirks is not available
    # Remove common prefixes
    model = model.replace('azure/', '').replace('openai/', '')

    # Try exact match first
    if model in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[model]

    # Try lowercase match
    model_lower = model.lower()
    if model_lower in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[model_lower]

    # Try to find partial match
    for key, value in TRAPI_DEPLOYMENT_MAP.items():
        if key in model_lower or model_lower in key:
            return value

    # Return as-is if no mapping found (might be actual deployment name)
    return model


def create_trapi_client(
    endpoint: str = None,
    api_version: str = None,
    max_retries: int = None,
    timeout: float = None,
) -> AzureOpenAI:
    """Create AzureOpenAI client for TRAPI with auto token refresh."""
    endpoint = endpoint or os.environ.get('TRAPI_ENDPOINT', 'https://trapi.research.microsoft.com/gcr/shared')
    api_version = api_version or os.environ.get('TRAPI_API_VERSION', '2024-12-01-preview')
    scope = os.environ.get('TRAPI_SCOPE', 'api://trapi/.default')

    # Default retry/timeout settings for maximum resilience against rate limits
    if max_retries is None:
        max_retries = int(os.environ.get('TRAPI_MAX_RETRIES', 500))
    if timeout is None:
        timeout = float(os.environ.get('TRAPI_TIMEOUT', 1800))

    # Method 1: Try MSAL token provider (works in containers without az CLI)
    # This dynamically refreshes tokens using the mounted ~/.azure cache
    if MSAL_AVAILABLE:
        cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
        if os.path.exists(cache_path):
            try:
                token_provider = MSALTokenProvider(scope=scope)
                # Test that it works
                test_token = token_provider()
                if test_token:
                    print(f"[AzureDirectModel] Using MSAL token provider (dynamic refresh)")
                    return AzureOpenAI(
                        azure_endpoint=endpoint,
                        azure_ad_token_provider=token_provider,
                        api_version=api_version,
                        max_retries=max_retries,
                        timeout=timeout,
                    )
            except Exception as e:
                print(f"[AzureDirectModel] MSAL token provider failed: {e}")

    # Method 2: Use pre-fetched token if available (fallback for containers)
    prefetched_token = os.environ.get('AZURE_OPENAI_AD_TOKEN')
    if prefetched_token:
        print(f"[AzureDirectModel] Using pre-fetched Azure AD token (length: {len(prefetched_token)})")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token=prefetched_token,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    # Method 3: Use azure-identity credential chain (requires az CLI or managed identity)
    if AZURE_IDENTITY_AVAILABLE:
        print("[AzureDirectModel] Using azure-identity credential chain")
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        token_provider = get_bearer_token_provider(credential, scope)
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    raise RuntimeError("No authentication method available. Ensure ~/.azure is mounted or AZURE_OPENAI_AD_TOKEN is set.")


class AzureDirectModel(Model):
    """
    Direct Azure OpenAI Model for smolagents.
    Uses AzureOpenAI SDK with automatic token refresh.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 1800,  # 30 minutes for resilience
        num_retries: int = 500,  # High retry count for rate limit resilience
        endpoint: str = None,
        api_version: str = None,
        **kwargs,
    ):
        """
        Initialize the Azure Direct model.

        Args:
            model_id: Model name (e.g., 'gpt-4o', 'o3-mini', 'openai/gpt-5')
            temperature: Sampling temperature (ignored for reasoning models)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            num_retries: Number of retries on failure
            endpoint: TRAPI endpoint (default from env or gcr/shared)
            api_version: API version (default from env)
        """
        super().__init__()

        # Clean model ID
        self.model_id = model_id.replace('azure/', '').replace('openai/', '')
        self.deployment_name = resolve_deployment_name(self.model_id)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.num_retries = num_retries
        self.kwargs = kwargs

        # Create client with auto token refresh
        self.client = create_trapi_client(
            endpoint=endpoint,
            api_version=api_version,
            max_retries=num_retries,
            timeout=timeout,
        )

        print(f"[AzureDirectModel] Initialized: {self.model_id} -> {self.deployment_name}")

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Generate a response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            stop_sequences: Optional stop sequences
            grammar: Optional grammar constraint (not supported)
            tools_to_call_from: Optional list of Tool objects for function calling

        Returns:
            ChatMessage with the model's response
        """
        # Preprocess messages to fix role names for Azure OpenAI compatibility
        # smolagents uses 'tool-call' and 'tool-response' which need to be converted
        # Note: We can't use OpenAI's native 'tool' role because it requires
        # a preceding assistant message with 'tool_calls' structure, which
        # smolagents doesn't provide. Instead, convert to user messages.
        #
        # smolagents 1.24.0 passes ChatMessage objects, not dicts!
        # We need to convert them to dicts using .dict() method.
        processed_messages = []
        for msg in messages:
            # Handle both ChatMessage objects (smolagents 1.24+) and plain dicts
            if hasattr(msg, 'dict') and callable(msg.dict):
                # smolagents ChatMessage dataclass has .dict() method
                msg_dict = msg.dict()
            elif hasattr(msg, '__dict__'):
                # Fallback for other dataclass-like objects
                msg_dict = dict(msg.__dict__)
            elif isinstance(msg, dict):
                msg_dict = dict(msg)
            else:
                # Last resort: try to convert
                msg_dict = {'role': str(getattr(msg, 'role', 'user')), 'content': str(getattr(msg, 'content', ''))}

            # Get role - might be MessageRole enum or string
            role = msg_dict.get('role', '')
            if hasattr(role, 'value'):
                # MessageRole enum - get the string value
                role = role.value
            role = str(role)

            if role == 'tool-call':
                # Convert tool-call to assistant (the model's tool invocation)
                msg_dict['role'] = 'assistant'
            elif role == 'tool-response':
                # Convert tool-response to user (treat as user-provided information)
                # We can't use 'tool' role without proper tool_calls structure
                msg_dict['role'] = 'user'
            else:
                msg_dict['role'] = role

            # Ensure content is a string or proper format for Azure API
            content = msg_dict.get('content')
            if content is not None and not isinstance(content, (str, list)):
                msg_dict['content'] = str(content)

            processed_messages.append(msg_dict)

        # Build request parameters
        request_params = {
            "model": self.deployment_name,
            "messages": processed_messages,
        }

        # Use max_completion_tokens for reasoning models, max_tokens for others
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        if self._uses_max_completion_tokens():
            request_params["max_completion_tokens"] = max_tok
        else:
            request_params["max_tokens"] = max_tok

        # Temperature (not supported by all reasoning models)
        if self._supports_temperature():
            request_params["temperature"] = kwargs.get("temperature", self.temperature)

        # Stop sequences (not supported by some reasoning models)
        if stop_sequences and self._supports_stop():
            request_params["stop"] = stop_sequences

        # Tools parameter for function calling
        if tools_to_call_from:
            # Import here to avoid circular dependency
            if get_tool_json_schema is None:
                from smolagents.utils import get_tool_json_schema as _get_schema
                tools_json = [_get_schema(tool) for tool in tools_to_call_from]
            else:
                tools_json = [get_tool_json_schema(tool) for tool in tools_to_call_from]

            request_params.update({
                "tools": tools_json,
                "tool_choice": "required",
            })

        # Reasoning effort for O-series and GPT-5 models only
        # Only add if value is not None AND model supports it
        reasoning_effort = kwargs.get("reasoning_effort") or self.kwargs.get("reasoning_effort")
        if reasoning_effort is not None:
            # Check if model supports reasoning_effort
            model_lower = self.model_id.lower()
            is_reasoning_model = any(model_lower.startswith(p) for p in ['o1', 'o3', 'o4']) or 'gpt-5' in model_lower
            if is_reasoning_model:
                request_params["reasoning_effort"] = reasoning_effort

        # Handle deepseek extra headers
        if 'deepseek' in self.model_id.lower():
            request_params["extra_headers"] = {"extra-parameters": "pass-through"}

        try:
            # Call Azure OpenAI - Weave should autopatch this if enabled
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content or ""

            # Strip thinking tags for deepseek
            if 'deepseek' in self.model_id.lower():
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # Track token counts (required by smolagents Model.get_token_counts())
            if hasattr(response, 'usage') and response.usage:
                self.last_input_token_count = response.usage.prompt_tokens
                self.last_output_token_count = response.usage.completion_tokens
            else:
                self.last_input_token_count = None
                self.last_output_token_count = None

            return ChatMessage(role="assistant", content=content, raw=response)
        except Exception as e:
            print(f"[AzureDirectModel] Error calling {self.deployment_name}: {type(e).__name__}: {e}")
            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Generate method - required by smolagents 1.24.0+.
        In smolagents 1.24.0, __call__ delegates to generate().

        Args:
            messages: List of message dicts or ChatMessage objects
            stop_sequences: Optional stop sequences
            response_format: Optional response format (replaces 'grammar' in 1.24.0)
            tools_to_call_from: Optional list of Tool objects for function calling
        """
        # response_format is the new name for grammar in smolagents 1.24.0
        # Pass it through kwargs if provided
        if response_format is not None:
            kwargs['response_format'] = response_format
        return self.__call__(messages, stop_sequences, None, tools_to_call_from, **kwargs)

    def _supports_stop(self) -> bool:
        """Check if model supports stop parameter."""
        if MODEL_QUIRKS_AVAILABLE:
            return supports_stop_parameter(self.model_id)
        # Fallback if model_quirks not available
        model_lower = self.model_id.lower()
        if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            return False
        if model_lower.startswith("gpt-5"):
            return False
        return True

    def _supports_temperature(self) -> bool:
        """Check if model supports temperature parameter."""
        if MODEL_QUIRKS_AVAILABLE:
            return supports_temperature(self.model_id)
        # Fallback if model_quirks not available
        model_lower = self.model_id.lower()
        if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            return False
        if model_lower.startswith("gpt-5"):
            return False
        return True

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        if MODEL_QUIRKS_AVAILABLE:
            return uses_max_completion_tokens(self.model_id)
        # Fallback if model_quirks not available
        model_lower = self.model_id.lower()
        if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            return True
        if model_lower.startswith("gpt-5"):
            return True
        return False

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4


# Factory function for easy creation
def create_model(model_name: str, **kwargs) -> AzureDirectModel:
    """
    Create an AzureDirectModel instance.

    Args:
        model_name: Model name (e.g., 'openai/gpt-4o', 'azure/o3-mini')
        **kwargs: Additional arguments

    Returns:
        AzureDirectModel instance
    """
    return AzureDirectModel(model_id=model_name, **kwargs)


# Test
if __name__ == "__main__":
    print("Testing AzureDirectModel...")

    model = AzureDirectModel(model_id="gpt-4o", temperature=0.5)

    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]

    response = model(messages)
    print(f"Response: {response}")

    print("\nTesting deployment name resolution:")
    test_names = ["gpt-4o", "openai/gpt-4.1-2025-04-14", "azure/o3-mini", "o4-mini"]
    for name in test_names:
        print(f"  {name} -> {resolve_deployment_name(name)}")
