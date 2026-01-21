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
from typing import List, Dict, Any, Optional, Callable

from openai import AzureOpenAI

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
    """Token provider that uses MSAL to refresh tokens dynamically."""

    def __init__(self, scope: str = 'api://trapi/.default'):
        self.scope = scope
        self._cache = None
        self._app = None
        self._init_msal()

    def _init_msal(self):
        """Initialize MSAL app with the Azure CLI token cache."""
        if not MSAL_AVAILABLE:
            return

        cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
        if not os.path.exists(cache_path):
            return

        try:
            self._cache = msal.SerializableTokenCache()
            with open(cache_path, 'r') as f:
                self._cache.deserialize(f.read())

            self._app = msal.PublicClientApplication(
                AZURE_CLI_CLIENT_ID,
                authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
                token_cache=self._cache
            )
        except Exception as e:
            print(f"[MSALTokenProvider] Failed to initialize: {e}")

    def __call__(self) -> str:
        """Get a fresh access token."""
        if self._app is None:
            raise RuntimeError("MSAL not initialized - ensure ~/.azure is mounted")

        accounts = self._app.get_accounts()
        if not accounts:
            raise RuntimeError("No accounts found in MSAL cache")

        result = self._app.acquire_token_silent([self.scope], account=accounts[0])
        if result and 'access_token' in result:
            return result['access_token']

        raise RuntimeError(f"Failed to acquire token: {result.get('error_description', 'unknown error')}")

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


# TRAPI deployment name mapping
TRAPI_DEPLOYMENT_MAP = {
    # GPT-5 series
    'gpt-5': 'gpt-5_2025-08-07',
    'gpt-5-mini': 'gpt-5-mini_2025-08-07',
    'gpt-5-nano': 'gpt-5-nano_2025-08-07',
    'gpt-5-pro': 'gpt-5-pro_2025-10-06',

    # GPT-4 series
    'gpt-4o': 'gpt-4o_2024-11-20',
    'gpt-4o-mini': 'gpt-4o-mini_2024-07-18',
    'gpt-4.1': 'gpt-4.1_2025-04-14',
    'gpt-4.1-mini': 'gpt-4.1-mini_2025-04-14',
    'gpt-4.1-nano': 'gpt-4.1-nano_2025-04-14',
    'gpt-4-turbo': 'gpt-4_turbo-2024-04-09',
    'gpt-4-32k': 'gpt-4-32k_0613',
    'gpt-4': 'gpt-4_turbo-2024-04-09',

    # O-series (reasoning models)
    'o1': 'o1_2024-12-17',
    'o1-mini': 'o1-mini_2024-09-12',
    'o3': 'o3_2025-04-16',
    'o3-mini': 'o3-mini_2025-01-31',
    'o4-mini': 'o4-mini_2025-04-16',

    # GPT-5.1 series
    'gpt-5.1': 'gpt-5.1_2025-11-13',
    'gpt-5.1-chat': 'gpt-5.1-chat_2025-11-13',
    'gpt-5.1-codex': 'gpt-5.1-codex_2025-11-13',
    'gpt-5.1-codex-mini': 'gpt-5.1-codex-mini_2025-11-13',

    # Other models
    'grok-3.1': 'grok-3_1',
    'llama-3.3': 'gcr-llama-33-70b-shared',
    'llama-3.1-70b': 'gcr-llama-31-70b-shared',
    'llama-3.1-8b': 'gcr-llama-31-8b-instruct',
    'qwen3-8b': 'gcr-qwen3-8b',
    'phi4': 'gcr-phi-4-shared',
    'mistral': 'gcr-mistralai-8x7b-shared',
    'deepseek-r1': 'deepseek-r1_1',
    'deepseek': 'deepseek-r1_1',

    # Embeddings
    'text-embedding-3-large': 'text-embedding-3-large_1',
    'text-embedding-3-small': 'text-embedding-3-small_1',
    'text-embedding-ada-002': 'text-embedding-ada-002_2',
}


def resolve_deployment_name(model: str) -> str:
    """Resolve friendly model name to TRAPI deployment name."""
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
        processed_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            role = msg_copy.get('role', '')
            if role == 'tool-call':
                # Convert tool-call to assistant (the model's tool invocation)
                msg_copy['role'] = 'assistant'
            elif role == 'tool-response':
                # Convert tool-response to user (treat as user-provided information)
                # We can't use 'tool' role without proper tool_calls structure
                msg_copy['role'] = 'user'
            processed_messages.append(msg_copy)

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

        # Reasoning effort for O-series models
        if "reasoning_effort" in kwargs:
            request_params["reasoning_effort"] = kwargs["reasoning_effort"]
        elif "reasoning_effort" in self.kwargs:
            request_params["reasoning_effort"] = self.kwargs["reasoning_effort"]

        # Handle deepseek extra headers
        if 'deepseek' in self.model_id.lower():
            request_params["extra_headers"] = {"extra-parameters": "pass-through"}

        try:
            # Call Azure OpenAI with optional Weave tracing
            response = self._traced_completion_create(request_params)
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

    def _traced_completion_create(self, request_params: Dict[str, Any]):
        """
        Call Azure OpenAI API with optional Weave tracing.
        Falls back to direct call if Weave is not available.
        """
        try:
            import weave
            # Create a traced wrapper for the API call
            @weave.op(name=f"AzureOpenAI.chat.completions.create")
            def _create_with_weave(params):
                return self.client.chat.completions.create(**params)

            return _create_with_weave(request_params)
        except (ImportError, Exception):
            # Fall back to direct call without tracing
            return self.client.chat.completions.create(**request_params)

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Any]] = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Generate method - alias for __call__ for compatibility with smolagents.
        Some versions of smolagents call generate() instead of __call__().
        """
        return self.__call__(messages, stop_sequences, grammar, tools_to_call_from, **kwargs)

    def _supports_stop(self) -> bool:
        """Check if model supports stop parameter."""
        model_lower = self.model_id.lower()
        # O-series reasoning models (except o3-mini) don't support stop
        if model_lower.startswith("o3") and "o3-mini" not in model_lower:
            return False
        if "o4-mini" in model_lower:
            return False
        if model_lower.startswith("gpt-5"):
            return False
        return True

    def _supports_temperature(self) -> bool:
        """Check if model supports temperature parameter."""
        model_lower = self.model_id.lower()
        # O-series models don't support temperature
        if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            return False
        # GPT-5 models only support temperature=1 (default), so don't pass it
        if model_lower.startswith("gpt-5"):
            return False
        return True

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        model_lower = self.model_id.lower()
        # O-series reasoning models and GPT-5 use max_completion_tokens
        if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            return True
        # GPT-5 also uses max_completion_tokens
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
