"""
LiteLLM Engine with Azure/TRAPI Direct Authentication

Uses shared azure_utils module for consistent Azure authentication across all agents.
Falls back to LiteLLM if Azure auth is not available.
"""

import os
import sys
from pathlib import Path

# Add agents directory to path for shared module import
_agents_dir = Path(__file__).resolve().parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

# Try to import shared utilities
try:
    from shared.azure_utils import (
        get_trapi_client,
        resolve_deployment_name,
        is_direct_azure_enabled,
    )
    from shared.model_utils import (
        supports_temperature,
        supports_top_p,
        uses_max_completion_tokens,
    )
    SHARED_AVAILABLE = True
except ImportError:
    SHARED_AVAILABLE = False
    print("[LiteLlmEngine] Warning: shared module not available, will use fallback")

    # Fallback implementations
    def is_direct_azure_enabled():
        return os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true'

    # TRAPI deployment mapping fallback
    _FALLBACK_DEPLOYMENT_MAP = {
        'gpt-4o': 'gpt-4o_2024-11-20',
        'gpt-4.1': 'gpt-4.1_2025-04-14',
        'gpt-5': 'gpt-5_2025-08-07',
        'o3': 'o3_2025-04-16',
        'o3-mini': 'o3-mini_2025-01-31',
        'o4-mini': 'o4-mini_2025-04-16',
    }

    def resolve_deployment_name(model: str) -> str:
        model = model.replace('azure/', '').replace('openai/', '')
        model_lower = model.lower()
        # Direct lookup
        if model in _FALLBACK_DEPLOYMENT_MAP:
            return _FALLBACK_DEPLOYMENT_MAP[model]
        if model_lower in _FALLBACK_DEPLOYMENT_MAP:
            return _FALLBACK_DEPLOYMENT_MAP[model_lower]
        return model

    class MSALTokenProvider:
        """
        Token provider that reloads MSAL cache on EVERY call for automatic refresh.
        This is critical for long-running Docker containers where tokens expire after ~1 hour.
        """
        AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
        MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'

        def __init__(self, scope: str = 'api://trapi/.default'):
            self.scope = scope
            self.cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
            self._refresh_count = 0

        def __call__(self) -> str:
            """Get token, reloading cache from disk each time for fresh tokens."""
            import msal

            if not os.path.exists(self.cache_path):
                raise RuntimeError(f"MSAL cache not found at {self.cache_path}")

            # CRITICAL: Reload cache from disk on EVERY call
            # This picks up tokens refreshed by other processes or the host
            cache = msal.SerializableTokenCache()
            with open(self.cache_path, 'r') as f:
                cache.deserialize(f.read())

            app = msal.PublicClientApplication(
                self.AZURE_CLI_CLIENT_ID,
                authority=f'https://login.microsoftonline.com/{self.MICROSOFT_TENANT_ID}',
                token_cache=cache,
            )

            accounts = app.get_accounts()
            if not accounts:
                raise RuntimeError("No accounts found in MSAL cache. Run 'az login' first.")

            # Try ALL accounts - different accounts may have tokens for different scopes
            last_error = None
            for account in accounts:
                result = app.acquire_token_silent([self.scope], account=account)
                if result and 'access_token' in result:
                    # CRITICAL: Persist cache after token refresh
                    if cache.has_state_changed:
                        try:
                            with open(self.cache_path, 'w') as f:
                                f.write(cache.serialize())
                        except Exception as e:
                            print(f"[MSALTokenProvider] Warning: Could not persist cache: {e}")

                    self._refresh_count += 1
                    if self._refresh_count == 1 or self._refresh_count % 100 == 0:
                        print(f"[MSALTokenProvider] Token acquired (refresh #{self._refresh_count})")
                    return result['access_token']
                else:
                    last_error = result.get('error_description', 'Unknown error') if result else 'No token'

            raise RuntimeError(f"Token acquisition failed for all accounts. Last error: {last_error}")

    def get_trapi_client():
        """Fallback TRAPI client creation with proper token refresh."""
        from openai import AzureOpenAI

        DEFAULT_TRAPI_ENDPOINT = 'https://trapi.research.microsoft.com/gcr/shared'
        DEFAULT_TRAPI_API_VERSION = '2025-03-01-preview'
        DEFAULT_TRAPI_SCOPE = 'api://trapi/.default'

        endpoint = os.environ.get('TRAPI_ENDPOINT', DEFAULT_TRAPI_ENDPOINT)
        api_version = os.environ.get('TRAPI_API_VERSION', DEFAULT_TRAPI_API_VERSION)
        scope = os.environ.get('TRAPI_SCOPE', DEFAULT_TRAPI_SCOPE)
        max_retries = int(os.environ.get('TRAPI_MAX_RETRIES', 500))
        timeout = float(os.environ.get('TRAPI_TIMEOUT', 1800))

        # Method 1: MSAL token provider (REQUIRED - supports automatic refresh)
        # Critical for long-running benchmarks (3-4+ hours)
        try:
            import msal
            cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
            if os.path.exists(cache_path):
                token_provider = MSALTokenProvider(scope=scope)
                # Test it works
                token_provider()
                print(f"[LiteLlmEngine] Using MSAL token provider (auto-refresh enabled for long-running tasks)")
                return AzureOpenAI(
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=api_version,
                    max_retries=max_retries,
                    timeout=timeout,
                )
        except ImportError:
            print(f"[LiteLlmEngine] MSAL not available, trying Azure Identity")
        except Exception as e:
            print(f"[LiteLlmEngine] MSAL token provider failed: {e}")

        # Method 2: Azure Identity (fallback - also supports token refresh)
        try:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            credential = AzureCliCredential()
            token_provider = get_bearer_token_provider(credential, scope)
            print(f"[LiteLlmEngine] Using AzureCliCredential (auto-refresh enabled)")
            return AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
                max_retries=max_retries,
                timeout=timeout,
            )
        except ImportError:
            pass
        except Exception as e:
            print(f"[LiteLlmEngine] Azure Identity fallback failed: {e}")

        raise RuntimeError(
            "No Azure credentials available for long-running tasks. Options:\n"
            "1. Mount ~/.azure directory with MSAL cache (REQUIRED for tasks >1 hour)\n"
            "2. Install azure-identity and run 'az login'\n"
            "NOTE: Pre-fetched tokens are NOT supported as they expire after ~1 hour."
        )


def _normalize_model_id(model_id: str) -> str:
    """Normalize model ID by stripping provider prefixes."""
    model_lower = model_id.lower()
    for prefix in ('openai/', 'azure/', 'anthropic/'):
        if model_lower.startswith(prefix):
            return model_lower[len(prefix):]
    return model_lower


def _supports_temperature(model_id: str) -> bool:
    """Check if model supports temperature parameter."""
    if SHARED_AVAILABLE:
        return supports_temperature(model_id)
    model_lower = _normalize_model_id(model_id)
    # O-series and GPT-5 don't support temperature
    if any(model_lower.startswith(p) for p in ('o1', 'o3', 'o4', 'gpt-5')):
        return False
    return True


def _supports_top_p(model_id: str) -> bool:
    """Check if model supports top_p parameter."""
    if SHARED_AVAILABLE:
        return supports_top_p(model_id)
    # top_p follows same rules as temperature
    return _supports_temperature(model_id)


def _uses_max_completion_tokens(model_id: str) -> bool:
    """Check if model uses max_completion_tokens instead of max_tokens."""
    if SHARED_AVAILABLE:
        return uses_max_completion_tokens(model_id)
    model_lower = _normalize_model_id(model_id)
    # O-series and GPT-5 use max_completion_tokens
    if any(model_lower.startswith(p) for p in ('o1', 'o3', 'o4', 'gpt-5')):
        return True
    return False


class LiteLlmEngine:
    """
    LLM Engine that uses Azure/TRAPI direct authentication.
    Falls back to LiteLLM if Azure auth is not available.
    """

    def __init__(self, model_name, reasoning_effort=None):
        self.llm_engine_name = model_name
        self.reasoning_effort = reasoning_effort
        self.client = None
        self.deployment_name = None
        self._use_azure = False

        # Try to initialize Azure client
        if self._should_use_azure():
            self._init_azure_client()

    def _should_use_azure(self) -> bool:
        """Check if we should use Azure/TRAPI. Default: YES for OpenAI models."""
        # Explicit opt-out
        if os.environ.get('USE_TRAPI', '').lower() == 'false':
            print(f"[LiteLlmEngine] USE_TRAPI=false, skipping Azure")
            return False

        # Normalize model name - strip provider prefixes
        model_lower = self.llm_engine_name.lower()
        for prefix in ('openai/', 'azure/', 'anthropic/'):
            if model_lower.startswith(prefix):
                model_lower = model_lower[len(prefix):]
                break

        # Check if it's an OpenAI model - if so, use TRAPI by default
        is_openai_model = (
            'gpt-' in model_lower or
            model_lower.startswith('gpt-') or
            model_lower.startswith('o1') or
            model_lower.startswith('o3') or
            model_lower.startswith('o4') or
            'deepseek' in model_lower
        )
        if is_openai_model:
            print(f"[LiteLlmEngine] Using TRAPI for OpenAI model: {self.llm_engine_name}")
            return True

        print(f"[LiteLlmEngine] Model {self.llm_engine_name} is not an OpenAI model, using LiteLLM")
        return False

    def _init_azure_client(self):
        """Initialize Azure OpenAI client using shared utilities."""
        try:
            self.client = get_trapi_client()
            self.deployment_name = resolve_deployment_name(self.llm_engine_name)
            self._use_azure = True
            print(f"[LiteLlmEngine] Azure client initialized: {self.llm_engine_name} -> {self.deployment_name}")
        except Exception as e:
            print(f"[LiteLlmEngine] Failed to initialize Azure client: {e}")
            if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true':
                raise RuntimeError(
                    "Direct Azure is enabled but Azure client initialization failed. "
                    "Aborting to avoid LiteLLM fallback."
                ) from e
            print("[LiteLlmEngine] Falling back to LiteLLM")

    def respond(self, messages, temperature, top_p, max_tokens):
        """Generate a response from the model."""
        if self._use_azure and self.client:
            return self._respond_azure(messages, temperature, top_p, max_tokens)
        if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true':
            raise RuntimeError(
                "Direct Azure is enabled but Azure client was not initialized. "
                "Refusing to fall back to LiteLLM."
            )
        return self._respond_litellm(messages, temperature, top_p, max_tokens)

    def _respond_azure(self, messages, temperature, top_p, max_tokens):
        """Generate response using Azure OpenAI SDK."""
        request_params = {
            "model": self.deployment_name,
            "messages": messages,
        }

        # Max tokens handling
        if _uses_max_completion_tokens(self.llm_engine_name):
            request_params["max_completion_tokens"] = max_tokens
        else:
            request_params["max_tokens"] = max_tokens

        # Temperature (only if supported)
        if _supports_temperature(self.llm_engine_name):
            request_params["temperature"] = temperature
            request_params["top_p"] = top_p

        # Reasoning effort for O-series
        if self.reasoning_effort is not None:
            request_params["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**request_params)

        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return content, prompt_tokens, completion_tokens

    def _respond_litellm(self, messages, temperature, top_p, max_tokens):
        """Fallback to LiteLLM."""
        import litellm
        from litellm import completion

        # Enable drop_params to handle unsupported parameters gracefully
        litellm.drop_params = True

        # Build request params with model-specific filtering
        request_params = {
            "model": self.llm_engine_name,
            "messages": messages,
            "num_retries": 10,
        }

        # Use max_completion_tokens for O-series/GPT-5, max_tokens for others
        if _uses_max_completion_tokens(self.llm_engine_name):
            request_params["max_completion_tokens"] = max_tokens
        else:
            request_params["max_tokens"] = max_tokens

        # Add reasoning_effort if specified (for O-series/GPT-5)
        if self.reasoning_effort is not None:
            request_params["reasoning_effort"] = self.reasoning_effort
        else:
            # Only add temperature and top_p if model supports them
            if _supports_temperature(self.llm_engine_name):
                request_params["temperature"] = temperature
            if _supports_top_p(self.llm_engine_name):
                request_params["top_p"] = top_p

        responses = completion(**request_params)

        return responses.choices[0].message.content, responses.usage.prompt_tokens, responses.usage.completion_tokens
