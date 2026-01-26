"""
Direct Azure OpenAI / TRAPI Client for HAL Agents

This module provides a unified Azure client that can be used by all HAL agents.
It bypasses LiteLLM proxy for better stability and direct TRAPI access.

Usage:
    from shared.azure_utils import get_trapi_client, resolve_deployment_name, TRAPI_DEPLOYMENT_MAP

    client = get_trapi_client()
    deployment = resolve_deployment_name('gpt-4o')
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Hello"}]
    )

Environment Variables:
    USE_DIRECT_AZURE: Set to "true" to enable direct Azure mode (auto-enabled if MSAL cache exists)
    USE_TRAPI: Set to "true" to use TRAPI (default: true)
    TRAPI_ENDPOINT: TRAPI endpoint URL
    TRAPI_API_VERSION: API version for TRAPI
    TRAPI_SCOPE: OAuth scope for TRAPI
    TRAPI_MAX_RETRIES: Max retries for failed requests
    TRAPI_TIMEOUT: Request timeout in seconds

Authentication:
    Uses MSAL token provider for automatic token refresh (required for long-running benchmarks).
    Pre-fetched tokens are NOT supported as they expire after ~1 hour.
    Mount ~/.azure directory with MSAL cache for Docker containers.
"""

import os
from typing import Optional, Callable

from openai import AzureOpenAI

# Import azure.identity only if available (not in minimal Docker containers)
try:
    from azure.identity import (
        ChainedTokenCredential,
        AzureCliCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

# Import MSAL for token refresh in containers
try:
    import msal
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False


# =============================================================================
# TRAPI Deployment Name Mapping
# =============================================================================
# Maps friendly model names to actual Azure/TRAPI deployment names.
# This is the single source of truth for all agents.

TRAPI_DEPLOYMENT_MAP = {
    # =========================================================================
    # GPT-5 series
    # =========================================================================
    'gpt-5': 'gpt-5_2025-08-07',
    'gpt-5_2025-08-07': 'gpt-5_2025-08-07',
    'gpt-5-2025-08-07': 'gpt-5_2025-08-07',
    'gpt-5-mini': 'gpt-5-mini_2025-08-07',
    'gpt-5-mini_2025-08-07': 'gpt-5-mini_2025-08-07',
    'gpt-5-mini-2025-08-07': 'gpt-5-mini_2025-08-07',
    'gpt-5-nano': 'gpt-5-nano_2025-08-07',
    'gpt-5-pro': 'gpt-5-pro_2025-10-06',

    # GPT-5.1 series
    'gpt-5.1': 'gpt-5.1_2025-11-13',
    'gpt-5.1-chat': 'gpt-5.1-chat_2025-11-13',
    'gpt-5.1-codex': 'gpt-5.1-codex_2025-11-13',
    'gpt-5.1-codex-mini': 'gpt-5.1-codex-mini_2025-11-13',

    # GPT-5.2 series
    'gpt-5.2': 'gpt-5.2_2025-12-11',
    'gpt-5.2-chat': 'gpt-5.2-chat_2025-12-11',

    # =========================================================================
    # GPT-4 series
    # =========================================================================
    'gpt-4o': 'gpt-4o_2024-11-20',
    'gpt-4o_2024-11-20': 'gpt-4o_2024-11-20',
    'gpt-4o-2024-11-20': 'gpt-4o_2024-11-20',
    'gpt-4o-mini': 'gpt-4o-mini_2024-07-18',
    'gpt-4o-mini_2024-07-18': 'gpt-4o-mini_2024-07-18',
    'gpt-4.1': 'gpt-4.1_2025-04-14',
    'gpt-4.1_2025-04-14': 'gpt-4.1_2025-04-14',
    'gpt-4.1-2025-04-14': 'gpt-4.1_2025-04-14',
    'gpt-4.1-mini': 'gpt-4.1-mini_2025-04-14',
    'gpt-4.1-mini_2025-04-14': 'gpt-4.1-mini_2025-04-14',
    'gpt-4.1-nano': 'gpt-4.1-nano_2025-04-14',
    'gpt-4.1-nano_2025-04-14': 'gpt-4.1-nano_2025-04-14',
    'gpt-4-turbo': 'gpt-4_turbo-2024-04-09',
    'gpt-4-32k': 'gpt-4-32k_0613',

    # =========================================================================
    # O-series (reasoning models)
    # =========================================================================
    'o1': 'o1_2024-12-17',
    'o1_2024-12-17': 'o1_2024-12-17',
    'o1-2024-12-17': 'o1_2024-12-17',
    'o1-mini': 'o1-mini_2024-09-12',
    'o1-mini_2024-09-12': 'o1-mini_2024-09-12',
    'o3': 'o3_2025-04-16',
    'o3_2025-04-16': 'o3_2025-04-16',
    'o3-2025-04-16': 'o3_2025-04-16',
    'o3-mini': 'o3-mini_2025-01-31',
    'o3-mini_2025-01-31': 'o3-mini_2025-01-31',
    'o3-mini-2025-01-31': 'o3-mini_2025-01-31',
    'o4-mini': 'o4-mini_2025-04-16',
    'o4-mini_2025-04-16': 'o4-mini_2025-04-16',
    'o4-mini-2025-04-16': 'o4-mini_2025-04-16',

    # =========================================================================
    # DeepSeek models
    # =========================================================================
    'deepseek-r1': 'DeepSeek-R1_1',
    'deepseek-r1_1': 'DeepSeek-R1_1',
    'DeepSeek-R1_1': 'DeepSeek-R1_1',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3-0324',
    'deepseek-v3-0324': 'deepseek-ai/DeepSeek-V3-0324',
    'deepseek-ai/deepseek-v3-0324': 'deepseek-ai/DeepSeek-V3-0324',
    'deepseek-ai/DeepSeek-V3-0324': 'deepseek-ai/DeepSeek-V3-0324',

    # =========================================================================
    # Open source models (via TRAPI)
    # =========================================================================
    'grok-3.1': 'grok-3_1',
    'llama-3.3': 'gcr-llama-33-70b-shared',
    'llama-3.3-70b': 'gcr-llama-33-70b-shared',
    'llama-3.1-70b': 'gcr-llama-31-70b-shared',
    'llama-3.1-8b': 'gcr-llama-31-8b-instruct',
    'qwen3-8b': 'gcr-qwen3-8b',
    'phi4': 'gcr-phi-4-shared',
    'mistral': 'gcr-mistralai-8x7b-shared',
    'mistral-8x7b': 'gcr-mistralai-8x7b-shared',

    # =========================================================================
    # Embeddings
    # =========================================================================
    'text-embedding-3-large': 'text-embedding-3-large_1',
    'text-embedding-3-small': 'text-embedding-3-small_1',
    'text-embedding-ada-002': 'text-embedding-ada-002_2',
}

# Reverse mapping for looking up friendly name from deployment name
DEPLOYMENT_TO_FRIENDLY = {v: k for k, v in TRAPI_DEPLOYMENT_MAP.items()}


# =============================================================================
# Azure Constants
# =============================================================================
AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'
DEFAULT_TRAPI_ENDPOINT = 'https://trapi.research.microsoft.com/gcr/shared'
DEFAULT_TRAPI_API_VERSION = '2025-03-01-preview'  # Required for GPT-5.2 and newer
DEFAULT_TRAPI_SCOPE = 'api://trapi/.default'
DEFAULT_AZURE_ENDPOINT = 'https://msrasc-swe.cognitiveservices.azure.com/'
DEFAULT_AZURE_API_VERSION = '2024-10-21'
DEFAULT_AZURE_SCOPE = 'https://cognitiveservices.azure.com/.default'


class MSALTokenProvider:
    """
    Token provider using MSAL cache with automatic refresh and persistence.

    This provider:
    1. Reloads the cache file before each token acquisition (handles external updates)
    2. Persists the cache after acquiring new tokens (keeps refresh tokens fresh)
    3. Handles token refresh failures gracefully with detailed logging
    4. Supports automatic retry for transient failures
    5. Uses shared locks for reading to support high parallelism (e.g. 400+ processes)
    """

    def __init__(self, scope: str = 'api://trapi/.default'):
        self.scope = scope
        self.cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
        self._last_token_time = None
        self._token_refresh_count = 0
        self._lock_held = False  # Track if we already hold an exclusive lock
        self.cache = msal.SerializableTokenCache()
        # Initialize app once (will rely on cache updates)
        self.app = msal.PublicClientApplication(
            AZURE_CLI_CLIENT_ID,
            authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
            token_cache=self.cache,
        )
        # Initial load attempt (best effort)
        self._load_cache()

    def _load_cache(self):
        """Load the MSAL cache from disk. Returns True if loaded."""
        if not MSAL_AVAILABLE:
            return False
        if not os.path.exists(self.cache_path):
            return False
        
        # If we already hold an exclusive lock from __call__, we can just read
        if self._lock_held:
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache.deserialize(f.read())
                return True
            except Exception as e:
                print(f"[MSALTokenProvider] Failed to read cache while locked: {e}")
                return False

        lock_path = self.cache_path + ".lock"
        try:
            import fcntl
            with open(lock_path, 'a+') as lock_file:
                # Use SHARED lock for reading to allow parallel processes to load cache simultaneously
                fcntl.flock(lock_file, fcntl.LOCK_SH)
                try:
                    with open(self.cache_path, 'r') as f:
                        self.cache.deserialize(f.read())
                    return True
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception as e:
            print(f"[MSALTokenProvider] Failed to load cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save the MSAL cache to disk if it has changed."""
        if self.cache.has_state_changed:
            # If we already hold an exclusive lock from __call__, we can just write
            if self._lock_held:
                try:
                    with open(self.cache_path, 'w') as f:
                        f.write(self.cache.serialize())
                    print(f"[MSALTokenProvider] Cache persisted to disk")
                    return
                except Exception as e:
                    print(f"[MSALTokenProvider] Failed to save cache while locked: {e}")
                    return

            lock_path = self.cache_path + ".lock"
            try:
                import fcntl
                with open(lock_path, 'a+') as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    try:
                        with open(self.cache_path, 'w') as f:
                            f.write(self.cache.serialize())
                        print(f"[MSALTokenProvider] Cache persisted to disk")
                    finally:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception as e:
                print(f"[MSALTokenProvider] Failed to save cache: {e}")

    def _find_valid_at_in_cache(self, accounts):
        """
        Manually check for a valid access token in the cache to avoid
        triggering a Refresh Token usage race condition.
        """
        import time
        # 5 minute buffer to be safe
        now = time.time() + 300
        
        for account in accounts:
            try:
                # This depends on MSAL internals structure which is stable enough
                for token in self.cache.find(msal.TokenCache.CredentialType.ACCESS_TOKEN, 
                                           query={"home_account_id": account["home_account_id"]}):
                    if self.scope in token.get("target", ""):
                        expires_on = int(token.get("expires_on", 0))
                        if expires_on > now:
                            return token.get("secret")
            except Exception:
                pass
        return None

    def __call__(self) -> str:
        """
        Get a fresh access token, refreshing if necessary.
        Uses Check-Lock-Check-Act pattern to minimize disk I/O and locking.
        """
        import time
        import fcntl

        accounts = self.app.get_accounts()
        if not accounts and os.path.exists(self.cache_path):
             # First run or empty memory: load from disk
             self._load_cache()
             accounts = self.app.get_accounts()

        if not accounts:
            # Fallback if still empty (maybe need login)
            if not os.path.exists(self.cache_path):
                 raise RuntimeError(f"MSAL cache not found at {self.cache_path}")
            # Try loading again inside lock in slow path
            pass

        # 1. Fast Path: Check in-memory cache for valid AT
        # We manually check expiry to avoid triggering RT usage race
        token = self._find_valid_at_in_cache(accounts)
        if token:
            return token

        print(f"[MSALTokenProvider] Fast path miss - waiting for lock...")
        start_wait = time.time()

        # 2. Slow Path: Lock, Reload, Check, Act
        lock_path = self.cache_path + ".lock"
        with open(lock_path, 'a+') as lock_file:
            # Wait for exclusive lock to perform refresh
            # We use a blocking flock, but we could use a loop with timeout if needed
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            self._lock_held = True
            
            wait_time = time.time() - start_wait
            if wait_time > 1.0:
                print(f"[MSALTokenProvider] Lock acquired (waited {wait_time:.2f}s)")
            
            try:
                # Reload cache (someone else might have refreshed)
                self._load_cache()
                
                # Update accounts list after reload
                accounts = self.app.get_accounts()
                if not accounts:
                    raise RuntimeError("No accounts found in MSAL cache. Run 'az login' first.")

                # Check again (did someone else refresh?)
                token = self._find_valid_at_in_cache(accounts)
                if token:
                    print(f"[MSALTokenProvider] Token found after reload")
                    return token

                # 3. Refresh Action (Authorized by Lock)
                # Now safe to use acquire_token_silent which might use RT
                last_error = None
                for account in accounts:
                    # msal's acquire_token_silent will refresh if AT is expired but RT is valid
                    result = self.app.acquire_token_silent([self.scope], account=account)
                    
                    if result and 'access_token' in result:
                        # If we refreshed, save to disk
                        self._save_cache()
                        
                        self._token_refresh_count += 1
                        if self._token_refresh_count == 1 or self._token_refresh_count % 100 == 0:
                            username = account.get('username', 'unknown')
                            print(f"[MSALTokenProvider] Token refreshed (count: {self._token_refresh_count}, account: {username})")
                        return result['access_token']
                    
                    if result:
                         last_error = result.get('error_description', 'unknown')

                raise RuntimeError(f"Token acquisition failed for all accounts. Last error: {last_error}")

            finally:
                self._lock_held = False
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _get_msal_token_provider(scope: str, skip_test: bool = False) -> Optional[Callable[[], str]]:
    """
    Create a token provider using MSAL cache.
    Works in Docker containers without requiring az CLI.

    Args:
        scope: The OAuth scope
        skip_test: If True, skips the initial "preflight" token acquisition test.
                  Useful for reducing overhead when launching many processes.
    """
    if not MSAL_AVAILABLE:
        print("[azure_utils] MSAL not available - install msal package")
        return None

    cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
    if not os.path.exists(cache_path):
        print(f"[azure_utils] MSAL cache not found at {cache_path}")
        return None

    try:
        # Create the token provider
        provider = MSALTokenProvider(scope=scope)

        if skip_test:
            print(f"[azure_utils] MSAL token provider initialized (lazy mode)")
            return provider

        # Test that it works (preflight)
        test_token = provider()
        if test_token:
            print(f"[azure_utils] MSAL token provider initialized successfully")
            return provider
        return None

    except Exception as e:
        print(f"[azure_utils] MSAL token provider failed: {e}")
        return None


def get_trapi_client(
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
) -> AzureOpenAI:
    """
    Create an AzureOpenAI client for TRAPI endpoints.
    Uses Azure CLI credentials (az login), MSAL cache, or pre-fetched token.

    Args:
        endpoint: TRAPI endpoint URL (default from env or gcr/shared)
        api_version: API version (default from env or 2025-03-01-preview for GPT-5.2)
        max_retries: Number of retries for failed requests (default: 500)
        timeout: Request timeout in seconds (default: 1800 = 30 minutes)

    Returns:
        AzureOpenAI client configured for TRAPI

    Raises:
        RuntimeError: If no Azure credentials are available
    """
    endpoint = endpoint or os.environ.get('TRAPI_ENDPOINT', DEFAULT_TRAPI_ENDPOINT)
    api_version = api_version or os.environ.get('TRAPI_API_VERSION', DEFAULT_TRAPI_API_VERSION)
    scope = os.environ.get('TRAPI_SCOPE', DEFAULT_TRAPI_SCOPE)

    # Default retry/timeout settings for maximum resilience against rate limits
    if max_retries is None:
        max_retries = int(os.environ.get('TRAPI_MAX_RETRIES', 500))
    if timeout is None:
        timeout = float(os.environ.get('TRAPI_TIMEOUT', 1800))

    # Method 1: MSAL token provider (REQUIRED for direct Azure mode)
    # MSAL can refresh tokens automatically using the refresh token in the cache
    # This is critical for long-running benchmarks (3-4+ hours)
    direct_required = os.environ.get("USE_DIRECT_AZURE", "").lower() == "true"
    skip_test = os.environ.get("HAL_SKIP_MSAL_PREFLIGHT", "").lower() == "true"
    msal_provider = _get_msal_token_provider(scope, skip_test=skip_test)
    if msal_provider:
        print(f"[azure_utils] Using MSAL token provider (auto-refresh enabled for long-running tasks)")
        print(f"[azure_utils] Retry config: max_retries={max_retries}, timeout={timeout}s")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=msal_provider,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )
    if direct_required:
        raise RuntimeError(
            "Direct Azure is enabled but MSAL cache is unavailable. "
            "Ensure ~/.azure/msal_token_cache.json is mounted into the container."
        )

    # Method 2: Azure Identity credential chain (fallback - also supports token refresh)
    if AZURE_IDENTITY_AVAILABLE:
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        token_provider = get_bearer_token_provider(credential, scope)

        print(f"[azure_utils] Using Azure Identity credential chain")
        print(f"[azure_utils] Retry config: max_retries={max_retries}, timeout={timeout}s")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    # No credentials available
    raise RuntimeError(
        "No Azure credentials available for long-running tasks. Options:\n"
        "1. Mount ~/.azure directory with MSAL cache (REQUIRED for tasks >1 hour)\n"
        "2. Install azure-identity and run 'az login'\n"
        "NOTE: Pre-fetched tokens are NOT supported as they expire after ~1 hour."
    )


def get_azure_client(
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
) -> AzureOpenAI:
    """
    Create an AzureOpenAI client for Azure Cognitive Services endpoints.
    Uses Azure CLI credentials (az login), MSAL cache, or pre-fetched token.

    Args:
        endpoint: Azure endpoint URL (default from env)
        api_version: API version (default from env or 2024-10-21)
        max_retries: Number of retries for failed requests (default: 500)
        timeout: Request timeout in seconds (default: 1800)

    Returns:
        AzureOpenAI client configured for Azure Cognitive Services

    Raises:
        RuntimeError: If no Azure credentials are available
    """
    endpoint = endpoint or os.environ.get('AZURE_ENDPOINT', DEFAULT_AZURE_ENDPOINT)
    api_version = api_version or os.environ.get('AZURE_API_VERSION', DEFAULT_AZURE_API_VERSION)
    scope = os.environ.get('AZURE_SCOPE', DEFAULT_AZURE_SCOPE)

    if max_retries is None:
        max_retries = int(os.environ.get('AZURE_MAX_RETRIES', 500))
    if timeout is None:
        timeout = float(os.environ.get('AZURE_TIMEOUT', 1800))

    # Method 1: MSAL token provider (REQUIRED for direct Azure mode)
    # Critical for long-running benchmarks (3-4+ hours)
    direct_required = os.environ.get("USE_DIRECT_AZURE", "").lower() == "true"
    skip_test = os.environ.get("HAL_SKIP_MSAL_PREFLIGHT", "").lower() == "true"
    msal_provider = _get_msal_token_provider(scope, skip_test=skip_test)
    if msal_provider:
        print(f"[azure_utils] Using MSAL token provider for Azure client (auto-refresh enabled)")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=msal_provider,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )
    if direct_required:
        raise RuntimeError(
            "Direct Azure is enabled but MSAL cache is unavailable. "
            "Ensure ~/.azure/msal_token_cache.json is mounted into the container."
        )

    # Method 2: Azure Identity (fallback - also supports token refresh)
    if AZURE_IDENTITY_AVAILABLE:
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, scope)
        print(f"[azure_utils] Using Azure Identity for Azure client (auto-refresh enabled)")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    raise RuntimeError(
        "No Azure credentials available for long-running tasks. Options:\n"
        "1. Mount ~/.azure directory with MSAL cache (REQUIRED for tasks >1 hour)\n"
        "2. Install azure-identity and run 'az login'\n"
        "NOTE: Pre-fetched tokens are NOT supported as they expire after ~1 hour."
    )


def resolve_deployment_name(model: str) -> str:
    """
    Resolve a friendly model name to its TRAPI deployment name.

    Args:
        model: Model name (e.g., 'gpt-4o', 'o3-mini', 'openai/gpt-4.1-2025-04-14')

    Returns:
        Deployment name (e.g., 'gpt-4o_2024-11-20', 'o3-mini_2025-01-31')

    Examples:
        >>> resolve_deployment_name('gpt-4o')
        'gpt-4o_2024-11-20'
        >>> resolve_deployment_name('openai/gpt-4.1-2025-04-14')
        'gpt-4.1_2025-04-14'
        >>> resolve_deployment_name('o3-mini')
        'o3-mini_2025-01-31'
    """
    # Remove common prefixes
    if model.startswith('azure/'):
        model = model[6:]
    if model.startswith('openai/'):
        model = model[7:]

    # Direct lookup (case-sensitive first)
    if model in TRAPI_DEPLOYMENT_MAP:
        return TRAPI_DEPLOYMENT_MAP[model]

    # Case-insensitive lookup
    model_lower = model.lower()
    for key, deployment in TRAPI_DEPLOYMENT_MAP.items():
        if key.lower() == model_lower:
            return deployment

    # Try partial match for models like "DeepSeek-R1_1" -> "deepseek-r1"
    for key, deployment in TRAPI_DEPLOYMENT_MAP.items():
        normalized_key = key.replace('-', '').replace('_', '').lower()
        normalized_model = model_lower.replace('-', '').replace('_', '')
        if normalized_key in normalized_model or normalized_model in normalized_key:
            return deployment

    # Fallback - return as-is (might be already a deployment name)
    return model


def is_trapi_enabled() -> bool:
    """Check if TRAPI mode is enabled."""
    return os.environ.get('USE_TRAPI', 'true').lower() == 'true'


def is_direct_azure_enabled() -> bool:
    """Check if direct Azure mode is enabled."""
    return os.environ.get('USE_DIRECT_AZURE', '').lower() == 'true'


def setup_direct_azure_env() -> None:
    """
    Remove proxy URLs from environment when using direct Azure.
    Call this early in your agent's main.py to ensure Azure config is correct.
    """
    if is_direct_azure_enabled():
        for key in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_BASE_URL', 'LITELLM_BASE_URL'):
            os.environ.pop(key, None)
        print("[azure_utils] Direct Azure mode: removed proxy URLs from environment")


# =============================================================================
# Test function
# =============================================================================
if __name__ == '__main__':
    print("Testing TRAPI client...")
    client = get_trapi_client()

    model = 'gpt-4o'
    deployment = resolve_deployment_name(model)
    print(f"Model: {model} -> Deployment: {deployment}")

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Say hello in one word"}],
        max_tokens=10
    )
    print(f"Response: {response.choices[0].message.content}")
