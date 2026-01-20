"""
Direct Azure OpenAI / TRAPI Client for HAL
Bypasses LiteLLM proxy for better stability

Usage:
    from azure_client import get_trapi_client, get_azure_client, TRAPI_DEPLOYMENT_MAP

    client = get_trapi_client()
    response = client.chat.completions.create(
        model=TRAPI_DEPLOYMENT_MAP.get('gpt-4o', 'gpt-4o_2024-11-20'),
        messages=[{"role": "user", "content": "Hello"}]
    )
"""

import os
from openai import AzureOpenAI

# Import azure.identity only if available (not in minimal Docker containers)
try:
    from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

# TRAPI deployment name mapping
# Maps friendly names to actual Azure deployment names
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

    # Embeddings
    'text-embedding-3-large': 'text-embedding-3-large_1',
    'text-embedding-3-small': 'text-embedding-3-small_1',
    'text-embedding-ada-002': 'text-embedding-ada-002_2',
}

# Reverse mapping for looking up friendly name from deployment name
DEPLOYMENT_TO_FRIENDLY = {v: k for k, v in TRAPI_DEPLOYMENT_MAP.items()}


def get_trapi_client(
    endpoint: str = None,
    api_version: str = None,
    max_retries: int = None,
    timeout: float = None,
) -> AzureOpenAI:
    """
    Create an AzureOpenAI client for TRAPI endpoints.
    Uses Azure CLI credentials (az login), or pre-fetched token from HAL Docker runner.

    Args:
        endpoint: TRAPI endpoint URL (default from env or gcr/shared)
        api_version: API version (default from env or 2024-12-01-preview)
        max_retries: Number of retries for failed requests (default: 35 for ~5 min retry window)
        timeout: Request timeout in seconds (default: 600 = 10 minutes)

    Returns:
        AzureOpenAI client configured for TRAPI
    """
    endpoint = endpoint or os.environ.get('TRAPI_ENDPOINT', 'https://trapi.research.microsoft.com/gcr/shared')
    api_version = api_version or os.environ.get('TRAPI_API_VERSION', '2024-12-01-preview')

    # Default retry/timeout settings for maximum resilience against rate limits
    # 500 retries with exponential backoff = very long retry window for rate limit recovery
    if max_retries is None:
        max_retries = int(os.environ.get('TRAPI_MAX_RETRIES', 500))
    if timeout is None:
        timeout = float(os.environ.get('TRAPI_TIMEOUT', 1800))  # 30 minutes per request

    # Check for pre-fetched token from HAL Docker runner
    prefetched_token = os.environ.get('AZURE_OPENAI_AD_TOKEN')
    if prefetched_token:
        print(f"[azure_client] Using pre-fetched Azure AD token (length: {len(prefetched_token)})")
        print(f"[azure_client] Retry config: max_retries={max_retries}, timeout={timeout}s")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token=prefetched_token,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    # Fall back to credential chain if azure.identity is available
    if AZURE_IDENTITY_AVAILABLE:
        scope = os.environ.get('TRAPI_SCOPE', 'api://trapi/.default')
        # Create credential chain - try Azure CLI first, then Managed Identity
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        token_provider = get_bearer_token_provider(credential, scope)

        print(f"[azure_client] Retry config: max_retries={max_retries}, timeout={timeout}s")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    # No credentials available - raise an error
    raise RuntimeError(
        "No Azure credentials available. Either set AZURE_OPENAI_AD_TOKEN "
        "(pre-fetched by HAL Docker runner), or install azure-identity and run 'az login'."
    )


def get_azure_client(
    endpoint: str = None,
    api_version: str = None,
    max_retries: int = None,
    timeout: float = None,
) -> AzureOpenAI:
    """
    Create an AzureOpenAI client for Azure Cognitive Services endpoints.
    Uses Azure CLI credentials (az login), or pre-fetched token from HAL Docker runner.

    Args:
        endpoint: Azure endpoint URL (default from env)
        api_version: API version (default from env or 2024-10-21)
        max_retries: Number of retries for failed requests (default: 35)
        timeout: Request timeout in seconds (default: 600)

    Returns:
        AzureOpenAI client configured for Azure Cognitive Services
    """
    endpoint = endpoint or os.environ.get('AZURE_ENDPOINT', 'https://msrasc-swe.cognitiveservices.azure.com/')
    api_version = api_version or os.environ.get('AZURE_API_VERSION', '2024-10-21')

    # Default retry/timeout settings for maximum resilience against rate limits
    if max_retries is None:
        max_retries = int(os.environ.get('AZURE_MAX_RETRIES', 500))
    if timeout is None:
        timeout = float(os.environ.get('AZURE_TIMEOUT', 1800))  # 30 minutes per request

    # Check for pre-fetched token from HAL Docker runner
    prefetched_token = os.environ.get('AZURE_OPENAI_AD_TOKEN')
    if prefetched_token:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token=prefetched_token,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    # Fall back to credential chain if azure.identity is available
    if AZURE_IDENTITY_AVAILABLE:
        scope = os.environ.get('AZURE_SCOPE', 'https://cognitiveservices.azure.com/.default')
        credential = AzureCliCredential()
        token_provider = get_bearer_token_provider(credential, scope)

        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    raise RuntimeError(
        "No Azure credentials available. Either set AZURE_OPENAI_AD_TOKEN "
        "(pre-fetched by HAL Docker runner), or install azure-identity and run 'az login'."
    )


def resolve_deployment_name(model: str) -> str:
    """
    Resolve a friendly model name to its TRAPI deployment name.

    Args:
        model: Model name (e.g., 'gpt-4o', 'o3-mini')

    Returns:
        Deployment name (e.g., 'gpt-4o_2024-11-20', 'o3-mini_2025-01-31')
    """
    # Remove 'azure/' prefix if present
    if model.startswith('azure/'):
        model = model[6:]

    # Check mapping
    return TRAPI_DEPLOYMENT_MAP.get(model.lower(), model)


# Test function
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
