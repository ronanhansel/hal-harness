"""
Direct Azure OpenAI / TRAPI Client for HAL

DEPRECATED: This file re-exports the shared azure_utils module.
Please use the shared module directly:
    from shared.azure_utils import get_trapi_client, resolve_deployment_name, TRAPI_DEPLOYMENT_MAP

The shared module provides:
- MSALTokenProvider with automatic cache persistence for token refresh
- Centralized deployment name mapping
- Consistent behavior across all agents
"""

import sys
from pathlib import Path

# Add shared module to path
_agents_dir = Path(__file__).resolve().parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

# Import everything from shared module
from shared.azure_utils import (
    get_trapi_client,
    get_azure_client,
    resolve_deployment_name,
    TRAPI_DEPLOYMENT_MAP,
    DEPLOYMENT_TO_FRIENDLY,
    is_trapi_enabled,
    is_direct_azure_enabled,
    setup_direct_azure_env,
    MSALTokenProvider,
)

# Import constants for backwards compatibility
try:
    from shared.azure_utils import (
        AZURE_CLI_CLIENT_ID,
        MICROSOFT_TENANT_ID,
        DEFAULT_TRAPI_ENDPOINT,
        DEFAULT_TRAPI_API_VERSION,
        DEFAULT_TRAPI_SCOPE,
    )
except ImportError:
    AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
    MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'
    DEFAULT_TRAPI_ENDPOINT = 'https://trapi.research.microsoft.com/gcr/shared'
    DEFAULT_TRAPI_API_VERSION = '2025-03-01-preview'
    DEFAULT_TRAPI_SCOPE = 'api://trapi/.default'

# Check for azure.identity availability
try:
    from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

__all__ = [
    "get_trapi_client",
    "get_azure_client",
    "resolve_deployment_name",
    "TRAPI_DEPLOYMENT_MAP",
    "DEPLOYMENT_TO_FRIENDLY",
    "is_trapi_enabled",
    "is_direct_azure_enabled",
    "setup_direct_azure_env",
    "MSALTokenProvider",
    "AZURE_CLI_CLIENT_ID",
    "MICROSOFT_TENANT_ID",
    "AZURE_IDENTITY_AVAILABLE",
]


# Test function
if __name__ == '__main__':
    print("Testing TRAPI client (from shared module)...")
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
