"""
Direct Azure OpenAI Model for smolagents - wrapper file

This wrapper imports from the shared module for centralized maintenance.
The shared folder is automatically copied into Docker containers by docker_runner.py.

Usage:
    from azure_direct_model import AzureDirectModel

    model = AzureDirectModel(model_id='gpt-4o')
    # Use with smolagents CodeAgent
"""

import sys
from pathlib import Path

# Add parent directory to path for shared module import
_agents_dir = Path(__file__).resolve().parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

# Import from shared module (works in Docker and locally)
# The docker_runner.py automatically copies the shared folder into containers
from shared.azure_direct_model import (
    AzureDirectModel,
    create_model,
    create_trapi_client,
    MSALTokenProvider,
    AZURE_CLI_CLIENT_ID,
    MICROSOFT_TENANT_ID,
    TRAPI_DEPLOYMENT_MAP,
)

# Re-export for backwards compatibility
__all__ = [
    "AzureDirectModel",
    "create_model",
    "create_trapi_client",
    "MSALTokenProvider",
    "AZURE_CLI_CLIENT_ID",
    "MICROSOFT_TENANT_ID",
    "TRAPI_DEPLOYMENT_MAP",
]
