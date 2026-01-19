"""
Direct Azure Model for smolagents
Bypasses LiteLLM proxy for better stability

Usage:
    from azure_model import AzureModel

    model = AzureModel(model_id='gpt-4o')  # Uses TRAPI
    agent = CodeAgent(model=model, tools=[...])
"""

import os
from typing import List, Dict, Any, Optional
from smolagents.models import Model, MessageRole
from azure_client import get_trapi_client, resolve_deployment_name, TRAPI_DEPLOYMENT_MAP


class AzureModel(Model):
    """
    A Model class that directly calls Azure OpenAI / TRAPI endpoints.
    Compatible with smolagents framework.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 600,
        num_retries: int = 35,
        **kwargs,
    ):
        """
        Initialize the Azure model.

        Args:
            model_id: Model name (e.g., 'gpt-4o', 'o3-mini', 'azure/gpt-5')
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            num_retries: Number of retries on failure
        """
        super().__init__()
        self.model_id = model_id.replace('azure/', '')
        self.deployment_name = resolve_deployment_name(self.model_id)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.num_retries = num_retries
        self.kwargs = kwargs

        # Create the Azure client
        self.client = get_trapi_client(max_retries=num_retries)

        print(f"[AzureModel] Initialized: {self.model_id} -> {self.deployment_name}")

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            stop_sequences: Optional stop sequences
            grammar: Optional grammar constraint (not supported by Azure)

        Returns:
            Generated text response
        """
        # Build request parameters
        request_params = {
            "model": self.deployment_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Handle stop sequences (not supported by some reasoning models)
        if stop_sequences and self._supports_stop():
            request_params["stop"] = stop_sequences

        # Add any extra parameters
        for key in ["reasoning_effort", "top_p", "presence_penalty", "frequency_penalty"]:
            if key in kwargs:
                request_params[key] = kwargs[key]

        try:
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[AzureModel] Error: {type(e).__name__}: {e}")
            raise

    def _supports_stop(self) -> bool:
        """Check if model supports stop parameter."""
        model_lower = self.model_id.lower()
        # Reasoning models don't support stop
        if model_lower.startswith("o3") and "o3-mini" not in model_lower:
            return False
        if "o4-mini" in model_lower:
            return False
        if model_lower.startswith("gpt-5"):
            return False
        return True

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: ~4 chars per token for English
        return len(text) // 4


def create_azure_model(model_name: str, **kwargs) -> AzureModel:
    """
    Factory function to create an AzureModel.

    Args:
        model_name: Model name (e.g., 'azure/gpt-4o', 'gpt-5', 'o3-mini')
        **kwargs: Additional arguments passed to AzureModel

    Returns:
        AzureModel instance
    """
    return AzureModel(model_id=model_name, **kwargs)


# Test
if __name__ == "__main__":
    print("Testing AzureModel...")
    model = AzureModel(model_id="gpt-4o", temperature=0.5)

    messages = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]

    response = model(messages)
    print(f"Response: {response}")
