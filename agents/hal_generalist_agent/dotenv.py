"""
Minimal subset of python-dotenv used by the HAL generalist agent.

The agent only needs a callable `load_dotenv`, so we provide a no-op implementation
to avoid importing the full dependency in restricted environments.
"""

from typing import Any, Optional


def load_dotenv(_path: Optional[str] = None, *_args: Any, **_kwargs: Any) -> None:
    """No-op replacement for python-dotenv's load_dotenv."""
    return None
