"""
Configuration Utilities for HAL Agent Benchmarking

This module provides utilities for loading and parsing the unified
model_to_baseline_*.json configuration files.

Usage:
    from shared.config_utils import load_benchmark_config, get_model_entry

    config = load_benchmark_config("scicode")
    entry = get_model_entry(config, "gpt-5_scicode_tool_calling")
    agent_dir = entry["agent_dir"]
    model_id = entry["model_id"]
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_benchmark_config(
    benchmark: str,
    config_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load the model_to_baseline configuration for a benchmark.

    Args:
        benchmark: Benchmark name (e.g., "scicode", "corebench", "colbench")
        config_dir: Directory containing config files (default: repo root)

    Returns:
        Dict containing the full configuration with _meta and model entries

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_dir is None:
        # Default to repo root (parent of hal-harness)
        config_dir = Path(__file__).resolve().parents[3]

    config_path = config_dir / f"model_to_baseline_{benchmark}.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return json.loads(config_path.read_text(encoding="utf-8"))


def get_model_entries(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get all model entries from config (excluding _meta).

    Args:
        config: Full config dict

    Returns:
        Dict mapping config keys to model entry dicts
    """
    return {k: v for k, v in config.items() if not k.startswith("_")}


def get_model_entry(
    config: Dict[str, Any],
    config_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific model entry from config.

    Args:
        config: Full config dict
        config_key: Key to look up (e.g., "gpt-5_scicode_tool_calling")

    Returns:
        Model entry dict or None if not found
    """
    return config.get(config_key)


def get_agent_info(entry: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract agent directory and function from a model entry.

    Args:
        entry: Model entry dict

    Returns:
        Tuple of (agent_dir, agent_function)

    Raises:
        KeyError: If agent_dir or agent_function not in entry
    """
    agent_dir = entry.get("agent_dir")
    agent_function = entry.get("agent_function", "main.run")

    if not agent_dir:
        raise KeyError("agent_dir not found in model entry")

    return agent_dir, agent_function


def build_agent_args(
    entry: Dict[str, Any],
    include_agent_info: bool = False,
) -> Dict[str, Any]:
    """
    Build agent_args dict from a model entry.

    Args:
        entry: Model entry dict
        include_agent_info: Whether to include agent_dir and agent_function

    Returns:
        Dict suitable for passing to agent run() function

    Example entry:
        {
            "model_id": "openai/gpt-4.1_2025-04-14",
            "short_name": "gpt-4.1",
            "agent_dir": "hal-harness/agents/scicode_tool_calling_agent",
            "agent_function": "main.run",
            "reasoning_effort": "high",
            "max_steps": 5
        }

    Returns:
        {
            "model_name": "openai/gpt-4.1_2025-04-14",
            "reasoning_effort": "high",
            "max_steps": 5
        }
    """
    args: Dict[str, Any] = {}

    # model_name is the API model ID
    model_id = entry.get("model_id")
    if model_id:
        args["model_name"] = model_id

    # Copy relevant parameters
    param_keys = [
        "reasoning_effort",
        "temperature",
        "max_steps",
        "budget",
        "max_tokens",
    ]
    for key in param_keys:
        if key in entry:
            args[key] = entry[key]

    # Optionally include agent info
    if include_agent_info:
        if "agent_dir" in entry:
            args["agent_dir"] = entry["agent_dir"]
        if "agent_function" in entry:
            args["agent_function"] = entry["agent_function"]

    return args


def list_models_by_agent(
    config: Dict[str, Any],
    agent_name: str,
) -> List[str]:
    """
    List all config keys that use a specific agent.

    Args:
        config: Full config dict
        agent_name: Agent name to filter by (e.g., "scicode_tool_calling_agent")

    Returns:
        List of config keys using that agent
    """
    results = []
    for key, entry in get_model_entries(config).items():
        agent_dir = entry.get("agent_dir", "")
        if agent_name in agent_dir:
            results.append(key)
    return results


def list_agents(config: Dict[str, Any]) -> List[str]:
    """
    List all unique agent directories in config.

    Args:
        config: Full config dict

    Returns:
        List of unique agent directory paths
    """
    agents = set()
    for entry in get_model_entries(config).values():
        if "agent_dir" in entry:
            agents.add(entry["agent_dir"])
    return sorted(agents)


def filter_by_model(
    config: Dict[str, Any],
    model_pattern: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Filter config entries by model pattern.

    Args:
        config: Full config dict
        model_pattern: Pattern to match in model_id or short_name

    Returns:
        Dict of matching entries
    """
    results = {}
    pattern_lower = model_pattern.lower()

    for key, entry in get_model_entries(config).items():
        model_id = entry.get("model_id", "").lower()
        short_name = entry.get("short_name", "").lower()

        if pattern_lower in model_id or pattern_lower in short_name:
            results[key] = entry

    return results


def get_benchmark_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get benchmark metadata from config.

    Args:
        config: Full config dict

    Returns:
        Dict with benchmark metadata (from _meta key)
    """
    return config.get("_meta", {})


# =============================================================================
# Convenience functions for common operations
# =============================================================================

def resolve_agent_path(
    agent_dir: str,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Resolve agent_dir to absolute path.

    Args:
        agent_dir: Relative agent directory (e.g., "hal-harness/agents/scicode_tool_calling_agent")
        repo_root: Repository root directory (default: auto-detect)

    Returns:
        Absolute Path to agent directory
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    agent_path = Path(agent_dir)
    if not agent_path.is_absolute():
        agent_path = repo_root / agent_path

    return agent_path


def validate_config_entry(entry: Dict[str, Any]) -> List[str]:
    """
    Validate a config entry and return list of issues.

    Args:
        entry: Model entry dict

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    required_fields = ["model_id", "agent_dir", "agent_function"]
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")

    if "model_id" in entry and not entry["model_id"]:
        errors.append("model_id cannot be empty")

    if "agent_dir" in entry:
        # Check if path seems reasonable
        agent_dir = entry["agent_dir"]
        if not agent_dir.startswith("hal-harness/agents/"):
            errors.append(f"agent_dir should start with 'hal-harness/agents/': {agent_dir}")

    return errors


# =============================================================================
# Test function
# =============================================================================
if __name__ == "__main__":
    import sys

    benchmark = sys.argv[1] if len(sys.argv) > 1 else "scicode"

    print(f"Loading config for benchmark: {benchmark}")
    config = load_benchmark_config(benchmark)

    print(f"\nBenchmark info: {get_benchmark_info(config)}")

    entries = get_model_entries(config)
    print(f"\nFound {len(entries)} model entries:")

    for key, entry in entries.items():
        agent_dir, agent_func = get_agent_info(entry)
        agent_args = build_agent_args(entry)
        print(f"\n  {key}:")
        print(f"    model_id: {entry.get('model_id')}")
        print(f"    agent_dir: {agent_dir}")
        print(f"    agent_args: {agent_args}")

        errors = validate_config_entry(entry)
        if errors:
            print(f"    ERRORS: {errors}")

    print(f"\nUnique agents: {list_agents(config)}")
