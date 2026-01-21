import os
import re
import sys
from pathlib import Path

# CRITICAL: Add current directory to path FIRST, before any other imports
# This is needed for sweet_rl which is bundled in this agent directory
_this_dir = Path(__file__).resolve().parent
_agents_dir = _this_dir.parent

# Force add to beginning of sys.path (don't just check if present)
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_agents_dir))

# Debug: Print path info
print(f"[colbench_agent] _this_dir = {_this_dir}")
print(f"[colbench_agent] sweet_rl dir exists = {(_this_dir / 'sweet_rl').exists()}")
print(f"[colbench_agent] sys.path[0:5] = {sys.path[0:5]}")

# Import sweet_rl - should work now that path is set
from sweet_rl.environments.human_interaction_env import HumanInteractionEnv
from sweet_rl.environments.human_design_interaction_env import HumanDesignInteractionEnv

from openai import OpenAI, AzureOpenAI
import anthropic

# Import from shared module
try:
    from shared.azure_utils import get_trapi_client, resolve_deployment_name, TRAPI_DEPLOYMENT_MAP
    from shared.model_utils import uses_max_completion_tokens, supports_temperature
    SHARED_AVAILABLE = True
except ImportError:
    SHARED_AVAILABLE = False
    print("[colbench_agent] Warning: shared module not available, using fallback")

    # Minimal fallback definitions
    TRAPI_DEPLOYMENT_MAP = {
        'gpt-4o': 'gpt-4o_2024-11-20',
        'gpt-4.1': 'gpt-4.1_2025-04-14',
        'gpt-5': 'gpt-5_2025-08-07',
        'o3': 'o3_2025-04-16',
        'o3-mini': 'o3-mini_2025-01-31',
        'o4-mini': 'o4-mini_2025-04-16',
    }

    def _normalize_model_id(model_id: str) -> str:
        """Normalize model ID by stripping provider prefixes."""
        model_lower = model_id.lower()
        for prefix in ('openai/', 'azure/', 'anthropic/'):
            if model_lower.startswith(prefix):
                return model_lower[len(prefix):]
        return model_lower

    def resolve_deployment_name(model: str) -> str:
        model_normalized = _normalize_model_id(model)
        return TRAPI_DEPLOYMENT_MAP.get(model_normalized, model_normalized)

    def uses_max_completion_tokens(model_id: str) -> bool:
        model_lower = _normalize_model_id(model_id)
        return any(model_lower.startswith(p) for p in ('o1', 'o3', 'o4', 'gpt-5'))

    def supports_temperature(model_id: str) -> bool:
        model_lower = _normalize_model_id(model_id)
        return not any(model_lower.startswith(p) for p in ('o1', 'o3', 'o4', 'gpt-5'))

    def get_trapi_client():
        """Fallback TRAPI client creation."""
        from openai import AzureOpenAI

        # Azure constants
        AZURE_CLI_CLIENT_ID = '04b07795-8ddb-461a-bbee-02f9e1bf7b46'
        MICROSOFT_TENANT_ID = '72f988bf-86f1-41af-91ab-2d7cd011db47'
        DEFAULT_TRAPI_ENDPOINT = 'https://trapi.research.microsoft.com/gcr/shared'
        DEFAULT_TRAPI_API_VERSION = '2025-03-01-preview'
        DEFAULT_TRAPI_SCOPE = 'api://trapi/.default'

        endpoint = os.environ.get('TRAPI_ENDPOINT', DEFAULT_TRAPI_ENDPOINT)
        api_version = os.environ.get('TRAPI_API_VERSION', DEFAULT_TRAPI_API_VERSION)
        scope = os.environ.get('TRAPI_SCOPE', DEFAULT_TRAPI_SCOPE)
        max_retries = int(os.environ.get('TRAPI_MAX_RETRIES', 500))
        timeout = float(os.environ.get('TRAPI_TIMEOUT', 1800))

        # Method 1: Pre-fetched token
        prefetched_token = os.environ.get('AZURE_OPENAI_AD_TOKEN')
        if prefetched_token:
            print(f"[colbench_agent] Using pre-fetched Azure AD token")
            return AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token=prefetched_token,
                api_version=api_version,
                max_retries=max_retries,
                timeout=timeout,
            )

        # Method 2: MSAL token provider
        try:
            import msal
            cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')
            if os.path.exists(cache_path):
                cache = msal.SerializableTokenCache()
                with open(cache_path, 'r') as f:
                    cache.deserialize(f.read())
                app = msal.PublicClientApplication(
                    AZURE_CLI_CLIENT_ID,
                    authority=f'https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}',
                    token_cache=cache,
                )
                accounts = app.get_accounts()
                if accounts:
                    def token_provider():
                        result = app.acquire_token_silent([scope], account=accounts[0])
                        if result and 'access_token' in result:
                            return result['access_token']
                        raise RuntimeError("Failed to acquire token from MSAL cache")
                    # Test it works
                    token_provider()
                    print(f"[colbench_agent] Using MSAL token provider")
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
            print(f"[colbench_agent] MSAL fallback failed: {e}")

        # Method 3: Azure Identity
        try:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            credential = AzureCliCredential()
            token_provider = get_bearer_token_provider(credential, scope)
            print(f"[colbench_agent] Using AzureCliCredential")
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
            print(f"[colbench_agent] Azure Identity fallback failed: {e}")

        raise RuntimeError(
            "No Azure credentials available. Options:\n"
            "1. Set AZURE_OPENAI_AD_TOKEN environment variable\n"
            "2. Mount ~/.azure directory with MSAL cache\n"
            "3. Install azure-identity and run 'az login'"
        )


def get_trapi_deployment(model_name: str) -> str:
    """Map model name to TRAPI deployment name."""
    return resolve_deployment_name(model_name)


class APIAgent:
    def __init__(self,
                 client,
                 model_id,
                 agent_prompt,
                 temperature=1.0,
                 reasoning_effort=None,
                 use_trapi=False):
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.client = client
        self.agent_prompt = agent_prompt
        self.reasoning_effort = reasoning_effort
        self.use_trapi = use_trapi

        # Map to TRAPI deployment name if using TRAPI
        if use_trapi:
            self.deployment_name = get_trapi_deployment(model_id)
        else:
            self.deployment_name = model_id

    def get_action(self, messages):
        if messages is None:
            return None
        messages = [{"role": "user", "content": self.agent_prompt}] + messages
        if self.model_id == "claude-3-7-sonnet-20250219":
            if self.reasoning_effort is not None:
                message = self.client.messages.create(
                    model=self.model_id,
                    thinking = {
                        "type": "enabled",
                        "budget_tokens": 4096
                    },
                    max_tokens=16384,
                    messages=messages
                )
            else:
                message = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=16384,
                    messages=messages
                )
            return message.content[-1].text
        elif "gemini" in self.model_id:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=16384,
                temperature=self.temperature,
            )

        elif self.reasoning_effort is not None:
            # Build request parameters for reasoning models
            request_params = {
                "model": self.deployment_name,
                "messages": messages,
                "max_completion_tokens": 16384,
                "reasoning_effort": self.reasoning_effort,
            }
            # Only add temperature if model supports it
            if supports_temperature(self.model_id):
                request_params["temperature"] = self.temperature
            # Add extra headers for DeepSeek models
            if 'deepseek' in self.model_id.lower():
                request_params["extra_headers"] = {"extra-parameters": "pass-through"}
            completion = self.client.chat.completions.create(**request_params)
        else:
            # Build request parameters
            request_params = {
                "model": self.deployment_name,
                "messages": messages,
            }
            # Use max_completion_tokens for GPT-5/O-series, max_tokens for others
            if uses_max_completion_tokens(self.model_id):
                request_params["max_completion_tokens"] = 16384
            else:
                request_params["max_tokens"] = 16384
            # Only add temperature if model supports it
            if supports_temperature(self.model_id):
                request_params["temperature"] = self.temperature
            # Add extra headers for DeepSeek models
            if 'deepseek' in self.model_id.lower():
                request_params["extra_headers"] = {"extra-parameters": "pass-through"}
            completion = self.client.chat.completions.create(**request_params)

        # Strip thinking tags for DeepSeek models
        content = completion.choices[0].message.content
        if 'deepseek' in self.model_id.lower() and content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        return content

def _run_task_impl(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """Internal function that runs the actual task logic (no tracing)."""
    assert 'model_name' in kwargs, 'model_name is required'

    # Use TRAPI by default for Azure deployment
    use_trapi = os.environ.get('USE_TRAPI', 'true').lower() == 'true'

    task_id = list(input.keys())[0]
    task_data = input[task_id]

    # Environment client - always use TRAPI for stability
    if use_trapi:
        env_client = get_trapi_client()
        env_model_name = get_trapi_deployment("gpt-4o")
    else:
        env_client = OpenAI()
        env_model_name = "gpt-4o-2024-08-06"

    # Agent client - use TRAPI for GPT/O-series models
    if "gpt" in kwargs['model_name'].lower() or "o3" in kwargs['model_name'].lower() or "o4-mini" in kwargs['model_name'].lower():
        if use_trapi:
            agent_client = get_trapi_client()
        else:
            agent_client = OpenAI()
    elif kwargs['model_name'] == "claude-3-7-sonnet-20250219":
        agent_client = anthropic.Anthropic()
        use_trapi = False  # Anthropic doesn't use TRAPI
    elif "gemini" in kwargs['model_name']:
        agent_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        use_trapi = False
    else:
        # Default to TRAPI for all other models (including DeepSeek)
        if use_trapi:
            agent_client = get_trapi_client()
        else:
            agent_client = OpenAI()

    # Load prompt file relative to this script's location
    if task_data["task_type"] == "code":
        prompt_file = _this_dir / "code_agent_prompt.txt"
    else:
        prompt_file = _this_dir / "html_agent_prompt.txt"

    with open(prompt_file, "r") as f:
        agent_prompt = f.read()
    agent = APIAgent(
        agent_client,
        kwargs['model_name'],
        agent_prompt,
        reasoning_effort=kwargs.get('reasoning_effort'),
        use_trapi=use_trapi
    )
    if task_data["task_type"] == "code":
        env = HumanInteractionEnv(env_client, task_data["human_prompt"], env_model_name)
    else:
        env = HumanDesignInteractionEnv(env_client, task_data["human_prompt"],
                                        env_model_name,
                                        temp_path=task_data['cache_path'],
                                        gpt_client=True)



    ### ENV SETUP (usually this should be untouched) ###
    observation = env.reset(task_data["problem_description"], task_data["hidden_information"])
    for i in range(10):
        response = agent.get_action(observation)
        observation, _, _ = env.step(response)
    dialogue_history = [{"role": d["role"], "content": d["content"]} for d in env.get_dialogue_history()]
    answer = env.answer

    if task_data["task_type"] == "html":
        env.driver.quit()

    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {task_id: {"answer": answer, "dialogue_history": dialogue_history, "task":{
                      "test_cases": task_data["test_cases"] if task_data["task_type"] == "code" else None,
                      "ground_truth": task_data["hidden_information"]}}}


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    Main entry point for HAL harness.
    Calls the implementation directly - tracing is handled by HAL harness.

    Note: We removed the dynamic @weave.op() wrapper because Weave cannot
    introspect dynamically-created nested functions, causing:
    "Error getting code deps: 'NoneType' object has no attribute '__dict__'"

    HAL harness already wraps agent calls in Weave ops, so individual LLM calls
    will still be traced as children of the main agent call.
    """
    return _run_task_impl(input, **kwargs)
