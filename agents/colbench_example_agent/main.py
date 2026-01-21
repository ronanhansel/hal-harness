import os
import re
import sys
from pathlib import Path

# Add current directory and agents directory to path for imports
_this_dir = Path(__file__).resolve().parent
_agents_dir = _this_dir.parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from sweet_rl.environments.human_interaction_env import HumanInteractionEnv
from sweet_rl.environments.human_design_interaction_env import HumanDesignInteractionEnv
from openai import OpenAI
import anthropic

# Import from shared module (preferred) or fall back to local azure_client
try:
    from shared.azure_utils import get_trapi_client, resolve_deployment_name, TRAPI_DEPLOYMENT_MAP
    from shared.model_utils import uses_max_completion_tokens
except ImportError:
    # Fallback to local azure_client for backwards compatibility
    from azure_client import get_trapi_client, TRAPI_DEPLOYMENT_MAP, resolve_deployment_name

    def uses_max_completion_tokens(model_id: str) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        model_lower = model_id.lower()
        if model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            return True
        if model_lower.startswith("gpt-5") or "gpt-5" in model_lower:
            return True
        return False


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
            # Build request parameters
            request_params = {
                "model": self.deployment_name,
                "messages": messages,
                "max_completion_tokens": 16384,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
            }
            # Add extra headers for DeepSeek models
            if 'deepseek' in self.model_id.lower():
                request_params["extra_headers"] = {"extra-parameters": "pass-through"}
            completion = self.client.chat.completions.create(**request_params)
        else:
            # Build request parameters
            request_params = {
                "model": self.deployment_name,
                "messages": messages,
                "temperature": self.temperature,
            }
            # Use max_completion_tokens for GPT-5/O-series, max_tokens for others
            if uses_max_completion_tokens(self.model_id):
                request_params["max_completion_tokens"] = 16384
            else:
                request_params["max_tokens"] = 16384
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
