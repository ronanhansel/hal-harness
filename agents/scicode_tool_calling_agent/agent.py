import os
import ast
from smolagents import LiteLLMModel, CodeAgent, Tool, DuckDuckGoSearchTool, PythonInterpreterTool, FinalAnswerTool
import time

import smolagents.models
import smolagents.local_python_executor as executor_module
import re

# Store the original evaluate_binop function
_original_evaluate_binop = executor_module.evaluate_binop

def _patched_evaluate_binop(binop, state, static_tools, custom_tools, authorized_imports):
    """
    Patched version of evaluate_binop that adds support for the @ (MatMult) operator.
    This fixes tasks 28, 71 which use matrix multiplication with numpy arrays.
    """
    # First try the MatMult operator
    if isinstance(binop.op, ast.MatMult):
        left_val = executor_module.evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
        right_val = executor_module.evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)
        # Use numpy's matmul for the @ operator
        import numpy as np
        return np.matmul(left_val, right_val)
    # Fall back to original implementation for other operators
    return _original_evaluate_binop(binop, state, static_tools, custom_tools, authorized_imports)

# Replace the function in smolagents local_python_executor
executor_module.evaluate_binop = _patched_evaluate_binop

def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3, openai/o4-mini, and gpt-5 (and their versioned variants).
    """
    model_name = model_id.split("/")[-1]
    # Normalize: replace underscores with dashes for consistent matching
    model_name = model_name.replace("_", "-")
    # o3, o4-mini, and gpt-5 (including versioned variants) don't support stop parameter
    # Pattern matches: o3, o3-2025-04-16, o4-mini, o4-mini-2025-04-16, gpt-5, etc.
    pattern = r"^(o[34](-mini)?|gpt-5)([-\d].*)?$"
    return not re.match(pattern, model_name, re.IGNORECASE)

# Replace the function in smolagents
smolagents.models.supports_stop_parameter = supports_stop_parameter

AUTHORIZED_IMPORTS = [
    "os",
    "time",
    "pickle",
    "itertools",
    "random",
    "copy",
    "math",
    "cmath",
    "collections",
    "functools",
    "heapq",  # For priority queue operations (task 35)
    "queue",  # Alternative for priority queues
    # Numpy - explicit to work with smolagents interpreter
    "numpy",
    "numpy.linalg",
    "numpy.fft",
    "numpy.random",  # For random number generation (task 80 - Andersen thermostat)
    # Scipy - must list explicit submodules for interpreter
    "scipy",
    "scipy.integrate",
    "scipy.optimize",
    "scipy.linalg",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "scipy.special",
    "scipy.signal",
    "scipy.interpolate",
    "scipy.constants",
    # Other
    "mpl_toolkits.mplot3d",
    "sympy",
    "builtins.dir",
    "builtins.slice"
]

class ModifiedWikipediaSearchTool(Tool):
    """
    Modifies WikipediaSearchTool to search for any pages relating to request, selects first page, and returns summary.

    Attributes:
        user_agent (str): A custom user-agent string to identify the project. This is required as per Wikipedia API policies, read more here: http://github.com/martin-majlis/Wikipedia-API/blob/master/README.rst
        language (str): The language in which to retrieve Wikipedia articles.
                http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (str): Defines the content to fetch. Can be "summary" for a short summary or "text" for the full article.
        extract_format (str): Defines the output format. Can be `"WIKI"` or `"HTML"`.

    """

    name = "wikipedia_search"
    description = "Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "Smolagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        super().__init__()
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e
        if not user_agent:
            raise ValueError("User-agent is required. Provide a meaningful identifier for your project.")

        self.user_agent = user_agent 
        self.language = language
        self.content_type = content_type

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")

        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    def forward(self, query: str) -> str:
        try:
            import wikipedia
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia` to run this tool: for instance run `pip install wikipedia`"
            ) from e
        try:
            page = self.wiki.page(query)

            if not page.exists():
                # Try searching for related pages
                search_results = wikipedia.search(query)
                if search_results:
                    # Use the top search result
                    top_result = search_results[0]
                    page = self.wiki.page(top_result)
                    if not page.exists():
                        return f"No Wikipedia page found for '{query}', even after searching."
                else:
                    return f"No Wikipedia page found for '{query}', and no related results were found."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class RateLimitAwareDuckDuckGoSearchTool(Tool):
    """
    DuckDuckGo search tool with rate limiting awareness and fallback to Wikipedia
    """
    name = "web_search"
    description = "Searches the web using DuckDuckGo with rate limiting protection and Wikipedia fallback."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query.",
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.last_search_time = 0
        self.min_interval = 2.0  # Minimum 2 seconds between searches
        self.wikipedia_tool = ModifiedWikipediaSearchTool()
        
    def forward(self, query: str) -> str:
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_search_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        
        try:
            # Try DuckDuckGo first
            ddg_tool = DuckDuckGoSearchTool()
            result = ddg_tool.forward(query)
            self.last_search_time = time.time()
            
            # Check if we got a rate limit error
            if "202 Ratelimit" in result or "rate limit" in result.lower():
                # Fallback to Wikipedia
                return f"⚠️ Web search rate limited. Falling back to Wikipedia.\n\n{self.wikipedia_tool.forward(query)}"
            
            return result
            
        except Exception as e:
            # If DuckDuckGo fails, fallback to Wikipedia
            error_msg = str(e)
            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                return f"⚠️ Web search rate limited. Falling back to Wikipedia.\n\n{self.wikipedia_tool.forward(query)}"
            else:
                return f"⚠️ Web search failed ({error_msg}). Falling back to Wikipedia.\n\n{self.wikipedia_tool.forward(query)}"
        
        
def get_agent(model_params) -> CodeAgent:
    """
    Returns a CodeAgent with the specified model name.
    
    Args:
        model_name (str): The name of the model to use.
        
    Returns:
        CodeAgent: An instance of CodeAgent configured with the specified model.
    """
    # Initialize the LiteLLMModel with the specified model name
    model = LiteLLMModel(**model_params)

    # Create a CodeAgent instance with the specified model
    # Note: The local Python executor is patched to support:
    # - @ operator for matrix multiplication (via numpy.matmul)
    # - numpy.random for stochastic simulations
    # - heapq for priority queue operations
    agent = CodeAgent(
        tools=[
            RateLimitAwareDuckDuckGoSearchTool(),
            PythonInterpreterTool(authorized_imports=AUTHORIZED_IMPORTS),
            ModifiedWikipediaSearchTool(),
            FinalAnswerTool(description = """Submit your final Python function implementation.

CRITICAL FORMAT REQUIREMENTS:
1. Your response MUST follow this exact format:
   Thought: I will submit the final implementation.
   Code:
   ```py
   final_answer(\"\"\"
   def function_name(...):
       # Your implementation here
       return result
   \"\"\")
   ```<end_code>

2. The argument to final_answer() must be the Python function code as a string - NOT prose or explanations.

3. Do NOT include:
   - Example usage or test code
   - Dependencies or imports (they are provided separately)
   - Functions from previous steps (use headers as provided)
   - Explanatory text outside the function

COMPATIBILITY NOTES:
- Use scipy.integrate.simpson (not simps - deprecated)
- Use numpy.trapezoid (not trapz - deprecated)
- You can use @ for matrix multiplication
- You can use numpy.random and heapq""")
        ],
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        model=model,
        planning_interval=3,
        max_steps=5,
        verbosity_level=2,
        code_block_tags="markdown",  # Use ```python blocks instead of <code> tags
    )

    return agent

