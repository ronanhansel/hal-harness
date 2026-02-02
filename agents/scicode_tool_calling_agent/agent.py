import os
import ast
import sys
from pathlib import Path
from smolagents import LiteLLMModel, CodeAgent, Tool, DuckDuckGoSearchTool, PythonInterpreterTool, FinalAnswerTool
import time

import smolagents.models
import smolagents.local_python_executor as executor_module
import re

# Add parent directory to path for model_quirks import
_agents_dir = Path(__file__).resolve().parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

# Import shared model quirks module
try:
    from model_quirks import supports_stop_parameter, patch_smolagents
    MODEL_QUIRKS_AVAILABLE = True
except ImportError:
    MODEL_QUIRKS_AVAILABLE = False

# ============================================================================
# SCIPY COMPATIBILITY SHIM
# The SciCode benchmark uses deprecated scipy.integrate.simps but modern scipy
# only has scipy.integrate.simpson. Add simps as an alias to prevent errors.
# This fixes task 12 and other tasks that use Simpson's rule integration.
# ============================================================================
try:
    from scipy import integrate
    if not hasattr(integrate, 'simps'):
        integrate.simps = integrate.simpson
except ImportError:
    pass

# Also add numpy.trapz alias for compatibility (deprecated in favor of numpy.trapezoid)
try:
    import numpy as np
    if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid
except ImportError:
    pass

# ============================================================================
# MATMULT OPERATOR PATCH
# Store the original evaluate_binop function
# ============================================================================
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

# Use shared model_quirks if available, otherwise fallback to local implementation
if MODEL_QUIRKS_AVAILABLE:
    # Patch smolagents with the shared implementation
    patch_smolagents()
else:
    # Fallback local implementation
    def supports_stop_parameter(model_id: str) -> bool:
        """
        Check if the model supports the `stop` parameter.

        Not supported with reasoning models openai/o1, o3, o4-mini, and gpt-5 (and their versioned variants).
        """
        model_name = model_id.split("/")[-1]
        # Normalize: replace underscores with dashes for consistent matching
        model_name = model_name.replace("_", "-")
        # o-series and gpt-5 (including versioned variants) don't support stop parameter
        pattern = r"^(o[134](-mini)?|gpt-5)([-\d].*)?$"
        return not re.match(pattern, model_name, re.IGNORECASE)

    # Replace the function in smolagents
    smolagents.models.supports_stop_parameter = supports_stop_parameter

AUTHORIZED_IMPORTS = [
    # === Core Python modules ===
    "os",
    "sys",
    "time",
    "datetime",
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
    "io",
    "re",
    "json",
    "csv",
    "zipfile",
    "pathlib",
    "glob",
    "shutil",
    "struct",
    "textwrap",
    "typing",
    "warnings",
    "logging",
    "builtins.dir",
    "builtins.slice",
    "unicodedata",
    "stat",

    # === Numpy - explicit submodules for smolagents interpreter ===
    "numpy", "numpy.*",
    "numpy.linalg",
    "numpy.fft",
    "numpy.random",  # For random number generation (task 80 - Andersen thermostat)
    "numpy.ma",
    "numpy.polynomial",

    # === Scipy - explicit submodules for smolagents interpreter ===
    "scipy", "scipy.*",
    "scipy.integrate",
    "scipy.optimize",
    "scipy.linalg",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "scipy.special",
    "scipy.signal",
    "scipy.interpolate",
    "scipy.constants",
    "scipy.stats",
    "scipy.ndimage",
    "scipy.io",
    "scipy.fft",
    "scipy.spatial",

    # === Data Science Core ===
    "pandas", "pandas.*",
    "sympy", "sympy.*",
    "sklearn", "sklearn.*",  # scikit-learn imports as sklearn
    "statsmodels", "statsmodels.*",
    "statistics",
    "fractions",

    # === Visualization ===
    "matplotlib", "matplotlib.*",
    "mpl_toolkits", "mpl_toolkits.*",
    "mpl_toolkits.mplot3d",
    "seaborn", "seaborn.*",
    "plotly", "plotly.*",
    "PIL", "PIL.*",  # pillow imports as PIL

    # === Deep Learning ===
    "torch", "torch.*",  # pytorch package imports as torch
    "transformers", "transformers.*",
    "dgl", "dgl.*",

    # === Single-cell / Bioinformatics ===
    "scanpy", "scanpy.*",
    "anndata", "anndata.*",
    "mudata", "mudata.*",
    "muon", "muon.*",
    "leidenalg", "leidenalg.*",
    "igraph", "igraph.*",
    "Bio", "Bio.*",  # biopython imports as Bio

    # === Neuroimaging / Biosignals ===
    "mne", "mne.*",
    "neurokit2", "neurokit2.*",
    "biopsykit", "biopsykit.*",

    # === Chemistry / Materials Science ===
    "rdkit", "rdkit.*",
    "deepchem", "deepchem.*",
    "pubchempy", "pubchempy.*",
    "pymatgen", "pymatgen.*",

    # === Molecular Dynamics / Structural Biology ===
    "MDAnalysis", "MDAnalysis.*",
    "prolif", "prolif.*",

    # === Geospatial / Climate ===
    "iris", "iris.*",  # scitools-iris imports as iris
    "cartopy", "cartopy.*",
    "geopandas", "geopandas.*",
    "xarray", "xarray.*",
    "netCDF4", "netCDF4.*",
    "shapely", "shapely.*",
    "pyproj", "pyproj.*",

    # === File Formats ===
    "h5py", "h5py.*",
    "tables", "tables.*",  # PyTables imports as tables
    "openpyxl", "openpyxl.*",
    "xlrd", "xlrd.*",
    "xml", "xml.*",

    # === Web / API ===
    "requests", "requests.*",
    "urllib", "urllib.*",
    "bs4", "bs4.*",
    "aiohttp", "aiohttp.*",

    # === Misc Scientific ===
    "networkx", "networkx.*",
    "cv2",  # opencv-python-headless imports as cv2
    "skimage", "skimage.*",  # scikit-image imports as skimage
    "imageio", "imageio.*",
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
    # Determine if we should use Azure/TRAPI direct access
    # Default to Azure/TRAPI for all OpenAI models (faster, more reliable)
    def _should_use_azure():
        # Explicit opt-out
        if os.environ.get('USE_TRAPI', '').lower() == 'false':
            return False
        if os.environ.get('USE_DIRECT_AZURE', '').lower() == 'false':
            return False

        # Normalize model name
        model_lower = model_params.get('model_id', '').lower()
        for prefix in ('openai/', 'azure/'):
            if model_lower.startswith(prefix):
                model_lower = model_lower[len(prefix):]
                break

        # Use TRAPI for all OpenAI models by default
        is_openai_model = (
            'gpt-' in model_lower or
            model_lower.startswith('gpt-') or
            model_lower.startswith('o1') or
            model_lower.startswith('o3') or
            model_lower.startswith('o4') or
            'deepseek' in model_lower
        )
        return is_openai_model

    # Initialize model - use AzureDirectModel for OpenAI models
    if _should_use_azure():
        azure_model_loaded = False
        # Try shared module first
        try:
            from shared.agent_wrapper import create_model_for_agent
            print(f"[INFO] Using shared module for Azure model creation")
            model = create_model_for_agent(
                model_name=model_params.get('model_id', 'gpt-4o'),
                reasoning_effort=model_params.get('reasoning_effort'),
                temperature=model_params.get('temperature', 0.7),
            )
            azure_model_loaded = True
        except ImportError:
            pass

        # Fallback to local azure_direct_model
        if not azure_model_loaded:
            try:
                from azure_direct_model import AzureDirectModel
                print(f"[INFO] Using AzureDirectModel for direct TRAPI access")
                model = AzureDirectModel(
                    model_id=model_params.get('model_id', 'gpt-4o'),
                    temperature=model_params.get('temperature', 0.7),
                    max_tokens=model_params.get('max_tokens', 32768),
                    num_retries=model_params.get('num_retries', 500),
                    timeout=model_params.get('timeout', 1800),
                )
            except Exception as e:
                import traceback
                print(f"[ERROR] Failed to use AzureDirectModel: {e}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"AzureDirectModel initialization failed: {e}. Check Azure credentials and shared module setup.")
    else:
        # Should not happen - we always want TRAPI
        raise RuntimeError(f"use_azure=False but TRAPI is required. Set USE_DIRECT_AZURE=true. model_params={model_params}")

    # Create customized PythonInterpreterTool that allows 'open'
    python_interpreter = PythonInterpreterTool(authorized_imports=AUTHORIZED_IMPORTS)
    python_interpreter.base_python_tools["open"] = open

    # Create a CodeAgent instance with the specified model
    # Note: The local Python executor is patched to support:
    # - @ operator for matrix multiplication (via numpy.matmul)
    # - numpy.random for stochastic simulations
    # - heapq for priority queue operations
    agent = CodeAgent(
        tools=[
            RateLimitAwareDuckDuckGoSearchTool(),
            python_interpreter,
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

