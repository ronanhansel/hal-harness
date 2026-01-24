import os
import json
import shutil
import uuid
import subprocess
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
from hal.benchmarks.base_benchmark import BaseBenchmark
from hal.utils.retry_handler import add_retry_to_runner
from hal.utils.trace_utils import get_trace_mode
from rich.progress import Progress, TaskID

# Get logger for verbose output
verbose_logger = logging.getLogger('agent_eval.verbose')

class LocalRunner:
    """Handles running agents locally in isolated environments"""
    
    def __init__(self, log_dir: str, max_concurrent: int = 1, conda_env: Optional[str] = None, benchmark: Optional[BaseBenchmark] = None, retry_config: Optional[Dict[str, Any]] = None):
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self.conda_env = conda_env
        self.temp_dirs: list[str] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self.benchmark = benchmark
        
        # Add retry functionality (enabled by default with sensible defaults)
        add_retry_to_runner(self, retry_config)

    async def run_agent(self, 
                       dataset: Dict[str, Any],
                       agent_function: str,
                       agent_dir: str,
                       agent_args: Dict[str, Any],
                       run_id: str,
                       benchmark: Optional[BaseBenchmark] = None,
                       progress: Optional[Progress] = None,
                       task: Optional[TaskID] = None) -> Dict[str, Any]:
        """
        Run agent on all tasks with concurrency control
        """
        try:
            self.benchmark = benchmark
            # Get run directory from benchmark if provided
            run_dir = benchmark.get_run_dir(run_id) if benchmark else f"results/{run_id}"
            submissions_file = os.path.join(run_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
            
            tasks = []
            for task_id, input_data in dataset.items():
                task_coro = self._process_task(
                    task_id=task_id,
                    input_data=input_data,
                    agent_function=agent_function,
                    agent_dir=agent_dir,
                    agent_args=agent_args,
                    run_id=run_id,
                    submissions_file=submissions_file,
                    progress=progress,
                    task=task
                )
                tasks.append(task_coro)
            
            # Run tasks with concurrency control
            results = await asyncio.gather(*tasks)
            
            # Merge results
            merged_results = {}
            for result in results:
                if result:
                    merged_results.update(result)
                    
            return merged_results

        finally:
            # Cleanup temp directories - CRITICAL: Must succeed to avoid filling up /tmp
            for temp_dir in self.temp_dirs:
                cleanup_success = False
                for attempt in range(3):
                    try:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                            cleanup_success = True
                            break
                        else:
                            cleanup_success = True
                            break
                    except Exception as e:
                        print(f"Cleanup attempt {attempt+1}/3 failed for {temp_dir}: {e}")
                        time.sleep(0.5)

                if not cleanup_success and os.path.exists(temp_dir):
                    # Last resort: try subprocess rm -rf
                    try:
                        import subprocess
                        subprocess.run(["rm", "-rf", str(temp_dir)], timeout=60, check=False)
                        if not os.path.exists(temp_dir):
                            cleanup_success = True
                    except Exception as e:
                        print(f"rm -rf cleanup failed for {temp_dir}: {e}")

                if not cleanup_success and os.path.exists(temp_dir):
                    print(f"CRITICAL: Failed to cleanup temp directory {temp_dir} - disk space may fill up!")

    async def _process_task(self,
                          task_id: str,
                          input_data: Any,
                          agent_function: str,
                          agent_dir: str,
                          agent_args: Dict[str, Any],
                          run_id: str,
                          submissions_file: str,
                          progress: Optional[Progress] = None,
                          task: Optional[TaskID] = None) -> Optional[Dict[str, Any]]:
        """Process a single task with semaphore control"""
        async with self._semaphore:
            print(f"Starting task {task_id} (active tasks: {self.max_concurrent - self._semaphore._value})")
            result = await self._run_single_task(
                task_id=task_id,
                input_data=input_data,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                run_id=run_id
            )
            
            # Write result to submissions file
            if result:
                async with self._file_lock:
                    with open(submissions_file, "a") as f:
                        json.dump(result, f)
                        f.write("\n")
            
            # Update progress after task completion
            if progress and task is not None:
                progress.update(task, advance=1)
            
            print(f"Completed task {task_id}")
            return result

    async def _run_single_task(self,
                             task_id: str,
                             input_data: Any,
                             agent_function: str,
                             agent_dir: str,
                             agent_args: Dict[str, Any],
                             run_id: str) -> Optional[Dict[str, Any]]:
        """
        Run agent on a single task in an isolated environment
        """
        # Create temporary directory
        temp_dir = Path(f"/tmp/agent_run_{uuid.uuid4()}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dirs.append(str(temp_dir))

        try:
            # Copy agent code
            shutil.copytree(agent_dir, temp_dir, dirs_exist_ok=True)

            # Write input and args files
            with open(temp_dir / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(temp_dir / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            # Copy task-specific files if they exist in input_data
            if isinstance(input_data, dict) and 'files' in input_data:
                for dest_path, src_path in input_data['files'].items():
                    # Remove 'root' prefix and leading slash if present
                    dest_path = dest_path.replace('/root/', '').lstrip('/')
                    
                    # Create destination directory structure
                    dest_full_path = temp_dir / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the file
                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as e:
                        error_msg = f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}"
                        verbose_logger.debug(error_msg)

            # Copy and run setup script if it exists
            # if self.benchmark and self.benchmark.setup_script:
            #     setup_script_src = Path(self.benchmark.setup_script)
            #     if setup_script_src.exists():
            #         setup_script_dest = temp_dir / "setup_script.sh"
            #         shutil.copy2(setup_script_src, setup_script_dest)
            #         setup_script_dest.chmod(0o755)

            #         verbose_logger.debug(f"Running setup script for task {task_id}")
            #         cmd = ["bash", str(setup_script_dest)]
            #         if self.conda_env:
            #             cmd = ["conda", "run", "-n", self.conda_env] + cmd
                    
            #         process = await asyncio.create_subprocess_exec(
            #             *cmd,
            #             cwd=str(temp_dir),
            #             stdout=asyncio.subprocess.PIPE,
            #             stderr=asyncio.subprocess.PIPE
            #         )
            #         stdout, stderr = await process.communicate()
                    
            #         # Log setup script output
            #         if stdout:
            #             verbose_logger.debug(f"Setup script stdout for task {task_id}:\n{stdout.decode()}")
            #         if stderr:
            #             verbose_logger.debug(f"Setup script stderr for task {task_id}:\n{stderr.decode()}")
                    
            #         if process.returncode != 0:
            #             error_msg = stderr.decode() if stderr else "Unknown error"
            #             verbose_logger.debug(f"Error running setup script for task {task_id}: {error_msg}")
            #             return {task_id: f"ERROR: Setup script failed: {error_msg}"}

            # Create runner script
            script = self._create_runner_script(
                agent_function=agent_function,
                task_id=task_id,
                run_id=run_id
            )
                        
            script_path = temp_dir / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)

            # Build command
            run_agent_cmd = ["python", str(script_path)]
            if self.conda_env:
                # Install weave in conda environment
                verbose_logger.debug(f"Running agent for task {task_id}")
                if get_trace_mode() in ("weave", "both"):
                    process = await asyncio.create_subprocess_exec(
                        *["conda", "run", "-n", self.conda_env, "pip", "install", "weave==0.51.41", "gql<4"],
                        cwd=str(temp_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )

                    stdout, stderr = await process.communicate()
                
                # new command to run the agent
                run_agent_cmd = ["conda", "run", "-n", self.conda_env] + run_agent_cmd
                
            # Run agent
            verbose_logger.debug(f"Running agent for task {task_id}")
            process = await asyncio.create_subprocess_exec(
                *run_agent_cmd,
                cwd=str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            
            # Log agent output
            if stdout:
                verbose_logger.debug(f"Agent stdout for task {task_id}:\n{stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Agent stderr for task {task_id}:\n{stderr.decode()}")
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                verbose_logger.debug(f"Error running task {task_id}: {error_msg}")
                return {task_id: f"ERROR: {error_msg}"}

            # Load results
            try:
                with open(temp_dir / "output.json") as f:
                    return json.load(f)
            except FileNotFoundError:
                error_msg = "ERROR: No output file generated"
                verbose_logger.debug(f"{error_msg} for task {task_id}")
                return {task_id: error_msg}

        except Exception as e:
            error_msg = f"Error processing task {task_id}: {e}"
            verbose_logger.debug(error_msg)
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            # Cleanup
            if str(temp_dir) in self.temp_dirs:
                self.temp_dirs.remove(str(temp_dir))
            try:
                # copy directory to log_dir
                shutil.copytree(temp_dir, os.path.join(self.log_dir, task_id), dirs_exist_ok=True)
                # Remove temp directory
                shutil.rmtree(temp_dir)
            except Exception as e:
                error_msg = f"Warning: Failed to cleanup {temp_dir}: {e}"
                verbose_logger.debug(error_msg)

    def _create_runner_script(self, agent_function: str, task_id: str, run_id: str) -> str:
        """
        Create the Python script that will run the agent
        """
        module_name, function_name = agent_function.rsplit(".", 1)
        
        return f'''
import os
import json
import importlib.util
import traceback
import contextlib
import time
import uuid
from datetime import datetime

TRACE_MODE = (os.getenv("HAL_TRACE_MODE") or "").strip().lower()
if not TRACE_MODE:
    TRACE_MODE = "weave" if os.getenv("WANDB_API_KEY") else "local"
LOCAL_TRACE_ENABLED = TRACE_MODE in ("local", "both")
WEAVE_TRACE_ENABLED = TRACE_MODE in ("weave", "both")

if LOCAL_TRACE_ENABLED and TRACE_MODE == "local":
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WEAVE_DISABLED", "true")

TRACE_TASK_ID = "{task_id}"
TRACE_RUN_ID = "{run_id}"
TRACE_PROJECT = "{run_id}"
TRACE_TASK_ROOT = os.getcwd()

def _iso(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).isoformat(timespec="milliseconds") + "Z"

def _trace_path() -> str:
    override = os.getenv("HAL_LOCAL_TRACE_PATH")
    if override:
        return override
    return os.path.join(TRACE_TASK_ROOT or ".", "local_trace.jsonl")

def _write_trace(entry: dict) -> None:
    try:
        path = _trace_path()
        trace_dir = os.path.dirname(path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\\n")
    except Exception:
        pass

def _should_trace_url(url: str) -> bool:
    text = str(url)
    return any(token in text for token in (
        "/chat/completions",
        "/completions",
        "/responses",
        "/v1/messages",
        "/embeddings",
    ))

def _parse_json_bytes(payload) -> dict | None:
    if not payload:
        return None
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except Exception:
            return None
    if not isinstance(payload, str):
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None

def _build_entry(payload: dict | None, output: dict | None, started_at: float, ended_at: float) -> dict:
    return dict(
        id=str(uuid.uuid4()),
        op_name="local.llm_call",
        display_name="local.llm_call",
        trace_id=TRACE_RUN_ID,
        parent_id=None,
        started_at=_iso(started_at),
        ended_at=_iso(ended_at),
        inputs=payload if payload is not None else dict(),
        output=output if output is not None else dict(),
        attributes=dict(
            weave_task_id=TRACE_TASK_ID,
            trace_source="local",
            weave_project=TRACE_PROJECT,
        ),
        weave_task_id=TRACE_TASK_ID,
    )

def _maybe_log_httpx(request, response, started_at: float) -> None:
    if not LOCAL_TRACE_ENABLED:
        return
    if request.method.upper() != "POST":
        return
    if not _should_trace_url(request.url):
        return
    payload = _parse_json_bytes(request.content)
    if isinstance(payload, dict) and payload.get("stream"):
        return
    output = None
    try:
        response.read()
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            output = json.loads(response.content.decode("utf-8"))
    except Exception:
        output = None
    _write_trace(_build_entry(payload, output, started_at, time.time()))

async def _maybe_log_httpx_async(request, response, started_at: float) -> None:
    if not LOCAL_TRACE_ENABLED:
        return
    if request.method.upper() != "POST":
        return
    if not _should_trace_url(request.url):
        return
    payload = _parse_json_bytes(request.content)
    if isinstance(payload, dict) and payload.get("stream"):
        return
    output = None
    try:
        await response.aread()
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            output = json.loads(response.content.decode("utf-8"))
    except Exception:
        output = None
    _write_trace(_build_entry(payload, output, started_at, time.time()))

def _maybe_log_requests(request, response, started_at: float) -> None:
    if not LOCAL_TRACE_ENABLED:
        return
    if request.method.upper() != "POST":
        return
    if not _should_trace_url(request.url):
        return
    payload = _parse_json_bytes(request.body)
    if isinstance(payload, dict) and payload.get("stream"):
        return
    output = None
    try:
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            output = json.loads(response.content.decode("utf-8"))
    except Exception:
        output = None
    _write_trace(_build_entry(payload, output, started_at, time.time()))

try:
    import httpx  # type: ignore

    _httpx_send = httpx.Client.send
    def _trace_send(self, request, *args, **kwargs):
        started_at = time.time()
        response = _httpx_send(self, request, *args, **kwargs)
        _maybe_log_httpx(request, response, started_at)
        return response

    _httpx_async_send = httpx.AsyncClient.send
    async def _trace_async_send(self, request, *args, **kwargs):
        started_at = time.time()
        response = await _httpx_async_send(self, request, *args, **kwargs)
        await _maybe_log_httpx_async(request, response, started_at)
        return response

    httpx.Client.send = _trace_send  # type: ignore[assignment]
    httpx.AsyncClient.send = _trace_async_send  # type: ignore[assignment]
except Exception:
    pass

try:
    import requests  # type: ignore

    _requests_send = requests.Session.send
    def _trace_requests_send(self, request, *args, **kwargs):
        started_at = time.time()
        response = _requests_send(self, request, *args, **kwargs)
        _maybe_log_requests(request, response, started_at)
        return response

    requests.Session.send = _trace_requests_send  # type: ignore[assignment]
except Exception:
    pass

try:
    import weave  # type: ignore
except Exception:  # pragma: no cover - weave optional
    class _WeaveStub:
        def init(self, *_args, **_kwargs):
            return None

        def attributes(self, *_args, **_kwargs):
            return contextlib.nullcontext()

        def finish(self):
            pass
    weave = _WeaveStub()

try:
    # Initialize weave
    if WEAVE_TRACE_ENABLED:
        weave.init("{run_id}")
    
    # Load input data
    with open("input.json", "r") as f:
        input_data = json.load(f)
    
    # Load agent arguments
    with open("agent_args.json", "r") as f:
        agent_args = json.load(f)

    # Import agent module
    spec = importlib.util.spec_from_file_location(
        "{module_name}",
        os.path.join(os.getcwd(), "{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, "{function_name}")
    
    # Wrap the agent function in a Weave op so the run produces at least one call record per task.
    if WEAVE_TRACE_ENABLED:
        @weave.op()
        def _agent_op(_input_data, _agent_args):
            return agent_fn(_input_data, **_agent_args)

        with weave.attributes({{"weave_task_id": "{task_id}"}}):
            result = _agent_op(input_data, agent_args)
    else:
        result = agent_fn(input_data, **agent_args)
    
    # Save output
    with open("output.json", "w") as f:
        json.dump(result, f)

except Exception as e:
    print(f"Error running agent: {{e}}")
    print(traceback.format_exc())
    with open("error.log", "w") as f:
        f.write(f"ERROR: {{str(e)}}\\n")
        f.write(traceback.format_exc())
    raise
''' 
