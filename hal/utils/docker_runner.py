import os
import json
import asyncio
import shutil
import uuid
import tempfile
import subprocess
import logging
import docker
from docker.types import DeviceRequest
import time
import hashlib
import shlex
import urllib.parse
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
from ..benchmarks.base_benchmark import BaseBenchmark
from rich.progress import Progress, TaskID
from dotenv import dotenv_values
# Get logger for verbose output
verbose_logger = logging.getLogger('agent_eval.verbose')

# Define the docker image name
DOCKER_IMAGE_NAME = "hal-agent-runner:latest"
# Build-time choices for the prepared image. Bump TEMPLATE_VERSION when changing the recipe.
# IMPORTANT: appworld requires Python >= 3.11 as of Jan 2026
AGENT_ENV_PYTHON_VERSION = "3.11"  # Hardcoded to ensure Python 3.11 is always used
AGENT_ENV_TEMPLATE_VERSION = "7"  # v7: use mamba instead of conda for faster env creation

# Task hang detection and automatic retry configuration
# When a task exceeds TASK_HANG_TIMEOUT_SECONDS, it will be forcibly terminated and retried automatically.
# This prevents a single hung task from blocking the entire evaluation indefinitely.
TASK_HANG_TIMEOUT_SECONDS = int(os.getenv("HAL_TASK_HANG_TIMEOUT", "1800"))  # 30 minutes default
TASK_MAX_RETRIES = int(os.getenv("HAL_TASK_MAX_RETRIES", "3"))  # Max retry attempts for hung tasks (4 total attempts)

@dataclass
class _ContainerLease:
    container_id: str
    host_root: Path

class DockerRunner:
    """Handles running agents in Docker containers for isolation"""
    
    def __init__(self, log_dir: str, max_concurrent: int = 1, benchmark: Optional[BaseBenchmark] = None):
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self._active_containers: List[str] = []
        self.benchmark = benchmark
        self.verbose = False
        self._prepared_image_by_requirements: dict[str, str] = {}
        self._toolchain_preflight_lock = asyncio.Lock()
        self._toolchain_preflight_by_image: dict[str, bool] = {}
        self._azure_preflight_lock = asyncio.Lock()
        self._azure_preflight_by_image: dict[str, bool] = {}
        reuse_setting = os.getenv("HAL_DOCKER_REUSE_CONTAINERS")
        if reuse_setting is None or reuse_setting.strip() == "":
            self._reuse_containers = True
        else:
            self._reuse_containers = reuse_setting.strip().lower() in ("1", "true", "yes")
        task_venv_setting = os.getenv("HAL_DOCKER_TASK_VENV")
        if task_venv_setting is None or task_venv_setting.strip() == "":
            self._task_venv = True
        else:
            self._task_venv = task_venv_setting.strip().lower() in ("1", "true", "yes")
        worker_setting = os.getenv("HAL_DOCKER_WORKER_MODE")
        if worker_setting is None or worker_setting.strip() == "":
            self._worker_mode = True
        else:
            self._worker_mode = worker_setting.strip().lower() in ("1", "true", "yes")
        metrics_setting = os.getenv("HAL_DOCKER_WORKER_METRICS")
        if metrics_setting is None or metrics_setting.strip() == "":
            self._worker_metrics = False
        else:
            self._worker_metrics = metrics_setting.strip().lower() in ("1", "true", "yes")
        verbose_setting = os.getenv("HAL_DOCKER_WORKER_VERBOSE")
        if verbose_setting is None or verbose_setting.strip() == "":
            self._worker_verbose = True
        else:
            self._worker_verbose = verbose_setting.strip().lower() in ("1", "true", "yes")
        status_setting = os.getenv("HAL_DOCKER_WORKER_STATUS_LOGS")
        if status_setting is None or status_setting.strip() == "":
            self._worker_status_logs = False
        else:
            self._worker_status_logs = status_setting.strip().lower() in ("1", "true", "yes")
        self._container_pool: Optional[asyncio.Queue[_ContainerLease]] = None
        self._pool_lock = asyncio.Lock()
        self._pool_spec: Optional[Dict[str, Any]] = None
        self._pool_leases: list[_ContainerLease] = []
        # Optional Docker network mode override (e.g. "host") for environments where the
        # default bridge network cannot reach external endpoints.
        self.network_mode: str | None = (
            os.getenv("HAL_DOCKER_NETWORK_MODE")
            or os.getenv("HAL_DOCKER_NETWORK")  # backwards-compat alias
            or None
        )
        # Debug: Print network mode at init
        print(f"[hal][docker] DockerRunner init: network_mode={self.network_mode}")
        self.dotenv_path: str = os.getenv("HAL_DOTENV_PATH") or ".env"
        
        # Initialize Docker client.
        # NOTE: docker-py does not honor Docker "contexts". If you're using Docker Desktop on macOS
        # and the SDK can't connect, set `HAL_DOCKER_HOST=unix:///var/run/docker.sock` (or your
        # desktop socket) to point the SDK at the right daemon.
        docker_host = os.getenv("HAL_DOCKER_HOST") or os.getenv("DOCKER_HOST")
        # Increase timeout for high-concurrency scenarios (default is 60s)
        docker_timeout = int(os.getenv("HAL_DOCKER_TIMEOUT", "600"))
        if docker_host:
            self.docker_client = docker.DockerClient(base_url=docker_host, timeout=docker_timeout)
        else:
            self.docker_client = docker.from_env(timeout=docker_timeout)
            
        # Initialize thread pool for blocking Docker operations
        # We need enough threads to handle max_concurrent tasks potentially blocking on Docker I/O
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(max_concurrent + 10, 50))
        
        # Check if Docker is available
        self._check_docker_available()
        
        # Ensure the Docker image exists
        self._ensure_docker_image()

    def shutdown(self):
        """Shutdown the thread pool executor"""
        if self._executor:
            self._executor.shutdown(wait=True)

    def _slugify_label(self, value: str, fallback: str) -> str:
        value = (value or "").strip()
        if not value:
            return fallback
        # Keep it W&B/Weave friendly: letters, digits, underscore, dash, dot.
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
        return value or fallback

    def _infer_prefix_from_run_id(self, run_id: str) -> str:
        prefix = (run_id or "").split("_", 1)[0]
        return self._slugify_label(prefix, "run")

    def _infer_benchmark_from_run_id(self, run_id: str) -> str:
        if self.benchmark is not None and getattr(self.benchmark, "benchmark_name", None):
            return self._slugify_label(str(self.benchmark.benchmark_name), "benchmark")
        # Fallback: try to find a token that looks like a benchmark name (e.g. corebench_hard).
        match = re.search(r"\b([a-zA-Z]+bench(?:_[a-zA-Z0-9]+)?)\b", run_id or "")
        if match:
            return self._slugify_label(match.group(1), "benchmark")
        return "benchmark"

    def _weave_project_for_run(self, run_id: str) -> str:
        # Allow explicit override via environment variable
        override = os.environ.get("HAL_WEAVE_PROJECT")
        if override:
            return override
        prefix = self._infer_prefix_from_run_id(run_id)
        benchmark = self._infer_benchmark_from_run_id(run_id)
        return f"{prefix}_{benchmark}"

    def _requirements_hash(self, requirements_path: str) -> str:
        """
        Hash requirements + the current base runner image ID.
        This ensures that when `hal-agent-runner:latest` is rebuilt (e.g. to add R/pandoc/TeX),
        we automatically build a fresh prepared image instead of reusing a stale one.
        """
        req_bytes = Path(requirements_path).read_bytes()
        try:
            base_image_id = self.docker_client.images.get(DOCKER_IMAGE_NAME).id.encode("utf-8")
        except Exception:
            base_image_id = b"unknown-base-image"
        recipe = (
            f"template={AGENT_ENV_TEMPLATE_VERSION}\n"
            f"python={AGENT_ENV_PYTHON_VERSION}\n"
            "weave=0.51.41\n"
            "wandb=0.17.9\n"
        ).encode("utf-8")
        return hashlib.sha256(req_bytes + b"\n" + base_image_id + b"\n" + recipe).hexdigest()[:16]

    def _pool_size(self) -> int:
        override = (os.getenv("HAL_DOCKER_POOL_SIZE") or "").strip()
        if override:
            try:
                value = int(override)
                if value > 0:
                    return value
            except ValueError:
                pass
        return max(1, self.max_concurrent)

    def _pool_root_dir(self) -> Path:
        override = os.getenv("HAL_DOCKER_POOL_ROOT")
        if override:
            return Path(override).expanduser()
        file_path = Path(__file__).resolve()
        for parent in file_path.parents:
            if parent.name == "hal-harness":
                return parent.parent / ".tmp"
        return Path.cwd() / ".tmp"

    def _worker_script_path(self, host_root: Path) -> Path:
        return host_root / "hal_worker.py"

    def _queue_dir(self, host_root: Path) -> Path:
        return host_root / "queue"

    def _staging_root(self, host_root: Path) -> Path:
        return host_root / "staging"

    def _create_worker_script(self) -> str:
        return textwrap.dedent(
            '''
            import json
            import os
            import shlex
            import shutil
            import subprocess
            import time
            import traceback
            from pathlib import Path

            QUEUE_DIR = os.getenv("HAL_TASK_QUEUE_DIR", "/workspace/queue")
            AGENT_PYTHON = os.getenv("HAL_AGENT_PYTHON", "/opt/conda/envs/agent_env/bin/python")
            VENV_PATH = os.getenv("HAL_TASK_VENV_PATH", "/workspace/hal_task_venv")
            USE_VENV = (os.getenv("HAL_DOCKER_TASK_VENV") or "1").strip().lower() in ("1", "true", "yes")

            def _tail(text: str, limit: int = 4000) -> str:
                if not text:
                    return ""
                if len(text) <= limit:
                    return text
                return text[-limit:]

            def _run(cmd, cwd=None, env=None, timeout=None):
                return subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )

            def _write_status(path: Path, payload: dict) -> None:
                try:
                    payload["updated_at"] = time.time()
                    path.write_text(json.dumps(payload))
                except Exception:
                    pass

            def _run_process(cmd, cwd, env, timeout, stdout_path, stderr_path, status_path, phase):
                start = time.time()
                with open(stdout_path, "a") as out, open(stderr_path, "a") as err:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=cwd,
                        env=env,
                        stdout=out,
                        stderr=err,
                        text=True,
                    )
                    last_status = 0.0
                    while True:
                        ret = proc.poll()
                        now = time.time()
                        if timeout and now - start > timeout:
                            proc.kill()
                            raise subprocess.TimeoutExpired(cmd, timeout)
                        if now - last_status >= 2.0:
                            _write_status(
                                status_path,
                                {
                                    "phase": phase,
                                    "elapsed": now - start,
                                },
                            )
                            last_status = now
                        if ret is not None:
                            return ret
                        time.sleep(0.5)

            queue_path = Path(QUEUE_DIR)
            queue_path.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(queue_path, 0o777)
            except Exception:
                pass
            try:
                ready_path = queue_path / "worker.ready"
                ready_path.write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
                os.chmod(ready_path, 0o666)
            except Exception:
                pass

            while True:
                jobs = sorted(queue_path.glob("*.json"))
                if not jobs:
                    time.sleep(0.2)
                    continue

                job_path = jobs[0]
                job_id = job_path.stem
                done_path = queue_path / f"{job_id}.done"
                payload = {
                    "job_id": job_id,
                    "status": "error",
                    "result": None,
                    "error": None,
                }
                started_at = time.time()

                try:
                    job = json.loads(job_path.read_text())
                    task_id = job.get("task_id")
                    task_dir = job.get("task_dir")
                    env_override = job.get("env_override") or {}
                    timeout = int(job.get("timeout") or 7200)
                    queued_at = job.get("queued_at")
                    if task_id:
                        payload["task_id"] = task_id
                    if task_dir:
                        payload["task_dir"] = task_dir
                    if not task_dir:
                        raise RuntimeError("task_dir missing from job payload")

                    task_dir_path = Path(task_dir)
                    task_dir_path.mkdir(parents=True, exist_ok=True)

                    env = os.environ.copy()
                    env.update({str(k): str(v) for k, v in env_override.items()})
                    env["HAL_TASK_DIR"] = task_dir
                    env["HAL_TASK_ENVIRONMENT_DIR"] = str(task_dir_path / "environment")
                    status_path = task_dir_path / "worker_status.json"
                    _write_status(status_path, {"phase": "starting"})

                    venv_started = time.time()
                    if USE_VENV:
                        try:
                            shutil.rmtree(VENV_PATH, ignore_errors=True)
                        except Exception:
                            pass
                        _write_status(status_path, {"phase": "venv"})
                        venv_stdout = task_dir_path / "worker_venv_stdout.log"
                        venv_stderr = task_dir_path / "worker_venv_stderr.log"
                        venv_result = _run_process(
                            [AGENT_PYTHON, "-m", "venv", "--system-site-packages", VENV_PATH],
                            cwd=None,
                            env=env,
                            timeout=600,
                            stdout_path=venv_stdout,
                            stderr_path=venv_stderr,
                            status_path=status_path,
                            phase="venv",
                        )
                        if venv_result != 0:
                            raise RuntimeError("venv creation failed")
                        env["VIRTUAL_ENV"] = VENV_PATH
                        env["PATH"] = f"{VENV_PATH}/bin:" + env.get("PATH", "")
                        env["PYTHONNOUSERSITE"] = "1"
                        python_cmd = str(Path(VENV_PATH) / "bin" / "python")
                    else:
                        python_cmd = AGENT_PYTHON
                    venv_finished = time.time()

                    setup_started = time.time()
                    setup_script = task_dir_path / "setup_script.sh"
                    if setup_script.exists():
                        task_dir_safe = shlex.quote(task_dir)
                        setup_safe = shlex.quote(str(setup_script))
                        _write_status(status_path, {"phase": "setup"})
                        setup_stdout = task_dir_path / "worker_setup_stdout.log"
                        setup_stderr = task_dir_path / "worker_setup_stderr.log"
                        setup_result = _run_process(
                            ["bash", "-lc", f"cd {task_dir_safe} && bash {setup_safe}"],
                            cwd=None,
                            env=env,
                            timeout=timeout,
                            stdout_path=setup_stdout,
                            stderr_path=setup_stderr,
                            status_path=status_path,
                            phase="setup",
                        )
                        if setup_result != 0:
                            raise RuntimeError("setup_script failed")
                    setup_finished = time.time()

                    run_started = time.time()
                    _write_status(status_path, {"phase": "run"})
                    run_stdout = task_dir_path / "worker_stdout.log"
                    run_stderr = task_dir_path / "worker_stderr.log"
                    run_result = _run_process(
                        [python_cmd, "run_agent.py"],
                        cwd=task_dir,
                        env=env,
                        timeout=timeout,
                        stdout_path=run_stdout,
                        stderr_path=run_stderr,
                        status_path=status_path,
                        phase="run",
                    )
                    run_finished = time.time()

                    output_path = task_dir_path / "output.json"
                    if output_path.exists():
                        try:
                            payload["result"] = json.loads(output_path.read_text())
                        except Exception:
                            payload["result"] = None

                    if run_result != 0:
                        raise RuntimeError(
                            "agent run failed; see worker_stdout.log/worker_stderr.log"
                        )

                    payload["status"] = "ok"
                    payload["metrics"] = {
                        "queued_at": queued_at,
                        "started_at": started_at,
                        "finished_at": time.time(),
                        "venv_seconds": venv_finished - venv_started,
                        "setup_seconds": setup_finished - setup_started,
                        "run_seconds": run_finished - run_started,
                    }

                except subprocess.TimeoutExpired:
                    payload["status"] = "timeout"
                    payload["error"] = "timeout"
                    payload["metrics"] = {
                        "queued_at": job.get("queued_at") if isinstance(job, dict) else None,
                        "started_at": started_at,
                        "finished_at": time.time(),
                    }
                except Exception as exc:
                    payload["status"] = "error"
                    payload["error"] = str(exc)
                    payload["traceback"] = _tail(traceback.format_exc())
                    payload["metrics"] = {
                        "queued_at": job.get("queued_at") if isinstance(job, dict) else None,
                        "started_at": started_at,
                        "finished_at": time.time(),
                    }

                try:
                    done_path.write_text(json.dumps(payload))
                    os.chmod(done_path, 0o666)
                except Exception:
                    pass

                try:
                    task_dir = payload.get("task_dir")
                    if task_dir:
                        subprocess.run(["chmod", "-R", "a+rwx", task_dir], check=False)
                except Exception:
                    pass

                try:
                    job_path.unlink()
                except Exception:
                    pass
            '''
        ).lstrip()

    def _collect_env_vars(self, env_override: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env_vars: Dict[str, str] = {k: v for k, v in os.environ.items() if isinstance(v, str)}
        try:
            env_vars.update(
                {
                    k: v
                    for k, v in dotenv_values(self.dotenv_path).items()
                    if v is not None
                }
            )
        except Exception:
            pass
        if env_override:
            env_vars.update({str(k): str(v) for k, v in env_override.items() if v is not None})
        env_vars.setdefault("BASH_ENV", "/opt/conda/etc/profile.d/conda.sh")
        env_vars = self._sanitize_forwarded_env(env_vars)
        env_vars["PATH"] = self._default_container_path()
        # Avoid noisy locale warnings when host sets en_US.UTF-8 but the container lacks that locale.
        for key in ("LANG", "LC_ALL"):
            val = (env_vars.get(key) or "").strip()
            if not val:
                env_vars[key] = "C.UTF-8"
            elif val.lower() in ("en_us.utf-8", "en_us.utf8"):
                env_vars[key] = "C.UTF-8"
        self._maybe_warn_about_openai_base_url(env_vars)
        return env_vars

    def _resolve_extra_hosts(self) -> Optional[Dict[str, str]]:
        extra_hosts: Optional[Dict[str, str]] = {"host.docker.internal": "host-gateway"}
        disable_host_gateway = (os.getenv("HAL_DOCKER_DISABLE_HOST_GATEWAY") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if disable_host_gateway:
            return None
        return extra_hosts

    def _configure_azure_mount(self, env_vars: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        volumes: Dict[str, Dict[str, str]] = {}
        azure_dir = os.path.expanduser("~/.azure")
        msal_cache = os.path.join(azure_dir, "msal_token_cache.json")
        azure_creds_exist = os.path.isdir(azure_dir) and os.path.isfile(msal_cache)

        # Auto-enable direct Azure if credentials exist (unless explicitly disabled)
        use_direct_azure = env_vars.get("USE_DIRECT_AZURE", "").lower()
        if use_direct_azure == "false":
            verbose_logger.debug("Direct Azure disabled (USE_DIRECT_AZURE=false)")
        elif use_direct_azure == "true" and not azure_creds_exist:
            raise RuntimeError(
                f"USE_DIRECT_AZURE=true but MSAL cache not found at {msal_cache}. "
                "Aborting to avoid non-Azure fallback."
            )
        elif azure_creds_exist:
            env_vars["USE_DIRECT_AZURE"] = "true"
            verbose_logger.debug(f"Auto-enabled direct Azure (MSAL cache found at {msal_cache})")

        if os.path.isdir(azure_dir) and env_vars.get("USE_DIRECT_AZURE", "").lower() == "true":
            volumes[azure_dir] = {"bind": "/root/.azure", "mode": "rw"}
            env_vars["HOME"] = "/root"
            verbose_logger.debug(f"Mounting Azure credentials from {azure_dir} (rw for MSAL token refresh)")
            for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_BASE_URL", "LITELLM_BASE_URL"):
                env_vars.pop(key, None)
            verbose_logger.debug("Azure MSAL mode enabled - tokens will auto-refresh for long-running tasks")

        return volumes

    def _seed_shared_workspace(self, host_root: Path, agent_dir_path: Path) -> None:
        agent_root = host_root / "hal-harness" / "agents"
        temp_agent_dir = agent_root / agent_dir_path.name
        temp_agent_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(agent_dir_path, temp_agent_dir, dirs_exist_ok=True)

        sibling_open_deep = agent_dir_path.parent / "open_deep_research"
        if sibling_open_deep.exists():
            shutil.copytree(
                sibling_open_deep,
                host_root / "hal-harness" / "open_deep_research",
                dirs_exist_ok=True,
            )

        sibling_shared = agent_dir_path.parent / "shared"
        if sibling_shared.exists():
            shutil.copytree(
                sibling_shared,
                agent_root / "shared",
                dirs_exist_ok=True,
            )

        model_quirks_file = agent_dir_path.parent / "model_quirks.py"
        if model_quirks_file.exists():
            shutil.copy2(model_quirks_file, agent_root / "model_quirks.py")

    async def _start_pool_container(self, host_root: Path) -> _ContainerLease:
        if not self._pool_spec:
            raise RuntimeError("Container pool spec missing; cannot start pooled container.")
        image_tag = self._pool_spec["image"]
        env_vars = self._pool_spec["env_vars"]
        volumes_base = self._pool_spec["volumes_base"]
        extra_hosts = self._pool_spec["extra_hosts"]
        tmpfs = self._pool_spec["tmpfs"]

        container_id = f"agentpool--{uuid.uuid4()}"[:32].lower().replace("_", "-")
        volumes = dict(volumes_base)
        volumes[str(host_root)] = {"bind": "/workspace", "mode": "rw"}

        loop = asyncio.get_running_loop()
        container = await loop.run_in_executor(
            self._executor,
            lambda: self.docker_client.containers.run(
                image=image_tag,
                name=container_id,
                detach=True,
                command=["tail", "-f", "/dev/null"],
                environment=env_vars,
                network_mode=self.network_mode,
                extra_hosts=extra_hosts,
                volumes=volumes if volumes else None,
                tmpfs=tmpfs,
                device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            )
        )

        self._active_containers.append(container_id)
        try:
            await self._toolchain_preflight(container, image_tag)
            await self._azure_token_preflight(container, image_tag, env_vars)
            if self._worker_mode:
                worker_path = self._worker_script_path(host_root)
                try:
                    worker_cmd = (
                        "nohup /opt/conda/envs/agent_env/bin/python /workspace/hal_worker.py "
                        ">/workspace/worker.log 2>&1 &"
                    )
                    result = await loop.run_in_executor(
                        self._executor, 
                        lambda: container.exec_run(["bash", "-lc", worker_cmd])
                    )
                    if result.exit_code != 0:
                        out = (result.output or b"").decode(errors="replace")
                        raise RuntimeError(out.strip() or f"worker exec failed (exit {result.exit_code})")
                    ready_path = self._queue_dir(host_root) / "worker.ready"
                    ready_deadline = time.time() + 10
                    while time.time() < ready_deadline:
                        if ready_path.exists():
                            break
                        await asyncio.sleep(0.2)
                    if not ready_path.exists():
                        raise RuntimeError("worker did not signal ready; check /workspace/worker.log in the container")
                except Exception as e:
                    raise RuntimeError(f"Failed to start worker in container {container_id}: {e}") from e
        except Exception:
            try:
                await loop.run_in_executor(self._executor, lambda: container.remove(force=True))
            except Exception:
                pass
            if container_id in self._active_containers:
                self._active_containers.remove(container_id)
            raise
            
        return _ContainerLease(container_id=container_id, host_root=host_root)

    async def _ensure_container_pool(self, prepared_image: str, agent_dir: str) -> None:
        if self._container_pool is not None:
            return
        async with self._pool_lock:
            if self._container_pool is not None:
                return
            env_vars = self._collect_env_vars()
            env_vars["HAL_HARNESS_ROOT"] = "/workspace/hal-harness"
            env_vars["HAL_TASK_DIR"] = "/workspace/task"
            env_vars["HAL_TASK_ENVIRONMENT_DIR"] = "/workspace/task/environment"
            env_vars["HAL_TASK_QUEUE_DIR"] = "/workspace/queue"
            env_vars["HAL_TASK_VENV_PATH"] = "/workspace/hal_task_venv"
            env_vars["HAL_AGENT_PYTHON"] = "/opt/conda/envs/agent_env/bin/python"
            env_vars["HAL_DOCKER_TASK_VENV"] = "1" if self._task_venv else "0"
            volumes_base = self._configure_azure_mount(env_vars)
            extra_hosts = self._resolve_extra_hosts()
            tmpfs = {"/tmp": "size=1G,mode=1777"}

            self._pool_spec = {
                "image": prepared_image,
                "env_vars": env_vars,
                "volumes_base": volumes_base,
                "extra_hosts": extra_hosts,
                "tmpfs": tmpfs,
            }

            pool = asyncio.Queue()
            self._pool_leases = []
            agent_dir_path = Path(agent_dir).resolve()
            
            async def _setup_and_start_container():
                pool_root = self._pool_root_dir()
                pool_root.mkdir(parents=True, exist_ok=True)
                
                loop = asyncio.get_running_loop()
                
                # Perform blocking file operations in executor
                def _setup_files():
                    host_root = Path(tempfile.mkdtemp(prefix="hal_docker_pool_", dir=pool_root))
                    (host_root / "task").mkdir(parents=True, exist_ok=True)
                    self._queue_dir(host_root).mkdir(parents=True, exist_ok=True)
                    self._staging_root(host_root).mkdir(parents=True, exist_ok=True)
                    worker_path = self._worker_script_path(host_root)
                    if self._worker_mode:
                        worker_path.write_text(self._create_worker_script(), encoding="utf-8")
                    self._seed_shared_workspace(host_root, agent_dir_path)
                    return host_root
                
                host_root = await loop.run_in_executor(self._executor, _setup_files)
                return await self._start_pool_container(host_root)

            # Start containers in parallel with a semaphore to prevent I/O storm
            # Limit concurrent creation to 10 per process (10 * 10 models = 100 global concurrent starts)
            creation_sem = asyncio.Semaphore(10)
            
            async def _throttled_create():
                async with creation_sem:
                    return await _setup_and_start_container()

            tasks = [_throttled_create() for _ in range(self._pool_size())]
            leases = await asyncio.gather(*tasks)

            for lease in leases:
                await pool.put(lease)
                self._pool_leases.append(lease)

            self._container_pool = pool

    async def _acquire_pool_container(self) -> _ContainerLease:
        if self._container_pool is None:
            raise RuntimeError("Container pool not initialized.")
        lease = await self._container_pool.get()
        try:
            container = self.docker_client.containers.get(lease.container_id)
            container.reload()
            if container.status != "running":
                raise docker.errors.NotFound("container not running")
            return lease
        except Exception:
            try:
                self.docker_client.containers.get(lease.container_id).remove(force=True)
            except Exception:
                pass
            try:
                if lease.container_id in self._active_containers:
                    self._active_containers.remove(lease.container_id)
            except Exception:
                pass
            lease = await self._start_pool_container(lease.host_root)
            return lease

    async def _release_pool_container(self, lease: _ContainerLease) -> None:
        if self._container_pool is None:
            return
        await self._container_pool.put(lease)

    async def _teardown_container_pool(self) -> None:
        if self._container_pool is None:
            return
        leases = list(self._pool_leases)
        self._container_pool = None
        self._pool_leases = []
        for lease in leases:
            try:
                container = self.docker_client.containers.get(lease.container_id)
                container.remove(force=True)
            except Exception:
                pass
            try:
                if lease.container_id in self._active_containers:
                    self._active_containers.remove(lease.container_id)
            except Exception:
                pass
            try:
                shutil.rmtree(lease.host_root, ignore_errors=True)
            except Exception:
                pass
    def _ensure_prepared_image(self, requirements_path: str) -> str:
        """
        Build a derived image with a ready-to-run `agent_env` conda env.
        This avoids creating conda envs + pip installing requirements per task container,
        which is prohibitively slow when running many tasks in parallel.
        """
        req_hash = self._requirements_hash(requirements_path)
        cached = self._prepared_image_by_requirements.get(req_hash)
        if cached:
            return cached

        tag = f"hal-agent-runner:agent-env-{req_hash}"
        force_rebuild = (os.getenv("HAL_DOCKER_FORCE_REBUILD") or "").strip().lower() in ("1", "true", "yes")
        try:
            if not force_rebuild:
                self.docker_client.images.get(tag)
                self._prepared_image_by_requirements[req_hash] = tag
                return tag
        except docker.errors.ImageNotFound:
            pass

        # Avoid redundant parallel builds when multiple hal-eval processes start at once.
        lock_file = Path(tempfile.gettempdir()) / "hal_agent_env_build.lock"
        lock_handle = lock_file.open("a", encoding="utf-8")
        try:
            try:
                import fcntl  # type: ignore

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

            # Another process may have built it while we waited.
            try:
                if not force_rebuild:
                    self.docker_client.images.get(tag)
                    self._prepared_image_by_requirements[req_hash] = tag
                    return tag
            except docker.errors.ImageNotFound:
                pass

            build_dir = Path(tempfile.mkdtemp(prefix="hal_agent_env_build_"))
            try:
                (build_dir / "requirements.txt").write_text(
                    Path(requirements_path).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                (build_dir / "Dockerfile").write_text(
                    "\n".join(
                        [
                            f"FROM {DOCKER_IMAGE_NAME}",
                            "COPY requirements.txt /tmp/requirements.txt",
                            "RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \\",
                            " && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \\",
                            f" && mamba create -y -n agent_env python={AGENT_ENV_PYTHON_VERSION} \\",
                            " && conda run -n agent_env python -m pip install -U pip \\",
                            " && conda run -n agent_env pip install -r /tmp/requirements.txt \\",
                            " && conda run -n agent_env pip install weave==0.51.41 'gql<4' wandb==0.17.9",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                build_args: Dict[str, str] = {}
                for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
                    val = os.getenv(key)
                    if val:
                        build_args[key] = val

                build_stream = self.docker_client.api.build(
                    path=str(build_dir),
                    tag=tag,
                    decode=True,
                    buildargs=build_args or None,
                )
                log_lines: list[str] = []
                error_message: Optional[str] = None
                try:
                    for chunk in build_stream:
                        if not isinstance(chunk, dict):
                            continue
                        if "stream" in chunk:
                            line = str(chunk["stream"]).rstrip()
                            log_lines.append(line)
                            verbose_logger.debug(line)
                        if "errorDetail" in chunk:
                            detail = chunk.get("errorDetail") or {}
                            if isinstance(detail, dict):
                                error_message = str(detail.get("message") or detail).strip()
                            else:
                                error_message = str(detail).strip()
                            log_lines.append(error_message)
                        if "error" in chunk:
                            error_message = str(chunk["error"]).strip()
                            log_lines.append(error_message)
                except Exception as e:
                    # If docker-py errors while streaming, log what we have and raise.
                    tail = "\n".join(log_lines[-200:])
                    verbose_logger.debug("Prepared image build stream failed for %s\n%s", tag, tail)
                    raise RuntimeError(
                        "Prepared image build failed while streaming logs; "
                        "set HAL_DOCKER_PREPARED_IMAGE_KEEP=1 and inspect the build dir."
                    ) from e

                if error_message:
                    tail = "\n".join(log_lines[-200:])
                    verbose_logger.debug("Prepared image build failed for %s\n%s", tag, tail)
                    raise RuntimeError(
                        f"Prepared image build failed: {error_message}. "
                        "Common causes: proxy settings not forwarded, or dependency wheels "
                        "incompatible with the selected python version."
                    )

                # Verify the image exists after build.
                self.docker_client.images.get(tag)
                self._prepared_image_by_requirements[req_hash] = tag
                return tag
            finally:
                keep_build_dir = (os.getenv("HAL_DOCKER_PREPARED_IMAGE_KEEP") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
                if not keep_build_dir:
                    shutil.rmtree(build_dir, ignore_errors=True)
        finally:
            try:
                lock_handle.close()
            except Exception:
                pass
        
    def _check_docker_available(self) -> None:
        """Check if Docker is available on the system"""
        try:
            version = self.docker_client.version()
            verbose_logger.debug(f"Docker is available: {version.get('Version', 'unknown version')}")
        except docker.errors.DockerException as e:
            docker_host = os.getenv("HAL_DOCKER_HOST") or os.getenv("DOCKER_HOST") or "unset"
            error_message = (
                "Docker is not available to the Python SDK (docker-py), so the Docker runner cannot start containers.\n"
                f"- docker host (HAL_DOCKER_HOST/DOCKER_HOST): {docker_host}\n"
                "- Common fixes:\n"
                "  - Ensure Docker Desktop / dockerd is running.\n"
                "  - If you can run `docker ps` but the SDK fails, set `HAL_DOCKER_HOST` to the daemon endpoint "
                "(docker-py does not honor Docker contexts).\n"
                "  - On Linux, ensure your user is in the `docker` group or run with appropriate privileges.\n"
            )
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e

    async def _toolchain_preflight(self, container: "docker.models.containers.Container", image_tag: str) -> None:
        """
        Verify that baseline reproducibility tools (Rscript/pandoc/TeX) exist in the container image.
        Runs at most once per prepared image tag per process.
        """
        # 1. Fast Path: Check without lock
        cached = self._toolchain_preflight_by_image.get(image_tag)
        if cached is True:
            return
        if cached is False:
            raise RuntimeError(
                "Container image failed toolchain preflight previously (R/pandoc/TeX missing). "
                "Rebuild the runner image and force rebuild prepared images."
            )

        # 2. Slow Path: Acquire lock
        async with self._toolchain_preflight_lock:
            # 3. Double Check: Did someone else finish while we waited?
            cached = self._toolchain_preflight_by_image.get(image_tag)
            if cached is True:
                return
            if cached is False:
                raise RuntimeError(
                    "Container image failed toolchain preflight previously (R/pandoc/TeX missing). "
                    "Rebuild the runner image and force rebuild prepared images."
                )

            # Keep it POSIX-sh compatible and short: just enough to disambiguate missing toolchain.
            cmd = [
                "bash",
                "-lc",
                "set -e; "
                "echo '[hal][toolchain] PATH='\"$PATH\"; "
                "echo '[hal][toolchain] PATH contains /opt/conda/bin:'; "
                "(echo \"$PATH\" | tr ':' '\\n' | grep -qx '/opt/conda/bin' && echo yes) "
                "|| (echo no; exit 2); "
                "echo '[hal][toolchain] which Rscript:'; command -v Rscript; Rscript --version; "
                "echo '[hal][toolchain] which pandoc:'; command -v pandoc; pandoc --version | head -n 2; "
                "echo '[hal][toolchain] which pdflatex:'; (command -v pdflatex && pdflatex --version | head -n 1) || true; "
                "echo '[hal][toolchain] which latexmk:'; (command -v latexmk && latexmk -v | head -n 1) || true; ",
            ]
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: container.exec_run(cmd)
                )
                out = (result.output or b"").decode(errors="replace")
                verbose_logger.debug("[hal][toolchain] preflight for image=%s exit=%s\n%s", image_tag, result.exit_code, out)
                if result.exit_code != 0:
                    raise RuntimeError(out.strip() or f"toolchain preflight failed (exit {result.exit_code})")
                self._toolchain_preflight_by_image[image_tag] = True
            except Exception as e:
                self._toolchain_preflight_by_image[image_tag] = False
                raise RuntimeError(
                    "Missing baseline R/pandoc/TeX toolchain in the container image; this causes widespread "
                    "CoreBench 'environmental barrier' failures.\n"
                    "- Rebuild base image: `cd hal-harness && docker build -t hal-agent-runner:latest -f hal/utils/docker/Dockerfile .`\n"
                    "- Then force rebuild prepared images once: `HAL_DOCKER_FORCE_REBUILD=1`\n"
                    f"- Preflight error: {e}"
                ) from e

    async def _azure_token_preflight(
        self,
        container: "docker.models.containers.Container",
        image_tag: str,
        env_vars: Dict[str, str],
    ) -> None:
        """
        Verify MSAL refresh token works inside the container when direct Azure is enabled.
        Runs at most once per prepared image tag per process.
        """
        # DISABLED: This preflight check causes significant delays and hangs when running 
        # many parallel containers due to network contention and MSAL cache locking.
        # We will let the agent fail lazily if auth is broken.
        return

        if env_vars.get("USE_DIRECT_AZURE", "").lower() != "true":
            return

        async with self._azure_preflight_lock:
            cached = self._azure_preflight_by_image.get(image_tag)
            if cached is True:
                return
            if cached is False:
                raise RuntimeError(
                    "Azure token preflight failed previously for this image; "
                    "fix MSAL cache/auth before retrying."
                )

            scope = env_vars.get("TRAPI_SCOPE", "api://trapi/.default")
            cmd = [
                "bash",
                "-lc",
                "set -e; "
                "echo '[hal][azure] verifying MSAL refresh token...'; "
                "conda run -n agent_env python - <<'PY'\n"
                "import msal, os\n"
                "cache_path = os.path.expanduser('~/.azure/msal_token_cache.json')\n"
                "if not os.path.exists(cache_path):\n"
                "    raise SystemExit(f'MSAL cache not found at {cache_path}')\n"
                "cache = msal.SerializableTokenCache()\n"
                "cache.deserialize(open(cache_path).read())\n"
                "app = msal.PublicClientApplication(\n"
                "    '04b07795-8ddb-461a-bbee-02f9e1bf7b46',\n"
                "    authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47',\n"
                "    token_cache=cache,\n"
                ")\n"
                "accounts = app.get_accounts()\n"
                "if not accounts:\n"
                "    raise SystemExit('No accounts found in MSAL cache')\n"
                f"scope = '{scope}'\n"
                "last_error = None\n"
                "ok = False\n"
                "for idx, account in enumerate(accounts):\n"
                "    username = account.get('username', 'unknown')\n"
                "    result = app.acquire_token_silent([scope], account=account, force_refresh=True)\n"
                "    if result and 'access_token' in result:\n"
                "        if cache.has_state_changed:\n"
                "            with open(cache_path, 'w') as f:\n"
                "                f.write(cache.serialize())\n"
                "        print(f'[hal][azure] refresh ok account {idx}: {username}')\n"
                "        ok = True\n"
                "        break\n"
                "    if result:\n"
                "        last_error = result.get('error_description', 'unknown')\n"
                "    else:\n"
                "        last_error = f'No token for account {username}'\n"
                "if not ok:\n"
                "    raise SystemExit(f'Token refresh failed for all accounts. Last error: {last_error}')\n"
                "print('[hal][azure] refresh token usable')\n"
                "PY\n",
            ]
            try:
                result = container.exec_run(cmd)
                out = (result.output or b"").decode(errors="replace")
                verbose_logger.debug("[hal][azure] preflight for image=%s exit=%s\n%s", image_tag, result.exit_code, out)
                if result.exit_code != 0:
                    raise RuntimeError(out.strip() or f"azure preflight failed (exit {result.exit_code})")
                self._azure_preflight_by_image[image_tag] = True
            except Exception as e:
                self._azure_preflight_by_image[image_tag] = False
                raise RuntimeError(
                    "Azure MSAL refresh token preflight failed inside the container. "
                    "Ensure ~/.azure/msal_token_cache.json is mounted RW and contains a TRAPI refresh token."
                    f"\n- Preflight error: {e}"
                ) from e

    def _sanitize_forwarded_env(self, env: Dict[str, str]) -> Dict[str, str]:
        keep_tls_overrides = (os.getenv("HAL_DOCKER_KEEP_TLS_ENV") or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not keep_tls_overrides:
            # Avoid forwarding host TLS overrides that usually reference host-only paths and
            # break requests/wandb/weave inside containers.
            for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
                env.pop(key, None)
        # Avoid clobbering the image PATH (contains /opt/conda/bin) with the host PATH.
        env.pop("PATH", None)
        # Host python env vars often point to host-only paths and can break container python.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        # Host conda env vars point to host-only paths and can break conda inside the container.
        for key in list(env.keys()):
            upper = key.upper()
            if upper.startswith("CONDA"):
                env.pop(key, None)
            elif upper.startswith("_CE_") or upper.startswith("_CONDA"):
                env.pop(key, None)
            elif upper.startswith("MAMBA") or upper.startswith("MICROMAMBA"):
                env.pop(key, None)
        return env

    def _default_container_path(self) -> str:
        """
        Provide a stable PATH inside containers.

        Why:
        - Some shells (via BASH_ENV/conda.sh) prepend /opt/conda/condabin but do not guarantee /opt/conda/bin
          is present unless an environment is activated.
        - Many CoreBench repos call tools like `Rscript`, `pandoc`, `jupyter`, etc. via PATH.
        """
        parts = [
            "/opt/conda/envs/agent_env/bin",
            "/opt/conda/bin",
            "/opt/conda/condabin",
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ]
        # Keep order but avoid duplicates
        seen = set()
        out: list[str] = []
        for p in parts:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return ":".join(out)

    def _maybe_warn_about_openai_base_url(self, env: Dict[str, str]) -> None:
        base = (
            env.get("OPENAI_BASE_URL")
            or env.get("OPENAI_API_BASE")
            or env.get("OPENAI_API_BASE_URL")
            or env.get("AZURE_OPENAI_ENDPOINT")
            or env.get("LITELLM_BASE_URL")
        )
        if not base:
            return
        try:
            parsed = urllib.parse.urlparse(base)
            host = parsed.hostname or base
        except Exception:
            host = base
        if host in ("localhost", "127.0.0.1", "0.0.0.0") and self.network_mode != "host":
            verbose_logger.debug(
                "Detected OpenAI base URL points at localhost (%s). Inside Docker this refers to the container; "
                "use `HAL_DOCKER_NETWORK_MODE=host` (Linux) or set the base URL to `http://host.docker.internal:<port>`.",
                base,
            )
    
    def _ensure_docker_image(self) -> None:
        """Ensure the Docker image exists, building it if necessary"""
        try:
            # Check if the image already exists
            try:
                self.docker_client.images.get(DOCKER_IMAGE_NAME)
                verbose_logger.debug(f"Docker image {DOCKER_IMAGE_NAME} already exists")
            except docker.errors.ImageNotFound:
                verbose_logger.debug(f"Docker image {DOCKER_IMAGE_NAME} not found, building it...")
                
                # Get the Dockerfile path - it should be in the same directory as this file
                dockerfile_dir = os.path.join(os.path.dirname(__file__), "docker")
                dockerfile_path = os.path.join(dockerfile_dir, "Dockerfile")
                
                if not os.path.exists(dockerfile_path):
                    raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
                
                # Build the Docker image
                verbose_logger.debug(f"Building Docker image from {dockerfile_path}")
                
                _, build_logs = self.docker_client.images.build(
                    path=dockerfile_dir,
                    dockerfile=os.path.basename(dockerfile_path),
                    tag=DOCKER_IMAGE_NAME
                )
                
                for log in build_logs:
                    if 'stream' in log:
                        verbose_logger.debug(log['stream'].strip())
                
                verbose_logger.debug(f"Docker image built successfully")
                
        except docker.errors.DockerException as e:
            error_message = f"Failed to build Docker image: {str(e)}"
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            error_message = f"Error ensuring Docker image: {str(e)}"
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e
        
    async def run_agent(self,
                       dataset: Dict[str, Any],
                       agent_function: str,
                       agent_dir: str,
                       agent_args: Dict[str, Any],
                       run_id: str,
                       benchmark: Optional[BaseBenchmark] = None,
                       progress: Optional[Progress] = None,
                       task: Optional[TaskID] = None,
                       timeout: int = 7200,
                       task_env_overrides: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Run agent on all tasks with concurrency control
        """
        try:
            self.benchmark = benchmark
            # Get run directory from benchmark if provided
            run_dir = benchmark.get_run_dir(run_id) if benchmark else f"results/{run_id}"
            os.makedirs(run_dir, exist_ok=True)
            submissions_file = os.path.join(run_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")

            # Fail fast once per run if the prepared image is missing baseline tooling (Rscript/pandoc/TeX).
            # This avoids burning time and tokens across many tasks that would all error with the same barrier.
            try:
                agent_dir_path = Path(agent_dir).resolve()
                requirements_path = str(agent_dir_path / "requirements.txt")
                prepared_image = self._ensure_prepared_image(requirements_path)
                if self._reuse_containers:
                    await self._ensure_container_pool(prepared_image, agent_dir)
                elif self._toolchain_preflight_by_image.get(prepared_image) is not True:
                    preflight_container_id = f"agentpreflight--{uuid.uuid4()}"[:32].lower().replace("_", "-")
                    preflight_env = {
                        "PATH": self._default_container_path(),
                        "BASH_ENV": "/opt/conda/etc/profile.d/conda.sh",
                    }
                    preflight_container = self.docker_client.containers.run(
                        image=prepared_image,
                        name=preflight_container_id,
                        detach=True,
                        command=["tail", "-f", "/dev/null"],
                        environment=preflight_env,
                        network_mode=self.network_mode,
                    )
                    try:
                        await self._toolchain_preflight(preflight_container, prepared_image)
                    finally:
                        try:
                            preflight_container.remove(force=True)
                        except Exception:
                            pass
            except Exception:
                # Let the underlying toolchain preflight error message propagate.
                raise
            
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
                    task=task,
                    env_override=(task_env_overrides or {}).get(task_id)
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
            if self._container_pool is not None:
                await self._teardown_container_pool()
            # Cleanup any remaining containers
            for container_id in self._active_containers.copy():  # Use copy to avoid mutation during iteration
                try:
                    container = self.docker_client.containers.get(container_id)
                    container.stop(timeout=5)
                    container.remove(force=True)
                    verbose_logger.debug(f"Cleaned up container {container_id}")
                except docker.errors.NotFound:
                    pass  # Container already removed
                except (docker.errors.APIError, Exception) as e:
                    verbose_logger.debug(f"Warning: Failed to cleanup container {container_id}: {e}")

    async def _process_task(self,
                          task_id: str,
                          input_data: Any,
                          agent_function: str,
                          agent_dir: str,
                          agent_args: Dict[str, Any],
                          run_id: str,
                          submissions_file: str,
                          progress: Optional[Progress] = None,
                          task: Optional[TaskID] = None,
                          env_override: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Process a single task with semaphore control, timeout enforcement, and automatic retry"""
        async with self._semaphore:
            verbose_logger.debug(f"Starting task {task_id} (active tasks: {self.max_concurrent - self._semaphore._value})")

            # Extract timeout from agent_args if present
            timeout = 28800
            if agent_args and 'timeout' in agent_args:
                try:
                    timeout = int(agent_args['timeout'])
                except (ValueError, TypeError):
                    verbose_logger.debug(f"Invalid timeout value in agent_args: {agent_args['timeout']}, using default")

            # Use hang detection timeout - the maximum time before we consider a task hung
            # This is separate from the task's internal timeout to catch cases where the task
            # is stuck (e.g., network hang, Docker daemon issue) and not making progress
            hang_timeout = max(timeout + 300, TASK_HANG_TIMEOUT_SECONDS)  # At least task timeout + 5 min buffer

            result = None
            last_error = None

            # Retry loop for hung tasks
            for attempt in range(TASK_MAX_RETRIES + 1):
                try:
                    if attempt > 0:
                        verbose_logger.warning(
                            f"[RETRY] Task {task_id} - Attempt {attempt + 1}/{TASK_MAX_RETRIES + 1} "
                            f"after previous hang/timeout"
                        )
                        print(f"[hal][WARNING] Retrying task {task_id} (attempt {attempt + 1}/{TASK_MAX_RETRIES + 1})")

                    # Wrap the task execution with asyncio.wait_for to enforce a hard timeout
                    # This catches cases where the subprocess hangs indefinitely
                    result = await asyncio.wait_for(
                        self._run_single_task(
                            task_id=task_id,
                            input_data=input_data,
                            agent_function=agent_function,
                            agent_dir=agent_dir,
                            agent_args=agent_args,
                            run_id=run_id,
                            timeout=timeout,
                            env_override=env_override
                        ),
                        timeout=hang_timeout
                    )

                    # Check if result indicates a timeout/error that might be retryable
                    if result and task_id in result:
                        result_value = result[task_id]
                        if isinstance(result_value, str) and result_value.startswith("TIMEOUT"):
                            # Internal timeout - might be worth retrying
                            if attempt < TASK_MAX_RETRIES:
                                verbose_logger.warning(
                                    f"[HANG] Task {task_id} timed out internally, will retry"
                                )
                                last_error = result_value
                                await asyncio.sleep(5)  # Brief pause before retry
                                continue

                    # Success or non-retryable result
                    break

                except asyncio.TimeoutError:
                    # Task hung beyond the hang detection timeout
                    verbose_logger.warning(
                        f"[HANG DETECTED] Task {task_id} exceeded hang timeout of {hang_timeout}s "
                        f"(attempt {attempt + 1}/{TASK_MAX_RETRIES + 1})"
                    )
                    print(
                        f"[hal][WARNING] Task {task_id} HUNG for >{hang_timeout}s - "
                        f"{'retrying' if attempt < TASK_MAX_RETRIES else 'giving up'}"
                    )
                    last_error = f"HANG_TIMEOUT after {hang_timeout}s"

                    if attempt < TASK_MAX_RETRIES:
                        # Clean up any lingering containers for this task before retry
                        await self._cleanup_hung_task_containers(task_id)
                        await asyncio.sleep(10)  # Longer pause for hung tasks
                        continue
                    else:
                        # Max retries exceeded
                        result = {task_id: f"ERROR: Task hung after {TASK_MAX_RETRIES + 1} attempts. Last error: {last_error}"}

                except Exception as e:
                    verbose_logger.error(f"[ERROR] Task {task_id} failed with exception: {e}")
                    if attempt < TASK_MAX_RETRIES:
                        last_error = str(e)
                        await asyncio.sleep(5)
                        continue
                    result = {task_id: f"ERROR: {str(e)}"}

            # Write result to submissions file
            if result:
                async with self._file_lock:
                    with open(submissions_file, "a") as f:
                        json.dump(result, f)
                        f.write("\n")

            # Update progress after task completion
            if progress and task is not None:
                progress.update(task, advance=1)

            verbose_logger.debug(f"Completed task {task_id}")
            return result

    async def _cleanup_hung_task_containers(self, task_id: str) -> None:
        """Clean up any containers that might be associated with a hung task"""
        try:
            # Find and kill containers that might be related to this task
            loop = asyncio.get_running_loop()
            containers = await loop.run_in_executor(
                self._executor,
                lambda: self.docker_client.containers.list(all=True)
            )
            for container in containers:
                container_name = container.name or ""
                # Check if container name contains task-related identifiers
                if "agentrun" in container_name.lower():
                    try:
                        # Check if container is stuck
                        status = container.status
                        if status in ("running", "created"):
                            verbose_logger.debug(f"Cleaning up potentially hung container: {container_name}")
                            container.kill()
                            container.remove(force=True)
                    except Exception as e:
                        verbose_logger.debug(f"Failed to cleanup container {container_name}: {e}")
        except Exception as e:
            verbose_logger.debug(f"Error during hung task cleanup: {e}")

    async def _run_single_task(self,
                             task_id: str,
                             input_data: Any,
                             agent_function: str,
                             agent_dir: str,
                             agent_args: Dict[str, Any],
                             run_id: str,
                             timeout: int = 7200,
                             env_override: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Process a single task in a Docker container with timeout"""
        if self._reuse_containers:
            return await self._run_single_task_reuse(
                task_id=task_id,
                input_data=input_data,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                run_id=run_id,
                timeout=timeout,
                env_override=env_override,
            )
        # Create temporary directory for mounting into container
        temp_dir = Path(tempfile.mkdtemp())
        container_id = f"agentrun--{uuid.uuid4()}"[:32].lower().replace("_", "-")
        
        try:
            # Mirror a minimal hal-harness agents layout so agent code that expects sibling agents
            # (e.g. open_deep_research) keeps working inside the container.
            agent_dir_path = Path(agent_dir).resolve()
            agent_root = temp_dir / "hal-harness" / "agents"
            temp_agent_dir = agent_root / agent_dir_path.name
            temp_agent_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(agent_dir_path, temp_agent_dir, dirs_exist_ok=True)

            sibling_open_deep = agent_dir_path.parent / "open_deep_research"
            if sibling_open_deep.exists():
                shutil.copytree(
                    sibling_open_deep,
                    temp_dir / "hal-harness" / "open_deep_research",
                    dirs_exist_ok=True,
                )

            # Copy shared module for Azure/TRAPI support
            sibling_shared = agent_dir_path.parent / "shared"
            if sibling_shared.exists():
                shutil.copytree(
                    sibling_shared,
                    agent_root / "shared",
                    dirs_exist_ok=True,
                )

            # Copy model_quirks.py for model parameter handling
            model_quirks_file = agent_dir_path.parent / "model_quirks.py"
            if model_quirks_file.exists():
                shutil.copy2(model_quirks_file, agent_root / "model_quirks.py")

            # Write input and args files
            with open(temp_dir / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(temp_dir / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            # Copy task-specific files if they exist in input_data
            if isinstance(input_data, dict) and 'files' in input_data:
                for dest_path, src_path in input_data['files'].items():
                    dest_path = dest_path.replace('/root/', '').lstrip('/')
                    dest_full_path = temp_dir / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as e:
                        verbose_logger.debug(f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}")

            # Create runner script
            weave_project = self._weave_project_for_run(run_id)
            trace_filename = f"{run_id}_UPLOAD.json"
            script = self._create_runner_script(
                agent_function=agent_function,
                task_id=task_id,
                run_id=run_id,
                agent_name=agent_dir_path.name,
                weave_project=weave_project,
                weave_op_name=task_id,
                trace_filename=trace_filename,
            )
                        
            script_path = temp_dir / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)
            # Keep a copy in the agent folder as well for debugging, but execute from /workspace.
            shutil.copy2(script_path, temp_agent_dir / "run_agent.py")
            
            # create container from image and mount temp dir
            requirements_path = str(agent_dir_path / "requirements.txt")
            prepared_image = self._ensure_prepared_image(requirements_path)

            # Pass through host env + .env + per-task overrides so Weave/W&B can upload traces.
            env_vars = self._collect_env_vars(env_override)

            # Provide a stable way for containers to reach host-local services (e.g. a local OpenAI/LiteLLM proxy).
            # On Linux, Docker requires an explicit host-gateway mapping.
            extra_hosts = self._resolve_extra_hosts()

            # Mount Azure CLI credentials if available (for direct Azure access)
            volumes = self._configure_azure_mount(env_vars)

            # Add tmpfs mount for /tmp to fix Azure CLI issues in container
            tmpfs = {"/tmp": "size=1G,mode=1777"}

            try:
                loop = asyncio.get_running_loop()
                container = await loop.run_in_executor(
                    self._executor,
                    lambda: self.docker_client.containers.run(
                        image=prepared_image,
                        name=container_id,
                        detach=True,
                        command=["tail", "-f", "/dev/null"],  # Keep container running
                        environment=env_vars,
                        network_mode=self.network_mode,
                        extra_hosts=extra_hosts,
                        volumes=volumes if volumes else None,
                        tmpfs=tmpfs,
                        device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                    )
                )
            except docker.errors.APIError as e:
                # Some Docker engines don't support host-gateway; retry without it.
                if extra_hosts and "host-gateway" in str(e).lower():
                    loop = asyncio.get_running_loop()
                    container = await loop.run_in_executor(
                        self._executor,
                        lambda: self.docker_client.containers.run(
                            image=prepared_image,
                            name=container_id,
                            detach=True,
                            command=["tail", "-f", "/dev/null"],
                            environment=env_vars,
                            network_mode=self.network_mode,
                            volumes=volumes if volumes else None,
                            tmpfs=tmpfs,
                            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
                        )
                    )
                else:
                    raise
            
            # Add container to active list
            self._active_containers.append(container_id)

            # Fail fast if the prepared image is missing baseline tooling (Rscript/pandoc/TeX).
            # This avoids running agent logic that will inevitably error with "Rscript: not found".
            await self._toolchain_preflight(container, prepared_image)
            await self._azure_token_preflight(container, prepared_image, env_vars)
            
            # Using asyncio subprocess instead of subprocess.run
            # copy all the contents of temp dir into container
            proc = await asyncio.create_subprocess_exec(
                "docker", "cp", f"{temp_dir}/.", f"{container_id}:/workspace",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            # Make both `./results` (from /workspace/environment) and `../results` resolve to a writable location.
            # Many CoreBench tasks mention ../results but also instruct agents not to write to parent dirs; a stable
            # symlink reduces spurious permission/path failures.
            try:
                container.exec_run(
                    [
                        "bash",
                        "-lc",
                        "set -e; "
                        "mkdir -p /workspace/results /workspace/environment; "
                        "chmod -R a+rwx /workspace/results || true; "
                        "if [ ! -e /workspace/environment/results ]; then ln -s /workspace/results /workspace/environment/results; fi; ",
                    ]
                )
            except Exception:
                pass
            
            # run setup script if it exists
            if self.benchmark and self.benchmark.setup_script:
                print(f"Running setup script: {self.benchmark.setup_script}")
                setup_script_src = Path(self.benchmark.setup_script)
                if setup_script_src.exists():
                    # copy setup script to container
                    proc = await asyncio.create_subprocess_exec(
                        "docker", "cp", f"{setup_script_src}", f"{container_id}:/workspace/setup_script.sh",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    if self.verbose:
                        if stdout:
                            verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                    if stderr:
                        verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")
                    
                    # run setup script and wait for it to complete
                    proc = await asyncio.create_subprocess_exec(
                        "docker", "exec", container_id, "bash", "/workspace/setup_script.sh",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    if self.verbose:    
                        if stdout:
                            verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                    if stderr:
                        verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")   
                        
            # Run the script and capture output with timeout handling
            start_time = time.time() 
        
            # Container env already includes host + .env; only prefix any per-task overrides.
            env_prefix = ""
            if env_override:
                parts: List[str] = []
                for k, v in env_override.items():
                    if v is None:
                        continue
                    parts.append(f"{shlex.quote(str(k))}={shlex.quote(str(v))}")
                if parts:
                    env_prefix = " ".join(parts) + " "

            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                # Run from /workspace so the task repo appears as "the current directory",
                # matching corebench instructions like "the repository cloned to your current directory".
                f"{env_prefix}cd /workspace && /opt/conda/bin/conda run -n agent_env python run_agent.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def _stream_log(stream, label):
                if not stream:
                    return
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    verbose_logger.debug(f"Container {container_id}: {line.decode(errors='replace').rstrip()}")

            await asyncio.gather(
                _stream_log(proc.stdout, "stdout"),
                _stream_log(proc.stderr, "stderr"),
                proc.wait()
            )        
            
            # Poll for output.json with timeout
            result = None
            while time.time() - start_time < timeout:
                # Check if output.json exists
                check_result = container.exec_run(["test", "-f", "/workspace/output.json"])
                if check_result.exit_code == 0:
                    # copy files from container back to host
                    proc = await asyncio.create_subprocess_exec(
                        "docker", "cp", f"{container_id}:/workspace/.", f"{temp_dir}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()                    
                    if stdout:
                        verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                    if stderr:
                        verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")
                    
                    # Load and return results
                    with open(temp_dir / "output.json") as f:
                        result = json.load(f)
                        break
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            if result is None:
                verbose_logger.debug(f"Task {task_id} timed out after {timeout} seconds")
                return {task_id: f"TIMEOUT after {timeout} seconds"}
            
            return result

        except Exception as e:
            error_msg = f"Error processing task {task_id}: {e}"
            verbose_logger.debug(error_msg)
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            # Cleanup - CRITICAL: Must succeed to avoid filling up /tmp
            try:
                # Copy directory to log_dir if specified
                if self.log_dir:
                    task_log_dir = os.path.join(self.log_dir, task_id)
                    # Use ignore_dangling_symlinks and a custom ignore function to handle
                    # missing files/directories (common in ColBench and other benchmarks)
                    def ignore_missing(directory, files):
                        """Ignore files that don't exist (broken symlinks, etc.)"""
                        ignored = []
                        for f in files:
                            path = os.path.join(directory, f)
                            if not os.path.exists(path):
                                ignored.append(f)
                        return ignored
                    try:
                        shutil.copytree(temp_dir, task_log_dir, dirs_exist_ok=True,
                                       ignore=ignore_missing, ignore_dangling_symlinks=True)
                    except shutil.Error as copy_errors:
                        # Log but don't fail - some files may not copy but that's OK
                        verbose_logger.debug(f"Some files not copied for task {task_id}: {copy_errors}")

                # Remove temp directory with retry logic
                # IMPORTANT: Do NOT use ignore_errors=True as it silently fails and fills up /tmp
                cleanup_success = False
                for attempt in range(3):
                    try:
                        if temp_dir.exists():
                            shutil.rmtree(temp_dir)
                            cleanup_success = True
                            verbose_logger.debug(f"Cleaned up temp directory for task {task_id}")
                            break
                        else:
                            cleanup_success = True
                            break
                    except PermissionError as pe:
                        verbose_logger.debug(f"Cleanup attempt {attempt+1}/3 failed for task {task_id}: {pe}")
                        # Try to change permissions and retry
                        try:
                            import stat
                            for root, dirs, files in os.walk(temp_dir):
                                for d in dirs:
                                    os.chmod(os.path.join(root, d), stat.S_IRWXU)
                                for f in files:
                                    os.chmod(os.path.join(root, f), stat.S_IRWXU)
                        except Exception:
                            pass
                        time.sleep(0.5)
                    except Exception as e:
                        verbose_logger.debug(f"Cleanup attempt {attempt+1}/3 failed for task {task_id}: {e}")
                        time.sleep(0.5)

                if not cleanup_success and temp_dir.exists():
                    # Last resort: try subprocess rm -rf
                    try:
                        subprocess.run(["rm", "-rf", str(temp_dir)], timeout=60, check=False)
                        if not temp_dir.exists():
                            cleanup_success = True
                            verbose_logger.debug(f"Cleaned up temp directory via rm -rf for task {task_id}")
                    except Exception as e:
                        verbose_logger.debug(f"rm -rf cleanup failed for task {task_id}: {e}")

                if not cleanup_success and temp_dir.exists():
                    verbose_logger.warning(f"CRITICAL: Failed to cleanup temp directory {temp_dir} for task {task_id} - disk space may fill up!")

                # Remove container
                try:
                    container = self.docker_client.containers.get(container_id)
                    container.remove(force=True)
                    # Remove from active containers list
                    if container_id in self._active_containers:
                        self._active_containers.remove(container_id)
                except Exception:
                    pass  # Container may already be removed

            except Exception as e:
                error_msg = f"Warning: Failed to cleanup for task {task_id}: {e}"
                verbose_logger.debug(error_msg)
                # Even if overall cleanup fails, try to remove temp_dir as last resort
                try:
                    if temp_dir.exists():
                        subprocess.run(["rm", "-rf", str(temp_dir)], timeout=60, check=False)
                except Exception:
                    pass

    async def _run_single_task_reuse(self,
                             task_id: str,
                             input_data: Any,
                             agent_function: str,
                             agent_dir: str,
                             agent_args: Dict[str, Any],
                             run_id: str,
                             timeout: int = 18000,
                             env_override: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Process a single task using a pooled Docker container"""
        if self._worker_mode:
            return await self._run_single_task_worker(
                task_id=task_id,
                input_data=input_data,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                run_id=run_id,
                timeout=timeout,
                env_override=env_override,
            )
        lease = await self._acquire_pool_container()
        container_id = lease.container_id
        host_root = lease.host_root
        task_root = host_root / "task"
        result = None

        try:
            task_root.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "bash",
                "-lc",
                "rm -rf /workspace/task/*",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            agent_dir_path = Path(agent_dir).resolve()
            venv_path = "/workspace/hal_task_venv"
            venv_enabled = self._task_venv

            with open(task_root / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(task_root / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            if isinstance(input_data, dict) and 'files' in input_data:
                for dest_path, src_path in input_data['files'].items():
                    dest_path = dest_path.replace('/root/', '').lstrip('/')
                    dest_full_path = task_root / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as e:
                        verbose_logger.debug(f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}")

            try:
                results_dir = task_root / "results"
                env_dir = task_root / "environment"
                results_dir.mkdir(parents=True, exist_ok=True)
                env_dir.mkdir(parents=True, exist_ok=True)
                try:
                    os.chmod(results_dir, 0o777)
                except Exception:
                    pass
                env_results = env_dir / "results"
                if not env_results.exists():
                    env_results.symlink_to(results_dir)
            except Exception:
                pass

            weave_project = self._weave_project_for_run(run_id)
            trace_filename = f"{run_id}_UPLOAD.json"
            script = self._create_runner_script(
                agent_function=agent_function,
                task_id=task_id,
                run_id=run_id,
                agent_name=agent_dir_path.name,
                weave_project=weave_project,
                weave_op_name=task_id,
                trace_filename=trace_filename,
            )
            script_path = task_root / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)

            env_prefix = ""
            parts: List[str] = []
            if env_override:
                for k, v in env_override.items():
                    if v is None:
                        continue
                    parts.append(f"{shlex.quote(str(k))}={shlex.quote(str(v))}")
            if venv_enabled:
                venv_path_value = f"{venv_path}/bin:{self._default_container_path()}"
                parts.append(f"VIRTUAL_ENV={shlex.quote(venv_path)}")
                parts.append(f"PATH={shlex.quote(venv_path_value)}")
                parts.append("PYTHONNOUSERSITE=1")
            if parts:
                env_prefix = " ".join(parts) + " "

            if venv_enabled:
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-lc",
                    f"rm -rf {shlex.quote(venv_path)} && "
                    "/opt/conda/bin/conda run -n agent_env python -m venv --system-site-packages "
                    f"{shlex.quote(venv_path)}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                if stderr:
                    verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            if self.benchmark and self.benchmark.setup_script:
                setup_script_src = Path(self.benchmark.setup_script)
                if setup_script_src.exists():
                    setup_dest = task_root / "setup_script.sh"
                    shutil.copy2(setup_script_src, setup_dest)
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "exec",
                        container_id,
                        "bash",
                        "-c",
                        f"{env_prefix}cd /workspace/task && bash /workspace/task/setup_script.sh",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()
                    if self.verbose and stdout:
                        verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
                    if stderr:
                        verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            start_time = time.time()
            python_cmd = "python"
            if not venv_enabled:
                python_cmd = "/opt/conda/bin/conda run -n agent_env python"
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                f"{env_prefix}cd /workspace/task && {python_cmd} run_agent.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if stdout:
                verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            while time.time() - start_time < timeout:
                output_path = task_root / "output.json"
                if output_path.exists():
                    with open(output_path) as f:
                        result = json.load(f)
                    break
                await asyncio.sleep(5)

            if result is None:
                verbose_logger.debug(f"Task {task_id} timed out after {timeout} seconds")
                return {task_id: f"TIMEOUT after {timeout} seconds"}

            return result

        except Exception as e:
            error_msg = f"Error processing task {task_id}: {e}"
            verbose_logger.debug(error_msg)
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-lc",
                    "chmod -R a+rwx /workspace/task >/dev/null 2>&1 || true",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                if self.log_dir:
                    task_log_dir = os.path.join(self.log_dir, task_id)
                    def ignore_missing(directory, files):
                        ignored = []
                        for f in files:
                            path = os.path.join(directory, f)
                            if not os.path.exists(path):
                                ignored.append(f)
                        return ignored
                    try:
                        shutil.copytree(task_root, task_log_dir, dirs_exist_ok=True,
                                       ignore=ignore_missing, ignore_dangling_symlinks=True)
                    except shutil.Error as copy_errors:
                        verbose_logger.debug(f"Some files not copied for task {task_id}: {copy_errors}")
            except Exception as e:
                verbose_logger.debug(f"Warning: Failed to cleanup pooled task root for task {task_id}: {e}")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-lc",
                    f"rm -rf /workspace/task/* {shlex.quote(venv_path)}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            except Exception:
                pass
            await self._release_pool_container(lease)

    async def _run_single_task_worker(self,
                             task_id: str,
                             input_data: Any,
                             agent_function: str,
                             agent_dir: str,
                             agent_args: Dict[str, Any],
                             run_id: str,
                             timeout: int = 18000,
                             env_override: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Process a single task using the pooled in-container worker."""
        lease = await self._acquire_pool_container()
        host_root = lease.host_root
        queue_dir = self._queue_dir(host_root)
        staging_root = self._staging_root(host_root)
        job_id = f"job-{uuid.uuid4().hex}"
        task_stage = staging_root / job_id
        result = None
        job_path: Optional[Path] = None
        done_path: Optional[Path] = None
        status_path = task_stage / "worker_status.json"
        stdout_path = task_stage / "worker_stdout.log"
        stderr_path = task_stage / "worker_stderr.log"
        last_stdout_offset = 0
        last_stderr_offset = 0
        last_status_phase: Optional[str] = None
        last_status_update = 0.0

        try:
            task_stage.mkdir(parents=True, exist_ok=True)

            agent_dir_path = Path(agent_dir).resolve()

            with open(task_stage / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(task_stage / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            if isinstance(input_data, dict) and 'files' in input_data:
                for dest_path, src_path in input_data['files'].items():
                    dest_path = dest_path.replace('/root/', '').lstrip('/')
                    dest_full_path = task_stage / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as e:
                        verbose_logger.debug(f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}")

            try:
                results_dir = task_stage / "results"
                env_dir = task_stage / "environment"
                results_dir.mkdir(parents=True, exist_ok=True)
                env_dir.mkdir(parents=True, exist_ok=True)
                try:
                    os.chmod(results_dir, 0o777)
                except Exception:
                    pass
                env_results = env_dir / "results"
                if not env_results.exists():
                    env_results.symlink_to(results_dir)
            except Exception:
                pass

            weave_project = self._weave_project_for_run(run_id)
            trace_filename = f"{run_id}_UPLOAD.json"
            script = self._create_runner_script(
                agent_function=agent_function,
                task_id=task_id,
                run_id=run_id,
                agent_name=agent_dir_path.name,
                weave_project=weave_project,
                weave_op_name=task_id,
                trace_filename=trace_filename,
            )
            script_path = task_stage / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)

            if self.benchmark and self.benchmark.setup_script:
                setup_script_src = Path(self.benchmark.setup_script)
                if setup_script_src.exists():
                    setup_dest = task_stage / "setup_script.sh"
                    shutil.copy2(setup_script_src, setup_dest)

            container_task_dir = f"/workspace/staging/{job_id}"
            job_payload = {
                "job_id": job_id,
                "task_id": task_id,
                "task_dir": container_task_dir,
                "timeout": timeout,
                "env_override": {str(k): str(v) for k, v in (env_override or {}).items()},
                "queued_at": time.time(),
            }
            job_path = queue_dir / f"{job_id}.json"
            done_path = queue_dir / f"{job_id}.done"
            job_path.write_text(json.dumps(job_payload))

            start_time = time.time()
            wait_timeout = timeout + 300
            while time.time() - start_time < wait_timeout:
                if self._worker_verbose:
                    if stdout_path.exists():
                        try:
                            with open(stdout_path, "rb") as f:
                                f.seek(last_stdout_offset)
                                data = f.read()
                                last_stdout_offset = f.tell()
                            if data:
                                for line in data.decode(errors="replace").splitlines():
                                    verbose_logger.debug("[task %s] %s", task_id, line)
                        except Exception:
                            pass
                    if stderr_path.exists():
                        try:
                            with open(stderr_path, "rb") as f:
                                f.seek(last_stderr_offset)
                                data = f.read()
                                last_stderr_offset = f.tell()
                            if data:
                                for line in data.decode(errors="replace").splitlines():
                                    verbose_logger.debug("[task %s] %s", task_id, line)
                        except Exception:
                            pass
                    if self._worker_status_logs and status_path.exists():
                        try:
                            status = json.loads(status_path.read_text())
                            phase = status.get("phase")
                            updated_at = float(status.get("updated_at") or 0)
                            if phase and (phase != last_status_phase or updated_at - last_status_update > 10):
                                last_status_phase = phase
                                last_status_update = updated_at
                                elapsed = status.get("elapsed")
                                if isinstance(elapsed, (int, float)):
                                    verbose_logger.debug(
                                        "[hal][worker][%s] phase=%s elapsed=%.1fs",
                                        task_id,
                                        phase,
                                        elapsed,
                                    )
                                else:
                                    verbose_logger.debug("[hal][worker][%s] phase=%s", task_id, phase)
                        except Exception:
                            pass
                if done_path and done_path.exists():
                    payload = json.loads(done_path.read_text())
                    status = payload.get("status")
                    if status == "ok":
                        result = payload.get("result")
                    elif status == "timeout":
                        result = {task_id: f"TIMEOUT after {timeout} seconds"}
                    else:
                        error_detail = payload.get("error") or "unknown error"
                        result = {task_id: f"ERROR: {error_detail}"}
                    if self._worker_metrics:
                        metrics = payload.get("metrics") or {}
                        queued_at = metrics.get("queued_at")
                        started_at = metrics.get("started_at")
                        finished_at = metrics.get("finished_at")
                        queue_delay = None
                        if queued_at and started_at:
                            try:
                                queue_delay = float(started_at) - float(queued_at)
                            except (TypeError, ValueError):
                                queue_delay = None
                        durations = []
                        if queue_delay is not None:
                            durations.append(f"queue={queue_delay:.2f}s")
                        venv_seconds = metrics.get("venv_seconds")
                        if isinstance(venv_seconds, (int, float)):
                            durations.append(f"venv={venv_seconds:.2f}s")
                        setup_seconds = metrics.get("setup_seconds")
                        if isinstance(setup_seconds, (int, float)):
                            durations.append(f"setup={setup_seconds:.2f}s")
                        run_seconds = metrics.get("run_seconds")
                        if isinstance(run_seconds, (int, float)):
                            durations.append(f"run={run_seconds:.2f}s")
                        if started_at and finished_at:
                            try:
                                total = float(finished_at) - float(started_at)
                                durations.append(f"total={total:.2f}s")
                            except (TypeError, ValueError):
                                pass
                        if durations:
                            verbose_logger.info(
                                "[hal][worker] task %s metrics: %s",
                                task_id,
                                ", ".join(durations),
                            )
                    break
                await asyncio.sleep(1)

            if result is None:
                verbose_logger.debug(f"Task {task_id} timed out after {wait_timeout} seconds")
                return {task_id: f"TIMEOUT after {timeout} seconds"}

            return result

        except Exception as e:
            error_msg = f"Error processing task {task_id}: {e}"
            verbose_logger.debug(error_msg)
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            try:
                if self.log_dir:
                    task_log_dir = os.path.join(self.log_dir, task_id)
                    def ignore_missing(directory, files):
                        ignored = []
                        for f in files:
                            path = os.path.join(directory, f)
                            if not os.path.exists(path):
                                ignored.append(f)
                        return ignored
                    try:
                        shutil.copytree(task_stage, task_log_dir, dirs_exist_ok=True,
                                       ignore=ignore_missing, ignore_dangling_symlinks=True)
                    except shutil.Error as copy_errors:
                        verbose_logger.debug(f"Some files not copied for task {task_id}: {copy_errors}")

                try:
                    if task_stage.exists():
                        shutil.rmtree(task_stage)
                except Exception as e:
                    verbose_logger.debug(f"Warning: Failed to cleanup worker staging dir for task {task_id}: {e}")

                try:
                    if job_path.exists():
                        job_path.unlink()
                except Exception:
                    pass
                try:
                    if done_path and done_path.exists():
                        done_path.unlink()
                except Exception:
                    pass
            except Exception as e:
                verbose_logger.debug(f"Warning: Failed to cleanup worker artifacts for task {task_id}: {e}")
            await self._release_pool_container(lease)

    def _create_runner_script(
        self,
        agent_function: str,
        task_id: str,
        run_id: str,
        agent_name: str,
        weave_project: str,
        weave_op_name: str,
        trace_filename: str,
    ) -> str:
        """
        Create the Python script that will run the agent
        """
        module_name, function_name = agent_function.rsplit(".", 1)
        
        return textwrap.dedent(f'''
import os
import json
import importlib.util
import traceback
import socket
import urllib.parse
import re
import sys
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
TRACE_PROJECT = "{weave_project}"
TRACE_TASK_ROOT = ""

def _iso(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).isoformat(timespec="milliseconds") + "Z"

def _trace_path() -> str:
    override = os.getenv("HAL_LOCAL_TRACE_PATH")
    if override:
        return override
    return os.path.join(TRACE_TASK_ROOT or ".", "local_trace.jsonl")

def _write_trace(entry: dict) -> None:
    if not TRACE_TASK_ROOT:
        return
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

def _network_preflight() -> None:
    enabled = (os.getenv("HAL_DOCKER_PREFLIGHT_NETWORK") or "").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return

    def _proxy_host(env_key: str) -> str:
        val = os.getenv(env_key)
        if not val:
            return "unset"
        try:
            parsed = urllib.parse.urlparse(val)
            host = parsed.hostname or "set"
            port = parsed.port
            if port:
                return "set(%s:%s)" % (host, port)
            return "set(%s)" % host
        except Exception:
            return "set"

    base = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_API_BASE_URL")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("LITELLM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    try:
        parsed = urllib.parse.urlparse(base)
        host = parsed.hostname or base
        scheme = parsed.scheme or "https"
        port = parsed.port or (443 if scheme == "https" else 80)
    except Exception:
        host = base
        scheme = "https"
        port = 443

    net_mode = os.getenv("HAL_DOCKER_NETWORK_MODE") or os.getenv("HAL_DOCKER_NETWORK") or "unset"
    print(f"[hal][docker][preflight] network_mode={{net_mode}}")
    print(f"[hal][docker][preflight] OPENAI_BASE={{base}} host={{host}} port={{port}}")
    print(f"[hal][docker][preflight] HTTPS_PROXY={{_proxy_host('HTTPS_PROXY')}} HTTP_PROXY={{_proxy_host('HTTP_PROXY')}}")

    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        print(
            "[hal][docker][preflight] WARNING: base URL points to localhost; inside Docker this refers to the container. "
            "Use `HAL_DOCKER_NETWORK_MODE=host` (Linux) or set base URL to `http://host.docker.internal:<port>`."
        )

    try:
        # TCP connect only (no TLS) to validate basic reachability.
        with socket.create_connection((host, port), timeout=5):
            print("[hal][docker][preflight] TCP connect OK")
    except Exception as e:
        print(f"[hal][docker][preflight] TCP connect FAILED: {{e}}")

try:
    weave = None
    try:
        import weave  # type: ignore
        weave_available = True
    except Exception as weave_import_error:
        weave_available = False
        if WEAVE_TRACE_ENABLED:
            print(f"Weave import failed, disabling tracing: {{weave_import_error}}")

    def _try_autopatch() -> None:
        # Best-effort: enable LLM/network instrumentation when available.
        # Different Weave versions expose different autopatch entry points.
        candidates = [
            ("weave", "autopatch"),
            ("weave.integrations.openai", "autopatch"),
            ("weave.integrations.litellm", "autopatch"),
        ]
        for mod_name, fn_name in candidates:
            try:
                mod = __import__(mod_name, fromlist=["*"])
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    fn()
            except Exception:
                pass

    if weave_available and WEAVE_TRACE_ENABLED:
        try:
            _try_autopatch()
            weave.init("{weave_project}")
            weave_enabled = True
        except Exception as weave_init_error:
            weave_enabled = False
            weave = None
            print(f"Weave init failed, disabling tracing: {{weave_init_error}}")
    else:
        weave_enabled = False
    
    task_root = os.getenv("HAL_TASK_DIR", "/workspace")
    TRACE_TASK_ROOT = task_root
    hal_harness_root = os.getenv("HAL_HARNESS_ROOT", os.path.join(task_root, "hal-harness"))
    agent_root = os.path.join(hal_harness_root, "agents", "{agent_name}")
    for path in (hal_harness_root, agent_root):
        if path and path not in sys.path:
            sys.path.insert(0, path)

    # CoreBench capsules are staged under /workspace/environment (mirrors /root/environment in the
    # original harness). Run the agent from there so relative paths in tasks resolve correctly.
    try:
        env_root = os.getenv("HAL_TASK_ENVIRONMENT_DIR", os.path.join(task_root, "environment"))
        if os.path.isdir(env_root):
            os.chdir(env_root)
    except Exception:
        pass

    # Load input data
    with open(os.path.join(task_root, "input.json"), "r") as f:
        input_data = json.load(f)
    
    # Load agent arguments
    with open(os.path.join(task_root, "agent_args.json"), "r") as f:
        agent_args = json.load(f)

    _network_preflight()

    # Import agent module
    spec = importlib.util.spec_from_file_location(
        "{module_name}",
        os.path.join(agent_root, "{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, "{function_name}")
    
    # Wrap the agent function in a Weave op so the run produces at least one call record per task.
    if weave_enabled:
        def _slugify_label(value: str, fallback: str) -> str:
            value = (value or "").strip()
            if not value:
                return fallback
            value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
            return value or fallback

        op_label = _slugify_label("{weave_op_name}", "task")
        def _agent_op(_input_data, _agent_args):
            return agent_fn(_input_data, **_agent_args)

        # Avoid relying on weave.op(name=...) across versions: set the function name explicitly.
        _agent_op.__name__ = op_label
        _agent_op.__qualname__ = op_label
        _agent_op = weave.op()(_agent_op)

        with weave.attributes(
             {{
                 "weave_task_id": "{task_id}",
                 "task_id": "{task_id}",
                 "run_id": "{run_id}",
                 "trace_file": "{trace_filename}",
                 "weave_project": "{weave_project}",
             }}
         ):
            result = _agent_op(input_data, agent_args)
    else:
        result = agent_fn(input_data, **agent_args)
    
    # Save output
    with open(os.path.join(task_root, "output.json"), "w") as f:
        json.dump(result, f)

except Exception as e:
    print(f"Error running agent: {{e}}")
    print(traceback.format_exc())
    with open("error.log", "w") as f:
        f.write(f"ERROR: {{str(e)}}\\n")
        f.write(traceback.format_exc())
    raise
finally:
    if 'weave' in globals() and weave is not None and weave_enabled:
        try:
            weave.finish()
        except Exception:
            pass
''').lstrip()
