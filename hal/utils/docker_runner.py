import os
import json
import asyncio
import shutil
import uuid
import tempfile
import subprocess
import logging
import docker
import time
import hashlib
import shlex
import urllib.parse
import re
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
AGENT_ENV_PYTHON_VERSION = os.getenv("HAL_AGENT_ENV_PYTHON_VERSION") or "3.10"
AGENT_ENV_TEMPLATE_VERSION = "2"

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
        # Optional Docker network mode override (e.g. "host") for environments where the
        # default bridge network cannot reach external endpoints.
        self.network_mode: str | None = (
            os.getenv("HAL_DOCKER_NETWORK_MODE")
            or os.getenv("HAL_DOCKER_NETWORK")  # backwards-compat alias
            or None
        )
        self.dotenv_path: str = os.getenv("HAL_DOTENV_PATH") or ".env"
        
        # Initialize Docker client.
        # NOTE: docker-py does not honor Docker "contexts". If you're using Docker Desktop on macOS
        # and the SDK can't connect, set `HAL_DOCKER_HOST=unix:///var/run/docker.sock` (or your
        # desktop socket) to point the SDK at the right daemon.
        docker_host = os.getenv("HAL_DOCKER_HOST") or os.getenv("DOCKER_HOST")
        self.docker_client = docker.DockerClient(base_url=docker_host) if docker_host else docker.from_env()
        
        # Check if Docker is available
        self._check_docker_available()
        
        # Ensure the Docker image exists
        self._ensure_docker_image()

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
                            f" && conda create -y -n agent_env python={AGENT_ENV_PYTHON_VERSION} \\",
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
        async with self._toolchain_preflight_lock:
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
                result = container.exec_run(cmd)
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
                if self._toolchain_preflight_by_image.get(prepared_image) is not True:
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
            # Cleanup any remaining containers
            for container_id in self._active_containers:
                try:
                    container = self.docker_client.containers.get(container_id)
                    # container.stop()
                    # container.remove()
                except (docker.errors.NotFound, docker.errors.APIError) as e:
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
        """Process a single task with semaphore control"""
        async with self._semaphore:
            verbose_logger.debug(f"Starting task {task_id} (active tasks: {self.max_concurrent - self._semaphore._value})")
            result = await self._run_single_task(
                task_id=task_id,
                input_data=input_data,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                run_id=run_id,
                env_override=env_override
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
            
            verbose_logger.debug(f"Completed task {task_id}")
            return result

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
            # Make `conda activate ...` work in non-interactive bash sessions started by tools.
            env_vars.setdefault("BASH_ENV", "/opt/conda/etc/profile.d/conda.sh")
            env_vars = self._sanitize_forwarded_env(env_vars)
            # Ensure common toolchains (R/pandoc/LaTeX/jupyter) and the prepared `agent_env` are discoverable via PATH.
            # This avoids widespread "Rscript: not found" environmental barrier failures caused by PATH mismatches.
            env_vars["PATH"] = self._default_container_path()
            self._maybe_warn_about_openai_base_url(env_vars)

            # Provide a stable way for containers to reach host-local services (e.g. a local OpenAI/LiteLLM proxy).
            # On Linux, Docker requires an explicit host-gateway mapping.
            extra_hosts: Optional[Dict[str, str]] = {"host.docker.internal": "host-gateway"}
            disable_host_gateway = (os.getenv("HAL_DOCKER_DISABLE_HOST_GATEWAY") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if disable_host_gateway:
                extra_hosts = None

            try:
                container = self.docker_client.containers.run(
                    image=prepared_image,
                    name=container_id,
                    detach=True,
                    command=["tail", "-f", "/dev/null"],  # Keep container running
                    environment=env_vars,
                    network_mode=self.network_mode,
                    extra_hosts=extra_hosts,
                )
            except docker.errors.APIError as e:
                # Some Docker engines don't support host-gateway; retry without it.
                if extra_hosts and "host-gateway" in str(e).lower():
                    container = self.docker_client.containers.run(
                        image=prepared_image,
                        name=container_id,
                        detach=True,
                        command=["tail", "-f", "/dev/null"],
                        environment=env_vars,
                        network_mode=self.network_mode,
                    )
                else:
                    raise
            
            # Add container to active list
            self._active_containers.append(container_id)

            # Fail fast if the prepared image is missing baseline tooling (Rscript/pandoc/TeX).
            # This avoids running agent logic that will inevitably error with "Rscript: not found".
            await self._toolchain_preflight(container, prepared_image)
            
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
            stdout, stderr = await proc.communicate()
            if stdout:
                verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")        
            
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
            # Cleanup
            try:
                # Copy directory to log_dir if specified
                if self.log_dir:
                    task_log_dir = os.path.join(self.log_dir, task_id)
                    shutil.copytree(temp_dir, task_log_dir, dirs_exist_ok=True)
                
                # Remove temp directory
                shutil.rmtree(temp_dir)
                
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
        
        return f'''
import os
import json
import importlib.util
import traceback
import socket
import urllib.parse
import re
import sys

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

    if weave_available:
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
    
    # Make hal-harness + agent modules importable regardless of current working directory.
    hal_harness_root = "/workspace/hal-harness"
    agent_root = os.path.join(hal_harness_root, "agents", "{agent_name}")
    for path in (hal_harness_root, agent_root):
        if path and path not in sys.path:
            sys.path.insert(0, path)

    # CoreBench capsules are staged under /workspace/environment (mirrors /root/environment in the
    # original harness). Run the agent from there so relative paths in tasks resolve correctly.
    try:
        if os.path.isdir("/workspace/environment"):
            os.chdir("/workspace/environment")
    except Exception:
        pass

    # Load input data
    with open("/workspace/input.json", "r") as f:
        input_data = json.load(f)
    
    # Load agent arguments
    with open("/workspace/agent_args.json", "r") as f:
        agent_args = json.load(f)

    _network_preflight()

    # Import agent module
    spec = importlib.util.spec_from_file_location(
        "{module_name}",
        os.path.join("/workspace", "hal-harness", "agents", "{agent_name}", "{module_name}.py")
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
    with open("/workspace/output.json", "w") as f:
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
''' 
