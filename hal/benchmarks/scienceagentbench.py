from typing import Dict, Any, Optional
from .base_benchmark import BaseBenchmark

from datasets import load_dataset
import os
import argparse
import json
import sys
import time
from pathlib import Path

# get the ABSOLUTE path of the directory where this file is located
# Using absolute paths ensures file copying works regardless of current working directory
this_file_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_path = os.path.join(this_file_dir, 'scienceagentbench', 'ScienceAgentBench_modified', 'benchmark')

submodule_path = os.path.join(this_file_dir, "scienceagentbench", "ScienceAgentBench_modified")
sys.path.insert(0, submodule_path)

# import this_file_dir.scienceagentbench.ScienceAgentBench.recover_pred_from_log
# module_path = os.path.join(this_file_dir, "scienceagentbench", "ScienceAgentBench", "recover_pred_from_log.py")
# module_name = "recover_pred_from_log"
# spec = importlib.util.spec_from_file_location(module_name, module_path)
# recover_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(recover_utils)
import recover_pred_from_log as recover_utils

# import this_file_dir.scienceagentbench.ScienceAgentBench.evaluation.harness as a package
# submodule_path = os.path.join(this_file_dir, "scienceagentbench", "ScienceAgentBench")
# sys.path.insert(0, submodule_path)
from evaluation.harness import run_evaluation as docker_eval, docker_build as docker_build

# import this_file_dir.scienceagentbench.ScienceAgentBench.calculate_metrics
# module_path = os.path.join(this_file_dir, "scienceagentbench", "ScienceAgentBench", "recover_pred_from_log.py")
# module_name = "recover_pred_from_log"
# spec = importlib.util.spec_from_file_location(module_name, module_path)
# recover_utils = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(recover_utils)
from calculate_metrics import evaluate_best_run


class ScienceAgentBench(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "scienceagentbench"

        # Support loading from local JSON file (for fix runner) or HuggingFace
        dataset_path = os.environ.get("SCIENCEAGENTBENCH_DATASET_PATH")
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "r") as f:
                ds = json.load(f)
        else:
            ds = load_dataset("osunlp/ScienceAgentBench", split="validation")
        self.benchmark = {}

        for task in ds:
            task_id = str(task['instance_id'])  # hal-harness requires the task_id to be a str, otherwise cannot match existing results
            self.benchmark[task_id] = {}
            for key in task:
                self.benchmark[task_id][key] = task[key]

            # Extract the dataset folder from the tree (e.g., "biosignals/" from "|-- biosignals/")
            dataset_folder = task["dataset_folder_tree"].split("\n")[0][4:]
            # Source path is in the submodule's benchmark/datasets/ directory
            src_dataset_path = os.path.join(submodule_path, "benchmark/datasets/", dataset_folder)
            # Destination path must be under "benchmark/" to match the prompt and evaluation environment
            dest_dataset_path = os.path.join("benchmark/datasets/", dataset_folder)
            self.benchmark[task_id]['files'] = {
                dest_dataset_path: src_dataset_path,
            }

        # Optional: Set if benchmark requires VM execution
        self.requires_sandbox = False
        # Optional: Path to VM setup script
        self.setup_script = "hal/benchmarks/scienceagentbench/setup.sh"
        super().__init__(agent_dir, config, setup_script=self.setup_script)
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent solutions"""
        self._recover_pred_from_log(agent_output, run_id)
        result_path = self._call_docker_eval(run_id)

        # Get task IDs in sorted order (matching how run_evaluation.py writes results)
        task_ids = sorted(self.benchmark.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))

        with open(result_path, "r") as f:
            result = {}
            lines = f.readlines()
            for i, task_id in enumerate(task_ids):
                if i < len(lines):
                    line = lines[i]
                    if line.strip() == '':
                        result[task_id] = {'valid_program': 0, 'codebert_score': 0, 'success_rate': 0, 'log_info': ''}
                    else:
                        instance_result = json.loads(line)
                        result[task_id] = instance_result
                else:
                    # Missing result for this task
                    result[task_id] = {'valid_program': 0, 'codebert_score': 0, 'success_rate': 0, 'log_info': ''}

        return {'agent_output': agent_output, 'eval_result': result}
        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        agent_output = eval_results['agent_output']
        eval_result = eval_results['eval_result']

        run_log = []
        eval_log = []
        # Iterate over actual task IDs in benchmark (not assuming sequential 1..N)
        # Sort numerically to maintain consistent ordering
        task_ids = sorted(self.benchmark.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
        for task_id in task_ids:
            output = agent_output[task_id]
            if output == "TIMEOUT after 900 seconds" or output == "TIMEOUT after 7200 seconds":
                run_log.append({
                    "history": [
                        {
                            "role": "assistant",
                            "content": "TIMEOUT",
                        }
                    ],
                    "cost": 0.0 # need to rely on weave log for accurate cost
                })
            elif isinstance(output, str):
                run_log.append({
                    "history": [
                        {
                            "role": "assistant",
                            "content": output,
                        }
                    ],
                    "cost": 0.0,
                })
            else:
                run_log.append(output)
            eval_log.append(eval_result[task_id])
        
        metrics = evaluate_best_run([run_log], [eval_log])
        return metrics


    def _recover_pred_from_log(self, agent_output, run_id):
        args = argparse.Namespace()
        args.benchmark_name_or_path = benchmark_path
        log_fname = os.path.join(self.get_run_dir(run_id), f"{run_id}_histories.jsonl")

        all_data = agent_output
        with open(log_fname, "w") as f:
            # Iterate over actual task IDs in benchmark (not assuming sequential 1..N)
            # Sort numerically to maintain consistent ordering
            task_ids = sorted(self.benchmark.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
            for task_id in task_ids:
                if all_data[task_id] == "TIMEOUT after 900 seconds" or all_data[task_id] == "TIMEOUT after 7200 seconds":
                    f.write(json.dumps(
                        {
                            "history": [
                                {
                                    "role": "assistant",
                                    "content": "TIMEOUT",
                                }
                            ],
                            "cost": 0.0 # need to rely on weave log for accurate cost
                        }
                    ) + "\n")
                else:
                    f.write(json.dumps(all_data[task_id]) + "\n")

        args.log_fname = log_fname
        args.pred_program_path = os.path.join(self.get_run_dir(run_id), "pred_programs")
        args.is_opendevin = False  # TODO: set this to True if using opendevin

        recover_utils.main(args)

    def _call_docker_eval(self, run_id: str):
        """Call docker eval script"""
        # Change logs dir
        run_path = Path(self.get_run_dir(run_id))
        docker_build.BASE_IMAGE_BUILD_DIR = run_path / docker_build.BASE_IMAGE_BUILD_DIR
        docker_build.INSTANCE_IMAGE_BUILD_DIR = run_path / docker_build.INSTANCE_IMAGE_BUILD_DIR
        docker_eval.INSTANCE_IMAGE_BUILD_DIR = run_path / docker_eval.INSTANCE_IMAGE_BUILD_DIR
        docker_eval.RUN_EVALUATION_LOG_DIR = run_path / docker_eval.RUN_EVALUATION_LOG_DIR

        log_fname = os.path.join(self.get_run_dir(run_id), f"{run_id}_eval.jsonl")

        # Only evaluate the tasks that are in the (possibly filtered) benchmark
        task_ids = list(self.benchmark.keys())

        docker_eval.main(  # TODO: These arguments are not configurable right now
            benchmark_path=benchmark_path,
            pred_program_path=os.path.join(self.get_run_dir(run_id), "pred_programs"),
            log_fname=log_fname,
            dataset_name='osunlp/ScienceAgentBench',
            split='validation',
            instance_ids=task_ids,
            max_workers=os.cpu_count() // 2,
            force_rebuild=True,
            force_rebuild_base=False,
            cache_level="none",
            clean=False,
            open_file_limit=4096,
            run_id=run_id + '_' + time.strftime("%Y%m%d-%H%M%S"),
            timeout=1800,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY", ""),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        )

        return log_fname
