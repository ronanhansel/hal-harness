"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC-By-NC license found in the
LICENSE file in the root directory of this source tree.
"""

import multiprocessing
import signal
from queue import Empty

disable_printing = """
import sys
class DisablePrint:
    def write(self, x):
        pass
    def flush(self):
        pass
# Save the current state of stdout
original_stdout = sys.stdout
# Disable all printing to the console
sys.stdout = DisablePrint()
"""


# Handler function to be called when the alarm signal is received
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def get_function_output(function_definition, test_case):
    # exec(disable_printing)
    try:
        # Use an explicit namespace so eval can see exec-defined symbols.
        namespace = {}
        exec(function_definition, namespace, namespace)
        return eval(test_case, namespace, namespace)
    except Exception as e:
        return None


def queue_get_function_output(function_definition, test_case, queue):
    queue.put(get_function_output(function_definition, test_case))


def subprocess_get_function_output(function_definition, test_case):
    # do not want any os functions
    if (
        "import os" in function_definition
        or "from os" in function_definition
        or "import sys" in function_definition
        or "from sys" in function_definition
    ):
        return None
    if (
        "open(" in function_definition
        or "print(" in function_definition
        or "write" in function_definition
    ):
        return None
    if "sudo" in function_definition or "transformers" in function_definition:
        return None
    if "exit(" in function_definition or "quit(" in function_definition:
        return None

    # Set the signal handler for SIGALRM
    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(1)  # Set an alarm for 10 seconds
    # try:
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=queue_get_function_output, args=(function_definition, test_case, queue)
    )
    process.start()
    process.join(timeout=5)

    if process.is_alive():
        process.kill()
        process.join()  # Reap the zombie

    try:
        result = queue.get(timeout=0.1)
    except Empty:
        result = None
    return result


def check_correctness(ground_truth_function, test_function, test_cases):
    # Although unlikely, there is a chance that this function may run malicious code outputted by the LLMs
    num_correct = 0

    if (
        "import os" in test_function
        or "from os" in test_function
        or "import sys" in test_function
        or "from sys" in test_function
    ):
        return 0
    if "sudo" in test_function or "transformers" in test_function:
        return 0
    if "exit(" in test_function or "quit(" in test_function:
        return 0
    if "argparse" in test_function:
        return 0

    if "```python" in test_function:
        test_function = test_function.split("```python")[1].split("```")[0]
    if "```" in test_function:
        test_function = test_function.split("```")[1].split("```")[0]

    for test_case in test_cases.values():
        ground_truth_output = subprocess_get_function_output(ground_truth_function, test_case)

        # Timeout is handled internally by subprocess_get_function_output
        test_output = subprocess_get_function_output(test_function, test_case)
        
        try:
            if ground_truth_output == test_output and ground_truth_output is not None:
                num_correct += 1
        except Exception:
            pass
    return num_correct / len(test_cases)


def code_evaluate(trajectories):
    all_correctness = []
    skipped = 0
    
    from concurrent.futures import ThreadPoolExecutor

    def evaluate_single_trajectory(item):
        i, trajectory = item
        # Skip failed tasks (returned as error strings instead of dicts)
        if not isinstance(trajectory, dict):
            print(f"[code_evaluate] Skipping task {i}: not a dict (likely failed task)")
            return 0, True

        # Check required fields exist
        if "task" not in trajectory or "answer" not in trajectory:
            print(f"[code_evaluate] Skipping task {i}: missing 'task' or 'answer' field")
            return 0, True

        if not isinstance(trajectory["task"], dict):
            print(f"[code_evaluate] Skipping task {i}: 'task' is not a dict")
            return 0, True

        ground_truth_function = trajectory["task"].get("ground_truth")
        test_function = trajectory["answer"]
        test_cases = trajectory["task"].get("test_cases")

        if not ground_truth_function or not test_cases:
            print(f"[code_evaluate] Skipping task {i}: missing ground_truth or test_cases")
            return 0, True

        try:
            correctness = check_correctness(
                ground_truth_function, test_function, test_cases
            )
            return correctness, False
        except Exception as e:
            print(f"[code_evaluate] Error evaluating task {i}: {e}. Marking as failed.")
            return 0, False

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(evaluate_single_trajectory, enumerate(trajectories)))

    for correctness, is_skipped in results:
        all_correctness.append(correctness)
        if is_skipped:
            skipped += 1

    if skipped > 0:
        print(f"[code_evaluate] WARNING: Skipped {skipped} failed/invalid tasks (counted as 0)")

    if len(all_correctness) > 0:
        print(f"Average correctness: {sum(all_correctness)/len(all_correctness)}")
        print(f"Number of trajectories: {len(all_correctness)}")
        print(
            f"Percentage of correct trajectories: {sum([1 for correctness in all_correctness if correctness == 1])/len(all_correctness)}"
        )
    else:
        print("[code_evaluate] No valid trajectories to evaluate!")
    return all_correctness