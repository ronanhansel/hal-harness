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


def _run_subprocess_with_timeout(function_definition, test_case, timeout=3):
    """Run subprocess and return result, with hard timeout."""
    result_holder = [None]
    process = None
    queue = None

    def worker():
        nonlocal process, queue
        try:
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=queue_get_function_output, args=(function_definition, test_case, queue)
            )
            process.start()
            process.join(timeout=2)

            if process.is_alive():
                process.terminate()
                process.join(timeout=0.3)
                if process.is_alive():
                    process.kill()

            try:
                result_holder[0] = queue.get_nowait()
            except Empty:
                pass
        except:
            pass

    import threading
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout)  # Hard timeout on the entire operation

    # Cleanup in background (don't wait)
    def cleanup():
        try:
            if process and process.is_alive():
                process.kill()
        except:
            pass
        try:
            if queue:
                queue.close()
        except:
            pass

    cleanup_thread = threading.Thread(target=cleanup, daemon=True)
    cleanup_thread.start()

    return result_holder[0]


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

    return _run_subprocess_with_timeout(function_definition, test_case, timeout=3)


def check_correctness(ground_truth_function, test_function, test_cases, max_time=30):
    """
    Check correctness of test_function against ground_truth_function.

    Args:
        max_time: Maximum seconds to spend on this task (default 30s).
                  Prevents tasks with many test cases from blocking forever.
    """
    import time
    start_time = time.time()

    # Although unlikely, there is a chance that this function may run malicious code outputted by the LLMs
    num_correct = 0
    num_evaluated = 0

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
        # Check if we've exceeded the time limit
        if time.time() - start_time > max_time:
            print(f"[check_correctness] Time limit ({max_time}s) reached after {num_evaluated}/{len(test_cases)} test cases")
            break

        ground_truth_output = subprocess_get_function_output(ground_truth_function, test_case)

        # Timeout is handled internally by subprocess_get_function_output
        test_output = subprocess_get_function_output(test_function, test_case)

        num_evaluated += 1
        try:
            if ground_truth_output == test_output and ground_truth_output is not None:
                num_correct += 1
        except Exception:
            pass

    # Return proportion correct out of evaluated tests
    # Note: We limit to 5 test cases max, so this is fair comparison
    return num_correct / max(num_evaluated, 1)


def code_evaluate(trajectories):
    all_correctness = []
    skipped = 0
    timed_out = 0
    cancelled = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
    import time

    # Timeout per task evaluation (seconds) - prevents hanging on faulty code
    TASK_EVAL_TIMEOUT = 60
    # Overall evaluation timeout (seconds) - ensures evaluation always completes
    # Set to 15 minutes
    OVERALL_TIMEOUT = 900

    def evaluate_single_trajectory(item):
        i, trajectory = item
        # Skip failed tasks (returned as error strings instead of dicts)
        if not isinstance(trajectory, dict):
            print(f"[code_evaluate] Skipping task {i}: not a dict (likely failed task)")
            return i, 0, True

        # Check required fields exist
        if "task" not in trajectory or "answer" not in trajectory:
            print(f"[code_evaluate] Skipping task {i}: missing 'task' or 'answer' field")
            return i, 0, True

        if not isinstance(trajectory["task"], dict):
            print(f"[code_evaluate] Skipping task {i}: 'task' is not a dict")
            return i, 0, True

        ground_truth_function = trajectory["task"].get("ground_truth")
        test_function = trajectory["answer"]
        test_cases = trajectory["task"].get("test_cases")

        if not ground_truth_function or not test_cases:
            print(f"[code_evaluate] Skipping task {i}: missing ground_truth or test_cases")
            return i, 0, True

        try:
            correctness = check_correctness(
                ground_truth_function, test_function, test_cases
            )
            return i, correctness, False
        except Exception as e:
            print(f"[code_evaluate] Error evaluating task {i}: {e}. Marking as failed.")
            return i, 0, False

    from tqdm import tqdm

    # Use as_completed with per-task timeout to prevent hanging
    results_dict = {}
    eval_start_time = time.time()

    # Don't use context manager - we need to shutdown(wait=False) to avoid hanging
    executor = ThreadPoolExecutor(max_workers=32)
    try:
        # Submit all tasks and track them by index
        future_to_idx = {
            executor.submit(evaluate_single_trajectory, (i, traj)): i
            for i, traj in enumerate(trajectories)
        }

        # Process completed tasks with progress bar
        # Use wait() with timeout instead of as_completed() for better control
        from concurrent.futures import wait, FIRST_COMPLETED

        total_tasks = len(trajectories)
        total_futures = len(future_to_idx)
        print(f"[code_evaluate] Total trajectories: {total_tasks}, Total futures submitted: {total_futures}")

        pending = set(future_to_idx.keys())
        completed_task_ids = []

        with tqdm(total=total_tasks, desc="Evaluating") as pbar:
            while pending:
                # Check overall timeout
                elapsed = time.time() - eval_start_time
                if elapsed > OVERALL_TIMEOUT:
                    print(f"\n[code_evaluate] OVERALL TIMEOUT ({OVERALL_TIMEOUT}s) reached")
                    print(f"[code_evaluate] Stuck tasks: {sorted([future_to_idx[f] for f in pending])}")
                    for future in pending:
                        idx = future_to_idx[future]
                        future.cancel()
                        results_dict[idx] = (0, False, True)
                        cancelled += 1
                        pbar.update(1)
                    break

                # Print remaining tasks when 10 or fewer left
                if len(pending) <= 10 and len(pending) > 0:
                    remaining_ids = sorted([future_to_idx[f] for f in pending])
                    print(f"\n[code_evaluate] {len(pending)} tasks remaining: {remaining_ids}")

                # Wait for any task to complete, with short timeout to allow checking overall timeout
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)

                if not done:
                    # No tasks completed in 5 seconds - log which tasks are still pending
                    if len(pending) <= 5:
                        print(f"\n[code_evaluate] Still waiting for tasks: {sorted([future_to_idx[f] for f in pending])}")
                    continue

                for future in done:
                    idx = future_to_idx[future]
                    completed_task_ids.append(idx)
                    try:
                        i, correctness, is_skipped = future.result(timeout=1)
                        results_dict[i] = (correctness, is_skipped, False)
                    except FuturesTimeoutError:
                        print(f"[code_evaluate] Task {idx} result timeout, marking as failed")
                        results_dict[idx] = (0, False, True)
                    except Exception as e:
                        print(f"[code_evaluate] Task {idx} raised exception: {e}, marking as failed")
                        results_dict[idx] = (0, False, False)
                    pbar.update(1)

        print(f"[code_evaluate] Completed {len(completed_task_ids)} tasks, results_dict has {len(results_dict)} entries")
    finally:
        # Shutdown executor WITHOUT waiting for stuck threads
        executor.shutdown(wait=False, cancel_futures=True)
        print("[code_evaluate] Executor shutdown (not waiting for stuck threads)")

    # Reconstruct results in original order
    for i in range(len(trajectories)):
        if i in results_dict:
            correctness, is_skipped, is_timed_out = results_dict[i]
            all_correctness.append(correctness)
            if is_skipped:
                skipped += 1
            if is_timed_out:
                timed_out += 1
        else:
            # Should not happen, but handle gracefully
            print(f"[code_evaluate] Task {i} missing from results, marking as failed")
            all_correctness.append(0)
            cancelled += 1

    if skipped > 0:
        print(f"[code_evaluate] WARNING: Skipped {skipped} failed/invalid tasks (counted as 0)")
    if timed_out > 0:
        print(f"[code_evaluate] WARNING: {timed_out} tasks timed out (counted as 0)")
    if cancelled > 0:
        print(f"[code_evaluate] WARNING: {cancelled} tasks cancelled due to overall timeout (counted as 0)")

    if len(all_correctness) > 0:
        print(f"Average correctness: {sum(all_correctness)/len(all_correctness)}")
        print(f"Number of trajectories: {len(all_correctness)}")
        print(
            f"Percentage of correct trajectories: {sum([1 for correctness in all_correctness if correctness == 1])/len(all_correctness)}"
        )
    else:
        print("[code_evaluate] No valid trajectories to evaluate!")
    return all_correctness