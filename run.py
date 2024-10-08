import os
import subprocess
import ray
from ray import tune

def run_training_script(config):
    script_path = config["script_path"]
    gpu_id = config["gpu_id"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Run the actual script and track progress
    process = subprocess.Popen(["python", script_path], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        if "progress:" in line.lower():
            try:
                progress = int(line.strip().split("progress:")[-1].strip().replace("%", ""))
                tune.report(progress=progress)
            except ValueError:
                continue
    
    process.wait()
    tune.report(progress=100)

if __name__ == "__main__":
    ray.init()
    
    # List of training scripts
    training_scripts = [
        "babylm.py", #1
        "edge_llm_compression.py", #2
        "edge_llm_training.py", #3
        "llm_efficiency.py", #4
        "llm_merging.py", #5
        "math_autoformalization.py", #6
        "minipile.py", #7
    ]

    # Number of available GPUs
    num_gpus = 8

    # Define the training tasks
    training_tasks = [
        {
            "script_path": script,
            "gpu_id": i % num_gpus
        }
        for i, script in enumerate(training_scripts)
    ]

    # Run the training tasks using Ray Tune
    tune.run(
        run_training_script,
        config={
            "script_path": tune.grid_search([task["script_path"] for task in training_tasks]),
            "gpu_id": tune.grid_search([task["gpu_id"] for task in training_tasks])
        },
        resources_per_trial={"gpu": 1}
    )

    print("All training processes completed.")