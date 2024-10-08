import os
import multiprocessing
import subprocess

def run_training_script(script_path, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(["python", script_path], env=env)

if __name__ == "__main__":
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

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_gpus)

    # Start training processes
    for i, script in enumerate(training_scripts):
        gpu_id = i % num_gpus
        pool.apply_async(run_training_script, args=(script, gpu_id))

    # Wait for all processes to complete
    pool.close()
    pool.join()

    print("All training processes completed.")