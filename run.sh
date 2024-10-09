#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --hf_token <your_huggingface_token_here>"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hf_token) HF_TOKEN="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Check if HF_TOKEN is provided
if [ -z "$HF_TOKEN" ]; then
    usage
fi

# Set Hugging Face CLI environment variable
export HUGGINGFACE_CLI_TOKEN="$HF_TOKEN"

# List of Python scripts
scripts=(
    "babylm.py"
    "edge_llm_training.py"
    "llm_merging.py"
    "minipile.py"
    "edge_llm_compression.py"
    "llm_effiency.py"
    "math_autoformalization.py"
)

# Function to run a script on a specific GPU
run_on_gpu() {
    script=$1
    gpu=$2
    echo "Running $script on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python "$script" &
}

# Check if we have enough GPUs
if [ ${#scripts[@]} -gt 8 ]; then
    echo "Error: More scripts than available GPUs"
    exit 1
fi

# Run each script on a separate GPU
for i in "${!scripts[@]}"; do
    if [ $i -lt 8 ]; then
        run_on_gpu "${scripts[$i]}" $i
    else
        echo "Warning: Not enough GPUs for ${scripts[$i]}"
    fi
done

# Wait for all background processes to finish
wait

echo "All scripts have completed execution"