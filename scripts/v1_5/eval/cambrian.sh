#!/bin/bash
#SBATCH --job-name=cambrian-eval-llava-v
#SBATCH --output=/work/tesi_fgaragnani/logs/cambrian/%x-%j
#SBATCH --error=/work/tesi_fgaragnani/logs/cambrian/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=all_usr_prod
#SBATCH --account=tesi_fgaragnani
#SBATCH --time=02:30:00
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --array=0-22

set -e

# Module loading
module unload cuda
module load cuda/11.8

# Environment setup
source activate cambrian
cd /work/tesi_fgaragnani/llava

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Cache directories - DO NOT use $HOME
export HF_HUB_CACHE="/work/tesi_fgaragnani/checkpoints/"
export HF_HOME="/work/tesi_fgaragnani/checkpoints/"
export HF_DATASETS_CACHE="/work/tesi_fgaragnani/datasets/"
export TRANSFORMERS_OFFLINE=1
export TOKENIZER_PATH="/work/tesi_fgaragnani/checkpoints/lmsys/vicuna-7b-v1.5"
export IS_LLAVA_MORE=0
export WORK="/work/tesi_fgaragnani"
export PYTHONPATH="$WORK/llava:$WORK:$PYTHONPATH"

llava_more="llava-v"
model_name="${1:-${llava_more}}"
conv_mode="${2:-llama_3_1}"
eval_output_dir="/work/tesi_fgaragnani/logs/cambrian-eval-llava-v"
gpu_devices="${3:-0}"
safe_model_name=$(tr '/' '_' <<< $model_name)

echo "Conversation mode: $conv_mode"
echo "Evaluation output directory: $eval_output_dir"
echo "GPU devices: $gpu_devices"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Create evaluation output directory
mkdir -p "$eval_output_dir"
echo "Created evaluation directory: $eval_output_dir"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES="$gpu_devices"

# All Cambrian benchmarks
benchmarks=(
    gqa
    vizwiz
    scienceqa
    textvqa
    pope
    mme
    mmbench_en
    mmbench_cn
    seed
    # mmvet
    mmmu
    mathvista
    ai2d
    chartqa
    # docvqa
    # infovqa
    # stvqa
    ocrbench
    mmstar
    realworldqa
    qbench
    blink
    mmvp
    vstar
    ade
    omni
    coco
    # synthdog
)

# Select benchmark based on array task ID
benchmark=${benchmarks[$SLURM_ARRAY_TASK_ID]}

echo "Running benchmark: $benchmark (array index: $SLURM_ARRAY_TASK_ID)"
start_time=$(date +%s)
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "Starting benchmark $benchmark at $timestamp"

# Create benchmark-specific output directory
benchmark_output_dir="${eval_output_dir}/${benchmark}"
mkdir -p "$benchmark_output_dir"

# Run the benchmark with custom output directory
if bash scripts/v1_5/eval/run_benchmark_custom.sh \
    --benchmark "$benchmark" \
    --model_name "$model_name" \
    --conv_mode "$conv_mode" \
    --output_dir "$benchmark_output_dir"; then
    echo "Successfully completed benchmark: $benchmark"
    
    # Create a completion marker file
    touch "${benchmark_output_dir}/.${safe_model_name}_completed"
    
else
    echo "Error: Failed to complete benchmark: $benchmark"
    exit 1
fi

end_time=$(date +%s)
duration=$(( (end_time - start_time) / 60 ))
cur_timestamp=$(date +"%Y-%m-%d %H:%M:%S")

echo "Benchmark $benchmark completed in $duration minutes"
echo "Finished at: $cur_timestamp"

echo "Benchmark $benchmark evaluation completed successfully!"
echo "Results saved in: $benchmark_output_dir"