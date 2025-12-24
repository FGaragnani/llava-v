#!/bin/bash
#SBATCH --job-name=cambrian-eval-llava
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/cambrian-eval-llava/%x-%j
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/cambrian-eval-llava/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_MLLM-RAG
#SBATCH --time=02:30:00
#SBATCH --array=0-20

set -e

# Module loading
module unload cuda
module load cuda/11.8
module load anaconda3/2022.05
module load profile/deeplrn

# Environment setup
source activate cambrian

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Cache directories - DO NOT use $HOME
export HF_HUB_CACHE="/leonardo_scratch/large/userexternal/fgaragna/dataset/mllm_evaluation/cvprw"
export HF_HOME="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/"
export HF_DATASETS_CACHE="/leonardo_scratch/large/userexternal/fgaragna/dataset/mllm_evaluation/cvprw"
export TRANSFORMERS_OFFLINE=1
export TOKENIZER_PATH="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys/vicuna-7b-v1.5"
export IS_LLAVA_MORE=1
PROJECT_ROOT="$HOME"
export PYTHONPATH="$PROJECT_ROOT/llava:$PROJECT_ROOT:$PYTHONPATH"

llava_more="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-base/llava_s2" # <--
model_name="${1:-${llava_more}}"
conv_mode="${2:-vicuna_v1}"
eval_output_dir="/leonardo_scratch/large/userexternal/fgaragna/logs/cambrian-eval-llava-base" # <--
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
    # qbench
    blink
    mmvp
    vstar
    ade
    # omni
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