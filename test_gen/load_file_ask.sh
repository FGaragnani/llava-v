#!/bin/bash
#SBATCH --job-name=load_and_ask
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%A_%a.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_S02_096
#SBATCH --time=00:30:00
#SBATCH --array=0-4

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8
module unload gcc 
module load gcc/11.3.0

source activate viral

PROJECT_ROOT="$HOME"
export PYTHONPATH="$PROJECT_ROOT/:$PROJECT_ROOT/llava/:$PYTHONPATH"

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

MODELS=(
    /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/llava-v_s2--ve-qwen2_5,
    /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/llava-v_s2--mean-midL_ve-qwen2_5,
    /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/stage_three/llava-base_s3--align_extra-data,
    /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/stage_three/llava-base_s3--align_mean-midL,
    /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-base/llava_s2
)

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit this script with sbatch."
    exit 1
fi

MODEL_PATH="${MODELS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "${MODEL_PATH}" ]; then
    echo "No model configured for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}."
    echo "Update MODELS and #SBATCH --array to matching lengths."
    exit 1
fi

echo "Running array task ${SLURM_ARRAY_TASK_ID} with model: ${MODEL_PATH}"
torchrun \
    --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --master-port=$MASTER_PORT --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
    test_gen/load_file_ask.py \
    --model-path "${MODEL_PATH}"