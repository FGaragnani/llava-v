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
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --array=0-2

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

export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODELS=(
    /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/llava-v_s2--mean_24L
    /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/llava-v_s2--mean_1L
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
python test_gen/load_file_ask.py --model-path "${MODEL_PATH}"