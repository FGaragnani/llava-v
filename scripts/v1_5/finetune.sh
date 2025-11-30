#!/bin/bash
#SBATCH --job-name=llava-v_s2
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_MLLM-RAG
#SBATCH --time=24:00:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8
module unload gcc 
module load gcc/11.3.0

source activate viral

PROJECT_ROOT="$HOME"
export PYTHONPATH="$PROJECT_ROOT/:$PROJECT_ROOT/llava/:$PYTHONPATH"

export PYTHONUNBUFFERED=1
# export TORCH_HOME="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_PROJECT=jeppetto
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models"
export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GRAND_FORCE_MASK=1

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

run_name="${SLURM_JOB_NAME}"
output_dir="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/${run_name}"

learning_rate=2e-5
per_device_train_batch_size=4
gradient_accumulation_steps=2

dataloader_num_workers=4

model_name="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/pretrain/llava-v_s1"
train_data_path="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/second_stage_LLaVA/llava_v1_5_mix665k.json"
train_image_folder="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset"

srun --exclusive -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
    torchrun \
    --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --master-port=$MASTER_PORT --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
    train/train.py \
    --seed 42 \
    --deepspeed ./scripts/zero3.json \
    --gradient_checkpointing True \
    --save_steps 500 \
    --save_total_limit 1 \
    --output_dir $output_dir \
    --run_name $run_name \
    --report_to wandb \
    --model_name $model_name \
    --attn_implementation sdpa \
    --conv_template vicuna_v1 \
    --instruction_tuning \
    --train_data_path $train_data_path \
    --train_image_folder $train_image_folder \
    --remove_unused_columns False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --eval_strategy "no" \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --dataloader_num_workers $dataloader_num_workers \
    --prompt_max_length 2048 \
    --image_aspect_ratio pad \
    --group_by_mod_lens True