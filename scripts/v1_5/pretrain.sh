#!/bin/bash
#SBATCH --job-name=llava-v_s1
#SBATCH --output=/work/tesi_fgaragnani/logs/%x_%j.out
#SBATCH --error=/work/tesi_fgaragnani/logs/%x_%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=32
#SBATCH --partition=all_usr_prod
#SBATCH --account=tesi_fgaragnani
#SBATCH --time=8:00:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8
module unload gcc 
module load gcc/11.3.0

source activate viral

PROJECT_ROOT="/work/tesi_fgaragnani/"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

export PYTHONUNBUFFERED=1
# export TORCH_HOME="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_PROJECT=jeppetto
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export HF_HUB_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models"
export HF_HUB_CACHE="/work/tesi_fgaragnani/checkpoints/"
export HF_HOME="/work/tesi_fgaragnani/checkpoints/"
export HF_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

learning_rate=2e-4
mm_projector_lr=2e-5
run_name="${SLURM_JOB_NAME}"
# output_dir="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/viral/${run_name}"
output_dir="/work/tesi_fgaragnani/checkpoints/viral/${run_name}"

per_device_train_batch_size=16
gradient_accumulation_steps=2

# language_model="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys/vicuna-7b-v1.5"
language_model="/work/tesi_fgaragnani/checkpoints/lmsys/vicuna-7b-v1.5"

((ws = $SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export WORLD_SIZE=$ws

dataloader_num_workers=$(( $SLURM_CPUS_PER_TASK / $SLURM_GPUS_PER_NODE))

echo "Nodes: ${SLURM_NNODES}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"
echo "WORLD SIZE: ${WORLD_SIZE}"
echo "DATALOADER WORKERS: ${dataloader_num_workers}"

srun --exclusive -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
    torchrun \
    --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --master-port=$MASTER_PORT --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
    train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path $language_model \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower /work/tesi_fgaragnani/checkpoints/openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --alignment_crop_size 768 \
    --bf16 False \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_glamm True \
    --grand_image_dir /work/tesi_fgaragnani/dataset/GLAMM/images \
    --grand_annotation_dir /work/tesi_fgaragnani/dataset/GLAMM/annotations/annotations/