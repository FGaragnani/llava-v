#!/bin/bash
#SBATCH --job-name=llava-v_s1--mean
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_MLLM-RAG
#SBATCH --time=10:00:00

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

learning_rate=2e-4
mm_projector_lr=2e-5
run_name="${SLURM_JOB_NAME}"
output_dir="/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/${run_name}"
# output_dir="/work/tesi_fgaragnani/checkpoints/llava-v/${run_name}"

per_device_train_batch_size=8
gradient_accumulation_steps=4

language_model="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys/vicuna-7b-v1.5"
# language_model="/work/tesi_fgaragnani/checkpoints/lmsys/vicuna-7b-v1.5"
clip_model_name_or_path="/leonardo_scratch/large/userexternal/fgaragna/models/lmsys/openai/clip-vit-large-patch14"

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
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $language_model \
    --version plain \
    --data_path /leonardo_scratch/large/userexternal/fgaragna/dataset/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /leonardo_scratch/large/userexternal/fgaragna/dataset/llava_pretrain/images/ \
    --vision_tower $clip_model_name_or_path \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --alignment_crop_size 768 \
    --bf16 True \
    --fp16 False \
    --output_dir /leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/pretrain/${run_name} \
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
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_glamm True \
    --grand_image_dir /leonardo_scratch/large/userexternal/fgaragna/dataset/GLAMM/images/images/ \
    --grand_annotation_dir /leonardo_scratch/large/userexternal/fgaragna/dataset/GLAMM/annotations/simple/ \
    --patch_agg_mode cls \
    --grand_alignment_loss_weight 0.5 \
    --text_token_pool mean \
    --address_layer last_layer