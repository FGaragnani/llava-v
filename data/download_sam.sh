#!/bin/bash
#SBATCH --job-name=downloading_sam--llava-v
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/%x-%j.err
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
#SBATCH --partition=lrd_all_serial
#SBATCH --account=IscrB_MLLM-RAG
#SBATCH --time=4:00:00

set -e

# Module loading
module unload cuda
module load cuda/11.8
module load anaconda3/2022.05
module load profile/deeplrn

cd $HOME/llava

python data/download_sam.py