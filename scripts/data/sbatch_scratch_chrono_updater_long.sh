#!/bin/bash
#SBATCH --job-name=chronos
#SBATCH --output=/leonardo_scratch/large/userexternal/fgaragna/logs/logs_%A_%a.out
#SBATCH --error=/leonardo_scratch/large/userexternal/fgaragna/logs/logs_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=24:00:00
#SBATCH -A IscrB_MLLM-RAG
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-4


########## RELEVANT #################
# Put the paths to update date time remember to not put very big folders as compute has to finish in 4 hours
# Adjust also the array size
#####################################

paths=(
    "/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"
    "/leonardo_scratch/large/userexternal/fgaragna/dataset/GLAMM/images"
    "/leonardo_scratch/large/userexternal/fgaragna/dataset/GLAMM/annotations"
    "/leonardo_scratch/large/userexternal/fgaragna/dataset/llava_pretrain"
    "/leonardo_scratch/large/userexternal/fgaragna/dataset/mllm_evaluation/cvprw/"
)

path=${paths[$SLURM_ARRAY_TASK_ID]}
echo "PATH: ${path}"

cd /leonardo/home/userexternal/fgaragna/llava/scripts/data
./update_metadata.sh "$path"