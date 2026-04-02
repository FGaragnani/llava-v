#!/bin/bash

sbatch scripts/v1_5/eval/cambrian.sh stage_three/llava-base_s3--align_mean-midL
sbatch scripts/v1_5/eval/cambrian.sh stage_three/llava-base_s3--align_mean-midL_clip
sbatch scripts/v1_5/eval/cambrian.sh stage_three/llava-base_s3--align_mean-midL_siglip
sbatch scripts/v1_5/eval/cambrian.sh stage_three/llava-base_s3--align_extra-data