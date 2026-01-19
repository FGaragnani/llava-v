#!/bin/bash

file="models_list.txt"

while IFS= read -r model_name; do
    echo "Processing: $model_name"
    
    mkdir -p /work/tesi_fgaragnani/checkpoints/llava-v/"$model_name"
    cd /work/tesi_fgaragnani/checkpoints/llava-v/"$model_name" || { echo "Failed to cd to /work/tesi_fgaragnani/checkpoints/llava-v/$model_name"; continue; }
    scp fgaragna@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/"$model_name"/* .
    echo "Completed copying for: $model_name"
    
done < "$file"