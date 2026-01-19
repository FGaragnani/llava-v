#!/bin/bash

while IFS= read -r model_name; do
    echo "Processing: $model_name"
    
    mkdir /work/tesi_fgaragnani/checkpoints/llava-v/"$model_name"
    cd /work/tesi_fgaragnani/checkpoints/llava-v/"$model_name"
    scp fgaragna@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/"$model_name"/* .
    echo "Completed copying for: $model_name"
    
done