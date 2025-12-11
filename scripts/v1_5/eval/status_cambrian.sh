#!/bin/bash

# Configuration - adjust these paths to match your setup
eval_output_dir="${2:-/leonardo_scratch/large/userexternal/fgaragna/logs/cambrian-eval-llava-base}"
_model_name="LLaVA-Base"
model_name="${1:-${_model_name}}"
echo "Model name: ${model_name}"

# All Cambrian benchmarks (should match the array in the main script)
benchmarks=(
    gqa
    vizwiz
    scienceqa
    textvqa
    pope
    mme
    mmbench_en
    mmbench_cn
    seed
    # mmvet
    mmmu
    mathvista
    ai2d
    chartqa
    # docvqa
    # infovqa
    # stvqa
    ocrbench
    mmstar
    realworldqa
    # qbench
    blink
    mmvp
    vstar
    ade
    # omni
    coco
    # synthdog
)

echo "Checking status of Cambrian benchmarks for model: $model_name"
echo "=========================================================="

completed=0
failed=0
running=0
pending=0

for i in "${!benchmarks[@]}"; do
    benchmark="${benchmarks[$i]}"
    benchmark_output_dir="${eval_output_dir}/${benchmark}"
    echo "Checking folder: $benchmark_output_dir"
    
    # if [[ -f "${benchmark_output_dir}/.${model_name}_completed" ]]; then
    if [[ -f "${benchmark_output_dir}/experiments.csv" ]]; then
        echo "[$i] $benchmark: ✅ COMPLETED"
        ((completed++))
    elif [[ -d "$benchmark_output_dir" ]]; then
        echo "[$i] $benchmark: 🔄 RUNNING/FAILED (check logs)"
        ((running++))
    else
        echo "[$i] $benchmark: ⏳ PENDING"
        ((pending++))
    fi
done

echo ""
echo "Summary:"
echo "========="
echo "Completed: $completed/${#benchmarks[@]}"
echo "Running/Failed: $running"
echo "Pending: $pending"

# Check if all benchmarks are completed
if [[ $completed -eq ${#benchmarks[@]} ]]; then
    echo ""
    echo "🎉 All benchmarks completed!"
    echo ""
    echo "Running tabulation..."
    
    # Change to the appropriate directory
    cd $HOME/llava
    
    # Run tabulation
    python scripts/v1_5/eval/tabulate.py \
        --eval_dir "$eval_output_dir" \
        --experiment_csv experiments.csv \
        --out_pivot "${eval_output_dir}/cambrian_results.xlsx"
    
    echo "Results tabulated in: ${eval_output_dir}/cambrian_results.xlsx"
else
    echo ""
    echo "⏳ Waiting for remaining benchmarks to complete..."
    echo ""
    echo "To check job status:"
    echo "squeue -u \$USER"
    echo ""
    echo "To run tabulation manually once all complete:"
    echo "python cambrian/eval/scripts/tabulate.py --eval_dir \"$eval_output_dir\" --experiment_csv experiments.csv --out_pivot \"${eval_output_dir}/cambrian_results.xlsx\""
fi