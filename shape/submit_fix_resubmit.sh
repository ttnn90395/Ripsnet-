#!/bin/bash
# Resubmit all missing jobs with fixed train_shape.py
# - HybridOnEquivariantTensorFieldNetwork now in TFN_MODELS (precomputed geometry works)
# - Hybrid forward_batch path uses precomputed KNN instead of recomputing
# - All jobs use CPU (CUDA_VISIBLE_DEVICES=) to avoid GPU OOM on shared SallesInfo nodes
# Run this ON the cluster: bash submit_fix_resubmit.sh [epochs]
set -euo pipefail

REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/shape"
EPOCHS=${1:-50}
cd "$REMOTE_DIR"

submitted=0
skipped=0

submit_job() {
    local ds=$1 model=$2 t=$3 mem=$4 time=$5
    local jname="shape_${ds}_${model}_t${t}"
    if [ -f "results/${jname}.json" ]; then
        skipped=$(( skipped + 1 ))
        return
    fi
    sbatch --job-name="$jname" --partition=SallesInfo \
        --ntasks=1 --cpus-per-task=4 --mem="$mem" \
        --time="$time" \
        --output="logs/${jname}.out" \
        --error="logs/${jname}.err" \
        --wrap="cd $REMOTE_DIR && CUDA_VISIBLE_DEVICES= python train_shape.py $ds $model $EPOCHS $t" 2>&1 | grep -q "Submitted" && submitted=$(( submitted + 1 ))
}

# 2D: Relaxed + Hybrid (1h CPU, 32G)
for ds in circles circles_noisy; do
    for model in RelaxedOnEquivariantTensorFieldNetwork HybridOnEquivariantTensorFieldNetwork; do
        for t in 0 1 2; do
            submit_job "$ds" "$model" "$t" 32G 01:00:00
        done
    done
done

# 2D: OnEquivariant + Attention (need more than 1h on CPU)
for ds in circles circles_noisy; do
    for model in OnEquivariantTensorFieldNetwork AttentionTensorFieldNetwork; do
        for t in 0 1 2; do
            submit_job "$ds" "$model" "$t" 32G 02:00:00
        done
    done
done

# 3D: Relaxed + Hybrid (8h CPU, 48G) — these are the slow ones
for ds in shapes3d_topology shapes3d_geometry shapes3d_complex shapes3d_8way; do
    for model in RelaxedOnEquivariantTensorFieldNetwork HybridOnEquivariantTensorFieldNetwork; do
        for t in 0 1 2; do
            submit_job "$ds" "$model" "$t" 48G 08:00:00
        done
    done
done

# 3D: ScalarDistanceDeepSet on topology (NaN loss, but still run for completeness)
for ds in shapes3d_topology; do
    for model in ScalarDistanceDeepSet; do
        for t in 0 1 2; do
            submit_job "$ds" "$model" "$t" 32G 02:00:00
        done
    done
done

echo "Submitted $submitted new jobs (skipped $skipped existing)"
