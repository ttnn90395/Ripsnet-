#!/bin/bash
# Submit robustness improvement experiments to Polytechnique cluster
# Tests attention pooling, noise augmentation, DTM filtering, and PD fusion
# Run this ON the cluster: bash submit_robustness.sh [epochs]
set -euo pipefail

REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/shape"
EPOCHS=${1:-20}
cd "$REMOTE_DIR"

submitted=0
skipped=0

submit_job() {
    local ds=$1 model=$2 t=$3 mem=$4 time=$5 flags=$6
    local jname="shape_${ds}_${model}_t${t}_${flags// /_}"
    if [ -f "results/${jname}.json" ]; then
        skipped=$(( skipped + 1 ))
        return
    fi
    sbatch --job-name="$jname" --partition=SallesInfo \
        --ntasks=1 --cpus-per-task=4 --mem="$mem" \
        --time="$time" \
        --output="logs/${jname}.out" \
        --error="logs/${jname}.err" \
        --wrap="cd $REMOTE_DIR && CUDA_VISIBLE_DEVICES= python train_shape.py $ds $model $EPOCHS $t $flags" 2>&1 | grep -q "Submitted" && submitted=$(( submitted + 1 ))
}

DATASETS_2D="circles circles_noisy"
DATASETS_3D="shapes3d_topology shapes3d_geometry shapes3d_complex shapes3d_8way"
ALL_DATASETS="$DATASETS_2D $DATASETS_3D"
TRIALS="0 1 2"

echo "=== 1. RipsPointNet (new model: PointNet + PD fusion) ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "RipsPointNet" "$t" 16G 01:00:00 ""
        else
            submit_job "$ds" "RipsPointNet" "$t" 32G 03:00:00 ""
        fi
    done
done

echo "=== 2. GTTensorFieldNetworkV2 + attention pooling ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 32G 02:00:00 "--attn-pool"
        else
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 48G 08:00:00 "--attn-pool"
        fi
    done
done

echo "=== 3. OnEquivariantTensorFieldNetwork + attention pooling ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "OnEquivariantTensorFieldNetwork" "$t" 32G 02:00:00 "--attn-pool"
        else
            submit_job "$ds" "OnEquivariantTensorFieldNetwork" "$t" 48G 08:00:00 "--attn-pool"
        fi
    done
done

echo "=== 4. PointNet3D + noise augmentation ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "PointNet3D" "$t" 16G 01:00:00 "--noise-aug"
        else
            submit_job "$ds" "PointNet3D" "$t" 32G 03:00:00 "--noise-aug"
        fi
    done
done

echo "=== 5. GTTensorFieldNetworkV2 + PD fusion ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 32G 02:00:00 "--pd-fusion"
        else
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 48G 08:00:00 "--pd-fusion"
        fi
    done
done

echo "=== 6. HierarchicalTensorFieldNetwork + attention pooling ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "HierarchicalTensorFieldNetwork" "$t" 32G 02:00:00 "--attn-pool"
        else
            submit_job "$ds" "HierarchicalTensorFieldNetwork" "$t" 48G 08:00:00 "--attn-pool"
        fi
    done
done

echo ""
echo "Submitted $submitted new jobs (skipped $skipped existing)"
echo "Monitor: squeue -u ten.nguyen-hanaoka"
