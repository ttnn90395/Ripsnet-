#!/bin/bash
# Submit new method experiments: denoise, geom-reg, feat-pd, and combinations
# Run this ON the cluster: bash submit_new_methods.sh [epochs]
set -euo pipefail

REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/shape"
EPOCHS=${1:-20}
cd "$REMOTE_DIR"
mkdir -p logs

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

echo "=== 1. PersNet + denoise (statistical) ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "PersNet" "$t" 16G 01:00:00 "--denoise"
        else
            submit_job "$ds" "PersNet" "$t" 32G 03:00:00 "--denoise"
        fi
    done
done

echo "=== 2. PersNet + denoise-meanshift ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "PersNet" "$t" 16G 01:00:00 "--denoise --denoise-method=meanshift"
        else
            submit_job "$ds" "PersNet" "$t" 32G 03:00:00 "--denoise --denoise-method=meanshift"
        fi
    done
done

echo "=== 3. PersNet + geom-reg ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "PersNet" "$t" 16G 01:00:00 "--geom-reg"
        else
            submit_job "$ds" "PersNet" "$t" 32G 03:00:00 "--geom-reg"
        fi
    done
done

echo "=== 4. PersNet + feat-pd ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "PersNet" "$t" 16G 01:00:00 "--feat-pd"
        else
            submit_job "$ds" "PersNet" "$t" 32G 03:00:00 "--feat-pd"
        fi
    done
done

echo "=== 5. ScalarInputMLP + feat-pd ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "ScalarInputMLP" "$t" 16G 01:00:00 "--feat-pd"
        else
            submit_job "$ds" "ScalarInputMLP" "$t" 32G 03:00:00 "--feat-pd"
        fi
    done
done

echo "=== 6. GTTensorFieldNetworkV2 + feat-pd ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 32G 02:00:00 "--feat-pd"
        else
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 48G 08:00:00 "--feat-pd"
        fi
    done
done

echo "=== 7. PersNet + combined (denoise + geom-reg + feat-pd) ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "PersNet" "$t" 16G 01:00:00 "--denoise --geom-reg --feat-pd"
        else
            submit_job "$ds" "PersNet" "$t" 32G 03:00:00 "--denoise --geom-reg --feat-pd"
        fi
    done
done

echo "=== 8. GTTensorFieldNetworkV2 + combined (denoise + feat-pd) ==="
for ds in $ALL_DATASETS; do
    for t in $TRIALS; do
        if [ "$ds" = "circles" ] || [ "$ds" = "circles_noisy" ]; then
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 32G 02:00:00 "--denoise --feat-pd"
        else
            submit_job "$ds" "GTTensorFieldNetworkV2" "$t" 48G 08:00:00 "--denoise --feat-pd"
        fi
    done
done

echo ""
echo "Submitted $submitted new jobs (skipped $skipped existing)"
echo "Monitor: squeue -u ten.nguyen-hanaoka"
