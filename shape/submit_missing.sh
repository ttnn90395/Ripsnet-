#!/bin/bash
# Submit missing shape classification experiments to Polytechnique cluster
# Run this ON the cluster directly: bash submit_missing.sh [epochs]
set -euo pipefail

REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/shape"

EPOCHS=${1:-50}

cd "$REMOTE_DIR"
mkdir -p results models logs

total=0
submitted=0

# --- Circles + circles_noisy (2D models that now support n=2) ---
DATASETS_2D="circles circles_noisy"
MODELS_2D="OnEquivariantTensorFieldNetwork AttentionTensorFieldNetwork StochasticTensorFieldNetwork RelaxedOnEquivariantTensorFieldNetwork HybridOnEquivariantTensorFieldNetwork HierarchicalTensorFieldNetwork"

for ds in $DATASETS_2D; do
    for model in $MODELS_2D; do
        for t in 0 1 2; do
            jname="shape_${ds}_${model}_t${t}"
            if [ -f "results/${jname}.json" ]; then
                echo "SKIP $jname (already exists)"
                total=$(( total + 1 ))
                continue
            fi
            sbatch --job-name="$jname" --partition=SallesInfo \
                --ntasks=1 --cpus-per-task=4 --mem=8G \
                --time=01:00:00 \
                --output="logs/${jname}.out" \
                --error="logs/${jname}.err" \
                --wrap="cd $REMOTE_DIR && python train_shape.py $ds $model $EPOCHS $t" 2>&1 | grep -E "Submitted batch" && submitted=$(( submitted + 1 ))
            total=$(( total + 1 ))
        done
    done
done

# Single missing 2D trials
for entry in "circles:PointNet3D:1" "circles_noisy:RipsPointNet:0" "circles_noisy:PointNet3D:1"; do
    IFS=':' read -r ds model t <<< "$entry"
    jname="shape_${ds}_${model}_t${t}"
    if [ -f "results/${jname}.json" ]; then
        echo "SKIP $jname (already exists)"
        total=$(( total + 1 ))
        continue
    fi
    sbatch --job-name="$jname" --partition=SallesInfo \
        --ntasks=1 --cpus-per-task=4 --mem=8G \
        --time=01:00:00 \
        --output="logs/${jname}.out" \
        --error="logs/${jname}.err" \
        --wrap="cd $REMOTE_DIR && python train_shape.py $ds $model $EPOCHS $t" 2>&1 | grep -E "Submitted batch" && submitted=$(( submitted + 1 ))
    total=$(( total + 1 ))
done

# --- HierarchicalTensorFieldNetwork on 3D ---
DATASETS_3D="shapes3d_topology shapes3d_geometry shapes3d_complex shapes3d_8way"
for ds in $DATASETS_3D; do
    for t in 0 1 2; do
        jname="shape_${ds}_HierarchicalTensorFieldNetwork_t${t}"
        if [ -f "results/${jname}.json" ]; then
            echo "SKIP $jname (already exists)"
            total=$(( total + 1 ))
            continue
        fi
        sbatch --job-name="$jname" --partition=SallesInfo \
            --ntasks=1 --cpus-per-task=4 --mem=8G \
            --time=02:00:00 \
            --output="logs/${jname}.out" \
            --error="logs/${jname}.err" \
            --wrap="cd $REMOTE_DIR && python train_shape.py $ds HierarchicalTensorFieldNetwork $EPOCHS $t" 2>&1 | grep -E "Submitted batch" && submitted=$(( submitted + 1 ))
        total=$(( total + 1 ))
    done
done

echo "Submitted $submitted/$total new jobs"
