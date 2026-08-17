#!/bin/bash
# Resubmit missing jobs with OOM/time fixes
# - Force CPU for 2D (circles) to avoid GPU contention
# - Give 3D jobs more time (4h)
set -euo pipefail

REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/shape"

EPOCHS=${1:-50}

cd "$REMOTE_DIR"
mkdir -p results models logs

total=0
submitted=0

# --- 2D jobs: FORCE CPU to avoid OOM ---
DATASETS_2D="circles circles_noisy"
MODELS_2D="OnEquivariantTensorFieldNetwork AttentionTensorFieldNetwork StochasticTensorFieldNetwork RelaxedOnEquivariantTensorFieldNetwork HybridOnEquivariantTensorFieldNetwork HierarchicalTensorFieldNetwork"

for ds in $DATASETS_2D; do
    for model in $MODELS_2D; do
        for t in 0 1 2; do
            jname="shape_${ds}_${model}_t${t}"
            if [ -f "results/${jname}.json" ]; then
                total=$(( total + 1 ))
                continue
            fi
            sbatch --job-name="$jname" --partition=SallesInfo \
                --ntasks=1 --cpus-per-task=4 --mem=8G \
                --time=01:00:00 \
                --output="logs/${jname}.out" \
                --error="logs/${jname}.err" \
                --wrap="cd $REMOTE_DIR && CUDA_VISIBLE_DEVICES= python train_shape.py $ds $model $EPOCHS $t" 2>&1 | grep -E "Submitted batch" && submitted=$(( submitted + 1 ))
            total=$(( total + 1 ))
        done
    done
done

# Single missing 2D trials: force CPU
for entry in "circles:PersNet:1" "circles_noisy:PersNet:1" "circles_noisy:AttentionTensorFieldNetwork:0"; do
    IFS=':' read -r ds model t <<< "$entry"
    jname="shape_${ds}_${model}_t${t}"
    if [ -f "results/${jname}.json" ]; then
        total=$(( total + 1 ))
        continue
    fi
    sbatch --job-name="$jname" --partition=SallesInfo \
        --ntasks=1 --cpus-per-task=4 --mem=8G \
        --time=01:00:00 \
        --output="logs/${jname}.out" \
        --error="logs/${jname}.err" \
        --wrap="cd $REMOTE_DIR && CUDA_VISIBLE_DEVICES= python train_shape.py $ds $model $EPOCHS $t" 2>&1 | grep -E "Submitted batch" && submitted=$(( submitted + 1 ))
    total=$(( total + 1 ))
done

# --- 3D big model jobs: MORE TIME, SERIAL on GPU ---
# Relaxed/Hybrid/HierarchicalTFN on 3D timed out at 1-2hr
for ds in shapes3d_topology shapes3d_geometry shapes3d_complex shapes3d_8way; do
    for model in RelaxedOnEquivariantTensorFieldNetwork HybridOnEquivariantTensorFieldNetwork HierarchicalTensorFieldNetwork; do
        for t in 0 1 2; do
            jname="shape_${ds}_${model}_t${t}"
            if [ -f "results/${jname}.json" ]; then
                total=$(( total + 1 ))
                continue
            fi
            # 4 hours for big models on 3D, request GPU exclusively
            sbatch --job-name="$jname" --partition=SallesInfo \
                --ntasks=1 --cpus-per-task=4 --mem=16G \
                --time=04:00:00 \
                --output="logs/${jname}.out" \
                --error="logs/${jname}.err" \
                --wrap="cd $REMOTE_DIR && python train_shape.py $ds $model $EPOCHS $t" 2>&1 | grep -E "Submitted batch" && submitted=$(( submitted + 1 ))
            total=$(( total + 1 ))
        done
    done
done

echo "Submitted $submitted/$total new jobs"
