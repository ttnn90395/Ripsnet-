#!/bin/bash
# Runner for training fraction + density ablation experiments
# Usage: bash run_ablations.sh [fraction|density|both]

set -euo pipefail

EXPERIMENT=${1:-both}
TRIALS=${2:-5}
PARTITION=${3:-ai-l40s}
IDENTIFIER="sweep_default"

# Datasets to run (small subset for MVP)
DATASETS=(
    "ItalyPowerDemand"   # Sensor, small (67 train, 22 pts)
    "CBF"                # Simulated, medium (30 train, 126 pts)
    "MiddlePhalanxOutlineCorrect"  # Image, medium (600 train, 78 pts)
    "ECG200"             # ECG (100 train, 94 pts)
)

# Models to test (one per group)
TFN_MODEL="OnEquivariantTensorFieldNetwork"
DMR_MODEL="DistanceMatrixRaggedModel"
PN_MODEL="PointNetTutorial"

MODELS=("$TFN_MODEL" "$DMR_MODEL" "$PN_MODEL")

FRACTIONS=(10 20 30 50 70 100)
EPOCHS=3000

RESULTS_DIR="/hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/results/ablations"
mkdir -p "$RESULTS_DIR"

submit_train_ablation() {
    local ds=$1 model=$2 frac=$3 trial=$4
    local jobname="trabs_${ds}_${model}_${frac}pct_t${trial}"
    sbatch --job-name="$jobname" \
           --partition="$PARTITION" \
           --gres=gpu:1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=16G \
           --time=02:00:00 \
           --output="$RESULTS_DIR/${jobname}.out" \
           --error="$RESULTS_DIR/${jobname}.err" \
           --wrap="cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes && python train_ablation.py $ds $model $frac $trial $EPOCHS $IDENTIFIER"
}

submit_density_ablation() {
    local ds=$1 model_label=$2 frac=$3 trial=$4
    local jobname="den_${ds}_${model_label}_${frac}pct_t${trial}"
    sbatch --job-name="$jobname" \
           --partition="$PARTITION" \
           --gres=gpu:1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=16G \
           --time=01:00:00 \
           --output="$RESULTS_DIR/${jobname}.out" \
           --error="$RESULTS_DIR/${jobname}.err" \
           --wrap="cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes && python density_ablation.py $ds $model_label $frac $trial $IDENTIFIER"
}

echo "=== Starting ablation experiments ==="
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Fractions: ${FRACTIONS[*]}"
echo "Trials: $TRIALS"
echo "Partition: $PARTITION"
echo ""

# Copy the scripts to the cluster
REMOTE_DIR="u0001943@login.cloud.r-ccs.riken.jp:/hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/"
scp train_ablation.py density_ablation.py "$REMOTE_DIR"

if [ "$EXPERIMENT" = "train" ] || [ "$EXPERIMENT" = "both" ]; then
    echo "--- Submitting training fraction ablations ---"
    for ds in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for frac in "${FRACTIONS[@]}"; do
                for ((t=0; t<TRIALS; t++)); do
                    submit_train_ablation "$ds" "$model" "$frac" "$t"
                done
            done
        done
    done
fi

if [ "$EXPERIMENT" = "density" ] || [ "$EXPERIMENT" = "both" ]; then
    echo "--- Submitting density ablations ---"
    for ds in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            for label in "$model" "${model}_GS"; do
                for frac in "${FRACTIONS[@]}"; do
                    for ((t=0; t<TRIALS; t++)); do
                        submit_density_ablation "$ds" "$label" "$frac" "$t"
                    done
                done
            done
        done
    done
fi

echo ""
echo "=== All jobs submitted ==="
echo "Check: squeue -u u0001943"
