#!/bin/bash
# Copy scripts to cluster and submit ablation experiments
set -euo pipefail

REMOTE="u0001943@login.cloud.r-ccs.riken.jp"
REMOTE_DIR="/hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/"

echo "=== Copying scripts to cluster ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
scp "$SCRIPT_DIR/train_ablation.py" "$SCRIPT_DIR/density_ablation.py" "$REMOTE:$REMOTE_DIR"

echo "=== Submitting training fraction ablation (Experiment 1) ==="
ssh "$REMOTE" bash << 'RUNNER'
cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes
IDENTIFIER="sweep_default"
EPOCHS=3000
PARTITION="ai-l40s"
TRIALS=5
FRACTIONS=(10 20 30 50 70 100)

# Use available datasets with compact config
DATASETS=(
    "ItalyPowerDemand"
    "CBF"
    "ECG200"
    "MiddlePhalanxOutlineCorrect"
)
# Models with checkpoints known to exist
MODELS=(
    "OnEquivariantTensorFieldNetwork"
    "PointNetTutorial"
)

mkdir -p results/ablations

echo "Submitting training fraction jobs..."
for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for frac in "${FRACTIONS[@]}"; do
            for ((t=0; t<TRIALS; t++)); do
                jname="trabs_${ds}_${model}_${frac}pct_t${t}"
                sbatch --job-name="$jname" --partition="$PARTITION" \
                    --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=16G \
                    --time=03:00:00 \
                    --output="results/ablations/${jname}.out" \
                    --error="results/ablations/${jname}.err" \
                    --wrap="cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes && python train_ablation.py $ds $model $frac $t $EPOCHS $IDENTIFIER"
            done
        done
    done
done
RUNNER

echo "=== Submitting density ablation (Experiment 2) ==="
ssh "$REMOTE" bash << 'RUNNER2'
cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes
IDENTIFIER="sweep_default"
TRIALS=5
FRACTIONS=(10 20 30 50 70 100)

DATASETS=(
    "ItalyPowerDemand"
    "GunPoint"
    "SonyAIBORobotSurface1"
    "UMD"
)
MODEL_LABELS=(
    "OnEquivariantTensorFieldNetwork"
    "OnEquivariantTensorFieldNetwork_GS"
    "PointNetTutorial"
    "PointNetTutorial_GS"
    "DistanceMatrixRaggedModel"
    "DistanceMatrixRaggedModel_GS"
)

mkdir -p results/ablations

# Spread across partitions using array job pattern
PARTITIONS=("ng-dgx-m0" "ng-dgx-m1" "ng-dgx-m2" "ng-dgx-m3" "qc-a100")
total=0
for ds in "${DATASETS[@]}"; do
    for ml in "${MODEL_LABELS[@]}"; do
        for frac in "${FRACTIONS[@]}"; do
            for ((t=0; t<TRIALS; t++)); do
                pi=$(( total % ${#PARTITIONS[@]} ))
                PART=${PARTITIONS[$pi]}
                jname="den_${ds}_$(echo $ml | tr -d .)_${frac}pct_t${t}"
                sbatch --job-name="$jname" --partition="$PART" \
                    --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=16G \
                    --time=01:00:00 \
                    --output="results/ablations/${jname}.out" \
                    --error="results/ablations/${jname}.err" \
                    --wrap="cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes && python density_ablation.py $ds $ml $frac $t $IDENTIFIER" 2>&1 | tail -1
                total=$(( total + 1 ))
            done
        done
    done
done
echo "Submitted $total density ablation jobs across ${#PARTITIONS[@]} partitions"
RUNNER2

echo "=== Done ==="
echo "Monitor with: squeue -u u0001943"
echo "Check results at: $REMOTE_DIR/results/ablations/"
echo "To fetch results: scp -r $REMOTE:$REMOTE_DIR/results/ablations/ ./results/"
