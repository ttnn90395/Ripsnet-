#!/bin/bash
# Comprehensive ablation submission for ALL 21 datasets
# Usage: bash submit_all_ablations.sh [train|density|both]
set -euo pipefail

REMOTE="u0001943@login.cloud.r-ccs.riken.jp"
REMOTE_DIR="/hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes/"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE="${1:-both}"
IDENTIFIER="try1"
TRIALS=2
FRACTIONS=(10 20 30 50 70 100)

# All 21 UCR datasets
DATASETS=(
    "ChlorineConcentration" "ProximalPhalanxTW" "Plane" "GunPoint"
    "PhalangesOutlinesCorrect" "SonyAIBORobotSurface2" "ProximalPhalanxOutlineAgeGroup"
    "ECG5000" "ECG200" "MedicalImages" "PowerCons" "DistalPhalanxOutlineCorrect"
    "ItalyPowerDemand" "MiddlePhalanxOutlineAgeGroup" "SonyAIBORobotSurface1"
    "UMD" "TwoLeadECG" "MiddlePhalanxOutlineCorrect" "GunPointOldVersusYoung"
    "MiddlePhalanxTW" "CBF"
)

# Copy scripts first
echo "=== Copying scripts to cluster ==="
scp "$SCRIPT_DIR/train_ablation.py" "$SCRIPT_DIR/density_ablation.py" "$REMOTE:$REMOTE_DIR"

# Partitions for load balancing
TRAIN_PARTITIONS=("ai-l40s" "ai-h200-brc" "qc-a100")
DENSITY_PARTITIONS=("ng-dgx-m0" "ng-dgx-m1" "ng-dgx-m2" "ng-dgx-m3" "qc-a100" "ai-l40s" "ai-h200-brc")

submit_training() {
    echo "=== Submitting TRAINING ablation for all 21 datasets ==="
    ssh "$REMOTE" bash << 'TRAIN'
cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes
IDENTIFIER="try1"
EPOCHS=3000
TRIALS=2
FRACTIONS=(10 20 30 50 70 100)
MODELS=("OnEquivariantTensorFieldNetwork" "PointNetTutorial")
TRAIN_PARTITIONS=("ai-l40s" "ai-h200-brc" "qc-a100")
DATASETS=(
    "ChlorineConcentration" "ProximalPhalanxTW" "Plane" "GunPoint"
    "PhalangesOutlinesCorrect" "SonyAIBORobotSurface2" "ProximalPhalanxOutlineAgeGroup"
    "ECG5000" "ECG200" "MedicalImages" "PowerCons" "DistalPhalanxOutlineCorrect"
    "ItalyPowerDemand" "MiddlePhalanxOutlineAgeGroup" "SonyAIBORobotSurface1"
    "UMD" "TwoLeadECG" "MiddlePhalanxOutlineCorrect" "GunPointOldVersusYoung"
    "MiddlePhalanxTW" "CBF"
)

mkdir -p results/ablations
total=0
for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for frac in "${FRACTIONS[@]}"; do
            for ((t=0; t<TRIALS; t++)); do
                pi=$(( total % ${#TRAIN_PARTITIONS[@]} ))
                PART="${TRAIN_PARTITIONS[$pi]}"
                jname="trabs_${ds}_${model}_${frac}pct_t${t}"
                sbatch --job-name="$jname" --partition="$PART" \
                    --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=16G \
                    --time=03:00:00 \
                    --output="results/ablations/${jname}.out" \
                    --error="results/ablations/${jname}.err" \
                    --wrap="cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes && python train_ablation.py $ds $model $frac $t $EPOCHS $IDENTIFIER" 2>&1 | grep -E "^Submitted|error"
                total=$(( total + 1 ))
            done
        done
    done
done
echo "Submitted $total training ablation jobs"
TRAIN
}

submit_density() {
    echo "=== Submitting DENSITY ablation for datasets with checkpoints ==="
    ssh "$REMOTE" bash << 'DENSITY'
cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes
IDENTIFIER="try1"
TRIALS=2
FRACTIONS=(10 20 30 50 70 100)
PARTITIONS=("ng-dgx-m0" "ng-dgx-m1" "ng-dgx-m2" "ng-dgx-m3" "qc-a100" "ai-l40s" "ai-h200-brc")

# Datasets that have ALL 3 models (plain + GS)
DATASETS=(
    "GunPointOldVersusYoung"
    "GunPoint"
    "ItalyPowerDemand"
    "SonyAIBORobotSurface1"
    "SonyAIBORobotSurface2"
    "TwoLeadECG"
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
total=0
for ds in "${DATASETS[@]}"; do
    for ml in "${MODEL_LABELS[@]}"; do
        for frac in "${FRACTIONS[@]}"; do
            for ((t=0; t<TRIALS; t++)); do
                pi=$(( total % ${#PARTITIONS[@]} ))
                PART="${PARTITIONS[$pi]}"
                jname="den_${ds}_${ml}_${frac}pct_t${t}"
                # ng-dgx partitions don't support --mem flag
                if [[ "$PART" == ng-dgx-* ]]; then
                    sbatch --job-name="$jname" --partition="$PART" \
                        --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
                        --time=01:00:00 \
                        --output="results/ablations/${jname}.out" \
                        --error="results/ablations/${jname}.err" \
                        --wrap="cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes && python density_ablation.py $ds $ml $frac $t $IDENTIFIER" 2>&1 | grep -E "^Submitted|error"
                else
                    sbatch --job-name="$jname" --partition="$PART" \
                        --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=16G \
                        --time=01:00:00 \
                        --output="results/ablations/${jname}.out" \
                        --error="results/ablations/${jname}.err" \
                        --wrap="cd /hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes && python density_ablation.py $ds $ml $frac $t $IDENTIFIER" 2>&1 | grep -E "^Submitted|error"
                fi
                total=$(( total + 1 ))
            done
        done
    done
done
echo "Submitted $total density ablation jobs"
DENSITY
}

case "$MODE" in
    train)    submit_training ;;
    density)  submit_density ;;
    both)     submit_training; submit_density ;;
    *)        echo "Usage: $0 [train|density|both]"; exit 1 ;;
esac

echo "=== Done ==="
echo "Monitor: squeue -u u0001943"
