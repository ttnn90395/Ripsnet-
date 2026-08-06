#!/bin/bash
# Full isometry robustness sweep across all 21 UCR datasets.
# Tests rotation+translation invariance for each model at multiple augmentation levels.
# Usage: bash submit_isometry_sweep.sh [trials] [partition]
set -euo pipefail

TRIALS=${1:-5}
PARTITION=${2:-ai-l40s}
IDENTIFIER="run1"
REMOTE="u0001943@login.cloud.r-ccs.riken.jp"
REMOTE_DIR="/hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/expes"
RESULTS_DIR="results/ablations"

DATASETS=(
    "ChlorineConcentration"
    "ProximalPhalanxTW"
    "Plane"
    "GunPoint"
    "PhalangesOutlinesCorrect"
    "SonyAIBORobotSurface2"
    "ProximalPhalanxOutlineAgeGroup"
    "ECG5000"
    "ECG200"
    "MedicalImages"
    "PowerCons"
    "DistalPhalanxOutlineCorrect"
    "ItalyPowerDemand"
    "MiddlePhalanxOutlineAgeGroup"
    "SonyAIBORobotSurface1"
    "UMD"
    "TwoLeadECG"
    "MiddlePhalanxOutlineCorrect"
    "GunPointOldVersusYoung"
    "MiddlePhalanxTW"
    "CBF"
)

# Core model representatives (one per architecture family)
MODEL_LABELS=(
    "OnEquivariantTensorFieldNetwork"
    "OnEquivariantTensorFieldNetwork_GS"
    "PointNetTutorial"
    "PointNetTutorial_GS"
    "PointNet3D"
    "PointNet3D_GS"
    "DistanceMatrixRaggedModel"
    "DistanceMatrixRaggedModel_GS"
    "ScalarDistanceDeepSet"
    "ScalarDistanceDeepSet_GS"
    "AttentionTensorFieldNetwork"
    "AttentionTensorFieldNetwork_GS"
    "TensorFieldNetwork"
    "TensorFieldNetwork_GS"
    "GTTensorFieldNetwork"
    "GTTensorFieldNetwork_GS"
    "HierarchicalGTTFN"
    "HierarchicalGTTFN_GS"
    "ScalarInputMLP"
    "ScalarInputMLP_GS"
    "MultiInputModel"
    "MultiInputModel_GS"
)

N_AUGMENTS=20

echo "=== Isometry Robustness Sweep ==="
echo "Datasets:     ${#DATASETS[@]}"
echo "Model labels: ${#MODEL_LABELS[@]}"
echo "Augments:     $N_AUGMENTS"
echo "Trials:       $TRIALS"
echo "Partition:    $PARTITION"
total_jobs=$(( ${#DATASETS[@]} * ${#MODEL_LABELS[@]} * TRIALS ))
echo "Total jobs:   $total_jobs"
echo ""

scp isometry_ablation.py "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true

submit_isometry() {
    local ds=$1 ml=$2 trial=$3
    local jobname="iso_${ds}_${ml}_t${trial}"
    sbatch --job-name="$jobname" \
           --partition="$PARTITION" \
           --gres=gpu:1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=16G \
           --time=02:00:00 \
           --output="$RESULTS_DIR/${jobname}.out" \
           --error="$RESULTS_DIR/${jobname}.err" \
           --wrap="cd $REMOTE_DIR && python isometry_ablation.py $ds $ml $N_AUGMENTS $trial $IDENTIFIER"
}

for ds in "${DATASETS[@]}"; do
    for ml in "${MODEL_LABELS[@]}"; do
        for ((t=0; t<TRIALS; t++)); do
            submit_isometry "$ds" "$ml" "$t"
        done
    done
done

echo "=== Done. Monitor: squeue -u u0001943 ==="
