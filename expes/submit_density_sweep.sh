#!/bin/bash
# Full density ablation sweep across all 21 UCR datasets.
# Submits for every model_label in MODEL_LABELS (raw + GS variants).
# Usage: bash submit_density_sweep.sh [trials] [partition]
set -euo pipefail

TRIALS=${1:-5}
PARTITION=${2:-ai-l40s}
EPOCHS=3000
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

MODEL_LABELS=(
    "OnEquivariantTensorFieldNetwork"
    "OnEquivariantTensorFieldNetwork_GS"
    "PointNetTutorial"
    "PointNetTutorial_GS"
    "PersNet"
    "PersNet_GS"
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
    "GTTensorFieldNetworkV2"
    "GTTensorFieldNetworkV2_GS"
    "HierarchicalGTTFN"
    "HierarchicalGTTFN_GS"
    "HierarchicalTensorFieldNetwork"
    "HierarchicalTensorFieldNetwork_GS"
    "StochasticTensorFieldNetwork"
    "StochasticTensorFieldNetwork_GS"
    "CrossAttentionTensorFieldNetwork"
    "CrossAttentionTensorFieldNetwork_GS"
    "ScalarInputMLP"
    "ScalarInputMLP_GS"
    "MultiInputModel"
    "MultiInputModel_GS"
    "RaggedPersistenceModel"
    "RaggedPersistenceModel_GS"
)

FRACTIONS=(10 20 30 50 70 100)

echo "=== Density Ablation Sweep ==="
echo "Datasets:     ${#DATASETS[@]}"
echo "Model labels: ${#MODEL_LABELS[@]}"
echo "Fractions:    ${FRACTIONS[*]}"
echo "Trials:       $TRIALS"
echo "Partition:    $PARTITION"
total_jobs=$(( ${#DATASETS[@]} * ${#MODEL_LABELS[@]} * ${#FRACTIONS[@]} * TRIALS ))
echo "Total jobs:   $total_jobs"
echo ""

scp density_ablation.py "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true

submit_density() {
    local ds=$1 ml=$2 frac=$3 trial=$4
    local jobname="den_${ds}_${ml}_${frac}pct_t${trial}"
    sbatch --job-name="$jobname" \
           --partition="$PARTITION" \
           --gres=gpu:1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=16G \
           --time=01:00:00 \
           --output="$RESULTS_DIR/${jobname}.out" \
           --error="$RESULTS_DIR/${jobname}.err" \
           --wrap="cd $REMOTE_DIR && python density_ablation.py $ds $ml $frac $trial $IDENTIFIER"
}

for ds in "${DATASETS[@]}"; do
    for ml in "${MODEL_LABELS[@]}"; do
        for frac in "${FRACTIONS[@]}"; do
            for ((t=0; t<TRIALS; t++)); do
                submit_density "$ds" "$ml" "$frac" "$t"
            done
        done
    done
done

echo "=== Done. Monitor: squeue -u u0001943 ==="
