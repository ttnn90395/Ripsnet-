#!/bin/bash
# Resubmit CUDA OOB models on CPU, and CrossAttention with fixed forward()
# Only run for the 11 datasets that CUDA OOB on GPU (and all 21 for CrossAttention)
FAILING_DATASETS=(
  "ChlorineConcentration" "DistalPhalanxOutlineCorrect" "GunPointOldVersusYoung"
  "ItalyPowerDemand" "MedicalImages" "MiddlePhalanxOutlineAgeGroup"
  "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW" "PhalangesOutlinesCorrect"
  "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW"
)
ALL_DATASETS=(
  "CBF" "ChlorineConcentration" "DistalPhalanxOutlineCorrect"
  "ECG200" "ECG5000" "GunPoint" "GunPointOldVersusYoung"
  "ItalyPowerDemand" "MedicalImages" "MiddlePhalanxOutlineAgeGroup"
  "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW"
  "PhalangesOutlinesCorrect" "Plane" "PowerCons"
  "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW"
  "SonyAIBORobotSurface1" "SonyAIBORobotSurface2"
  "TwoLeadECG" "UMD"
)
CUDA_OOB_MODELS=(
  "TensorFieldNetwork" "GTTensorFieldNetwork" "GTTensorFieldNetworkV2"
  "HierarchicalGTTFN" "HierarchicalTensorFieldNetwork"
  "AttentionTensorFieldNetwork" "StochasticTensorFieldNetwork"
)
PERCENTAGES=(10 20 30 50 70 100)
TRIALS=(0 1 2 3)

BASE="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/expes"
CD="cd $BASE"

# 1) CUDA OOB models on CPU (11 failing datasets only)
total=0
echo "=== CUDA OOB models on CPU ==="
for model in "${CUDA_OOB_MODELS[@]}"; do
  for ds in "${FAILING_DATASETS[@]}"; do
    for pct in "${PERCENTAGES[@]}"; do
      for trial in "${TRIALS[@]}"; do
        jname="ta_${model}_${ds}_${pct}_t${trial}_cpu"
        sbatch --job-name="$jname" \
          --partition=SallesInfo --ntasks=1 --cpus-per-task=4 \
          --time=04:00:00 \
          --output="$BASE/results/ablations/${jname}.out" \
          --error="$BASE/results/ablations/${jname}.err" \
          --wrap="CUDA_VISIBLE_DEVICES="" $CD && python3 train_ablation.py $ds $model $pct $trial 10 try1"
        total=$((total + 1))
      done
    done
  done
done

# 2) CrossAttentionTensorFieldNetwork on GPU (all 21 datasets)
echo "=== CrossAttentionTensorFieldNetwork on GPU ==="
for ds in "${ALL_DATASETS[@]}"; do
  for pct in "${PERCENTAGES[@]}"; do
    for trial in "${TRIALS[@]}"; do
      jname="ta_CrossAttentionTensorFieldNetwork_${ds}_${pct}_t${trial}"
      sbatch --job-name="$jname" \
        --partition=SallesInfo --ntasks=1 --cpus-per-task=4 \
        --time=01:00:00 \
        --output="$BASE/results/ablations/${jname}.out" \
        --error="$BASE/results/ablations/${jname}.err" \
        --wrap="$CD && python3 train_ablation.py $ds CrossAttentionTensorFieldNetwork $pct $trial 100 try1"
      total=$((total + 1))
    done
  done
done

echo "Submitted $total jobs total"
