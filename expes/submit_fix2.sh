#!/bin/bash
# Resubmit for the 11 failing datasets with geometry fix (records NaN instead of crashing)
FAILING_DATASETS=(
  "ChlorineConcentration" "DistalPhalanxOutlineCorrect" "GunPointOldVersusYoung"
  "ItalyPowerDemand" "MedicalImages" "MiddlePhalanxOutlineAgeGroup"
  "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW" "PhalangesOutlinesCorrect"
  "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW"
)
MODELS=(
  "TensorFieldNetwork" "GTTensorFieldNetwork" "GTTensorFieldNetworkV2"
  "HierarchicalGTTFN" "HierarchicalTensorFieldNetwork"
  "AttentionTensorFieldNetwork" "StochasticTensorFieldNetwork"
  "CrossAttentionTensorFieldNetwork"
)
PERCENTAGES=(10 20 30 50 70 100)
TRIALS=(0 1 2 3)

BASE="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/expes"
CD="cd $BASE"

total=0
for model in "${MODELS[@]}"; do
  for ds in "${FAILING_DATASETS[@]}"; do
    for pct in "${PERCENTAGES[@]}"; do
      for trial in "${TRIALS[@]}"; do
        jname="ta_${model}_${ds}_${pct}_t${trial}"
        sbatch --job-name="$jname" \
          --partition=SallesInfo --ntasks=1 --cpus-per-task=4 \
          --time=01:00:00 \
          --output="$BASE/results/ablations/${jname}.out" \
          --error="$BASE/results/ablations/${jname}.err" \
          --wrap="$CD && python3 train_ablation.py $ds $model $pct $trial 100 try1"
        total=$((total + 1))
      done
    done
  done
done

echo "Submitted $total jobs total"
