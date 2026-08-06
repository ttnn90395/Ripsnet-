#!/bin/bash
# Resubmit only the 5 models that had build_model bugs
DATASETS=(
  "CBF" "ChlorineConcentration" "DistalPhalanxOutlineCorrect"
  "ECG200" "ECG5000" "GunPoint" "GunPointOldVersusYoung"
  "ItalyPowerDemand" "MedicalImages" "MiddlePhalanxOutlineAgeGroup"
  "MiddlePhalanxOutlineCorrect" "MiddlePhalanxTW"
  "PhalangesOutlinesCorrect" "Plane" "PowerCons"
  "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW"
  "SonyAIBORobotSurface1" "SonyAIBORobotSurface2"
  "TwoLeadECG" "UMD"
)
BROKEN_MODELS=(
  "MultiInputModel"
)
PERCENTAGES=(10 20 30 50 70 100)
TRIALS=(0 1 2 3)

BASE="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/expes"
CD="cd $BASE"

total=0
for model in "${BROKEN_MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
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
echo "Submitted $total training ablation jobs for broken models"
