#!/bin/bash
# Resubmit failed ablation jobs using the fixed train_ablation.py
# Reads from results/ablations/resubmit_list.txt

BASE="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/expes"
RESULTS="$BASE/results/ablations"
LIST="$RESULTS/resubmit_list.txt"
EPOCHS=100
IDENT="try1"

cd "$BASE"

COUNT=0
while IFS=' ' read -r DS MODEL PCT TRIAL; do
  [ -z "$DS" ] && continue

  # Skip if result already exists
  RESULT_FILE="$RESULTS/ablation_train_${DS}_${MODEL}_${PCT}pct_t${TRIAL}.json"
  if [ -f "$RESULT_FILE" ]; then
    continue
  fi

  jname="redo_${DS}_${MODEL}_${PCT}_t${TRIAL}"
  sbatch --job-name="$jname" \
    --partition=SallesInfo \
    --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G \
    --time=00:30:00 \
    --output="$RESULTS/${jname}.out" \
    --error="$RESULTS/${jname}.err" \
    --wrap="cd $BASE && python3 train_ablation.py $DS $MODEL $PCT $TRIAL $EPOCHS $IDENT"
  COUNT=$((COUNT+1))
done < "$LIST"

echo "Resubmitted $COUNT jobs (skipped already-done). Total queued: $(squeue -u ten.nguyen-hanaoka -h | wc -l)"
