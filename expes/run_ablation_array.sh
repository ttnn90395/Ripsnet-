#!/bin/bash
# SLURM array job runner for train_ablation.py
# Called by submit_all_dindon.sh
# Reads parameters from results/param_map.txt using SLURM_ARRAY_TASK_ID

EPOCHS=${1:-100}
IDENT=${2:-try1}
BATCH=${3:-0}

BASE="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/expes"
RESULTS="$BASE/results/ablations"
PARAM_FILE="$RESULTS/param_map_b${BATCH}.txt"

cd "$BASE"

# Force CPU mode — cluster has GPUs but SLURM doesn't manage GPU gres,
# so multiple jobs share the same GPU causing CUDA assertion failures.
export CUDA_VISIBLE_DEVICES=""

# Read our line from the parameter map
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$PARAM_FILE")
DS=$(echo "$LINE" | awk '{print $2}')
MODEL=$(echo "$LINE" | awk '{print $3}')
PCT=$(echo "$LINE" | awk '{print $4}')
TRIAL=$(echo "$LINE" | awk '{print $5}')

echo "=== Job $SLURM_ARRAY_TASK_ID: $DS $MODEL ${PCT}% trial=$TRIAL ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Check if result already exists
RESULT_FILE="$RESULTS/ablation_train_${DS}_${MODEL}_${PCT}pct_t${TRIAL}.json"
if [ -f "$RESULT_FILE" ]; then
  echo "Result already exists, skipping: $RESULT_FILE"
  exit 0
fi

# Run
python3 train_ablation.py "$DS" "$MODEL" "$PCT" "$TRIAL" "$EPOCHS" "$IDENT" 2>&1

echo "=== Done: $DS $MODEL ${PCT}% trial=$TRIAL ==="
