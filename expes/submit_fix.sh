#!/bin/bash
#SBATCH --job-name=tfn_fix
#SBATCH --output=slurm_logs/fix_%A_%a.out
#SBATCH --error=slurm_logs/fix_%A_%a.err
#SBATCH --array=0-35
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --partition=SallesInfo

mkdir -p slurm_logs

MAPFILE="results/enhanced/missing_param_map.txt"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAPFILE")
DS=$(echo "$LINE" | awk '{print $2}')
MODEL=$(echo "$LINE" | awk '{print $3}')
PCT=$(echo "$LINE" | awk '{print $4}')
TRIAL=$(echo "$LINE" | awk '{print $5}')
CLF=$(echo "$LINE" | awk '{print $6}')
AUG=$(echo "$LINE" | awk '{print $7}')

echo "=== Fix job $SLURM_ARRAY_TASK_ID: $DS $MODEL ${PCT}% trial=$TRIAL clf=$CLF aug=$AUG ==="
echo "Node: $(hostname)"
echo "Time: $(date)"

export CUDA_VISIBLE_DEVICES=""

AUG_TAG=""
if [ "$AUG" = "--augment" ]; then
    AUG_TAG="_aug"
fi
RESULT="results/enhanced/train_${DS}_${MODEL}_${PCT}pct_t${TRIAL}_${CLF}${AUG_TAG}.json"
if [ -f "$RESULT" ]; then
    echo "Result already exists, skipping"
    exit 0
fi

EXTRA_ARGS=""
if [ "$CLF" = "mlp" ]; then
    EXTRA_ARGS="--classifier mlp"
elif [ "$CLF" = "xgboost" ]; then
    EXTRA_ARGS="--classifier xgboost"
fi
if [ "$AUG" = "--augment" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --augment"
fi

python3 train_enhanced.py "$DS" "$MODEL" "$PCT" "$TRIAL" 200 try1 $EXTRA_ARGS

echo "=== Done at $(date) ==="
