#!/bin/bash
# Resubmit ONLY missing results for 11 datasets to dindon
# Code fix for HybridGTTFN rbf attribute is already uploaded
set -e

REMOTE="ten.nguyen-hanaoka@129.104.253.36"
REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-"
SSH_CMD="sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=60 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no $REMOTE"
SCP_CMD="sshpass -p 'RHs2K.q-RsE4' scp -o ConnectTimeout=60 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

echo "=== Generating missing-only parameter map ==="
MODELS="GTTensorFieldNetworkV2 CrossAttentionTensorFieldNetwork HybridGTTFN"
DATASETS="ChlorineConcentration DistalPhalanxOutlineCorrect ItalyPowerDemand MedicalImages MiddlePhalanxOutlineAgeGroup MiddlePhalanxOutlineCorrect MiddlePhalanxTW PhalangesOutlinesCorrect ProximalPhalanxOutlineAgeGroup ProximalPhalanxTW GunPointOldVersusYoung"
FRACTIONS="10 20 30 50 70 100"
TRIALS="0 1 2 3"

MAP_LOCAL=$(mktemp /tmp/param_map_resubmit.XXXXXX)
python3 - "$MODELS" "$DATASETS" "$FRACTIONS" "$TRIALS" "$MAP_LOCAL" <<'PY'
import sys, os
models = sys.argv[1].split()
datasets = sys.argv[2].split()
fracs = [int(x) for x in sys.argv[3].split()]
trials = [int(x) for x in sys.argv[4].split()]
out = sys.argv[5]

existing_dir = "/tmp"
# We'll generate the full map and let the SLURM script skip existing
lines = []
i = 0
for clf in ('mlp', 'xgboost'):
    augment = '--augment' if clf == 'mlp' else '-'
    for ms in (False, True):
        mstag = '--multi-scale' if ms else '-'
        nsc = 3 if ms else '-'
        sf = 0.5 if ms else '-'
        for ds in datasets:
            for m in models:
                for f in fracs:
                    for t in trials:
                        lines.append(f'{i} {ds} {m} {f} {t} {clf} {augment} {mstag} {nsc} {sf}')
                        i += 1
with open(out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print(f'Generated {len(lines)} parameter combinations')
PY

$SSH_CMD "mkdir -p $REMOTE_DIR/expes/results/enhanced"
$SCP_CMD "$MAP_LOCAL" $REMOTE:$REMOTE_DIR/expes/results/enhanced/param_map_resubmit.txt

N_JOBS=$(wc -l < "$MAP_LOCAL" | tr -d '[:space:]')
echo "Total jobs: $N_JOBS"

echo "=== Generating SLURM array script ==="
SLURM_LOCAL=$(mktemp /tmp/submit_resubmit_array.XXXXXX)
cat > "$SLURM_LOCAL" << 'SLURM'
#!/bin/bash
#SBATCH --job-name=tfni_resub
#SBATCH --output=slurm_logs/resub_%A_%a.out
#SBATCH --error=slurm_logs/resub_%A_%a.err
#SBATCH --array=0-1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=SallesInfo

mkdir -p slurm_logs

OFFSET=${OFFSET:-0}
MAPFILE="results/enhanced/param_map_resubmit.txt"
REAL_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
LINE=$(sed -n "$((REAL_ID+1))p" "$MAPFILE")
IDX=$(echo "$LINE" | awk '{print $1}')
DS=$(echo "$LINE" | awk '{print $2}')
MODEL=$(echo "$LINE" | awk '{print $3}')
PCT=$(echo "$LINE" | awk '{print $4}')
TRIAL=$(echo "$LINE" | awk '{print $5}')
CLF=$(echo "$LINE" | awk '{print $6}')
AUG=$(echo "$LINE" | awk '{print $7}')
MS=$(echo "$LINE" | awk '{print $8}')
NSC=$(echo "$LINE" | awk '{print $9}')
SF=$(echo "$LINE" | awk '{print $10}')

echo "=== Job $SLURM_ARRAY_TASK_ID (real=$REAL_ID): $DS $MODEL ${PCT}% trial=$TRIAL clf=$CLF aug=$AUG ms=$MS ==="
echo "Node: $(hostname)"
echo "Time: $(date)"

export CUDA_VISIBLE_DEVICES=""

AUG_TAG=""
if [ "$AUG" = "--augment" ]; then
    AUG_TAG="_aug"
fi
MS_TAG=""
if [ "$MS" = "--multi-scale" ]; then
    MS_TAG="_ms"
fi
RESULT="results/enhanced/train_${DS}_${MODEL}_${PCT}pct_t${TRIAL}_${CLF}${AUG_TAG}${MS_TAG}.json"
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
if [ "$MS" = "--multi-scale" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --multi-scale"
    if [ "$NSC" != "-" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --num-scales $NSC"
    fi
    if [ "$SF" != "-" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --scale-factor $SF"
    fi
fi

python3 train_enhanced.py "$DS" "$MODEL" "$PCT" "$TRIAL" 100 try1 $EXTRA_ARGS

echo "=== Done at $(date) ==="
SLURM

$SCP_CMD "$SLURM_LOCAL" $REMOTE:$REMOTE_DIR/expes/submit_resubmit_array.sh
rm -f "$SLURM_LOCAL"

echo "=== Submitting (MaxArraySize=1001, using OFFSET) ==="
MAX_ARRAY=1000
BATCH_START=0
BATCH_NUM=0
while [ $BATCH_START -lt $N_JOBS ]; do
    BATCH_END=$((BATCH_START + MAX_ARRAY))
    if [ $BATCH_END -ge $N_JOBS ]; then
        BATCH_END=$((N_JOBS - 1))
    fi
    BATCH_SIZE=$((BATCH_END - BATCH_START + 1))
    BATCH_NUM=$((BATCH_NUM + 1))
    LAST_IDX=$((BATCH_SIZE - 1))
    echo "Submitting batch $BATCH_NUM: OFFSET=$BATCH_START array=0-$LAST_IDX ($BATCH_SIZE jobs)"
    $SSH_CMD "cd $REMOTE_DIR/expes && OFFSET=$BATCH_START sbatch --array=0-$LAST_IDX submit_resubmit_array.sh"
    BATCH_START=$((BATCH_END + 1))
done

echo "=== Done! ==="
rm -f "$MAP_LOCAL"
