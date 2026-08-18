#!/bin/bash
# Submit 3 models for 11 missing datasets to dindon
# Uses OFFSET approach: each batch uses array 0-1000 but reads different offset into param map
set -e

REMOTE="ten.nguyen-hanaoka@dindon.polytechnique.fr"
REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-"
SSH_CMD="sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no $REMOTE"
SCP_CMD="sshpass -p 'RHs2K.q-RsE4' scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

echo "=== Syncing code to cluster ==="
$SCP_CMD "$REPO_DIR/models.py" $REMOTE:$REMOTE_DIR/models.py
$SCP_CMD "$REPO_DIR/gt_improvements.py" $REMOTE:$REMOTE_DIR/gt_improvements.py
$SCP_CMD "$REPO_DIR/gt_tfn_layer.py" $REMOTE:$REMOTE_DIR/gt_tfn_layer.py
$SCP_CMD train_enhanced.py $REMOTE:$REMOTE_DIR/expes/train_enhanced.py
$SSH_CMD "cd $REMOTE_DIR && git pull || true"

echo "=== Generating parameter map (local) ==="
MODELS="GTTensorFieldNetworkV2 CrossAttentionTensorFieldNetwork HybridGTTFN"
DATASETS="ChlorineConcentration DistalPhalanxOutlineCorrect ItalyPowerDemand MedicalImages MiddlePhalanxOutlineAgeGroup MiddlePhalanxOutlineCorrect MiddlePhalanxTW PhalangesOutlinesCorrect ProximalPhalanxOutlineAgeGroup ProximalPhalanxTW GunPointOldVersusYoung"
FRACTIONS="10 20 30 50 70 100"
TRIALS="0 1 2 3"

MAP_LOCAL=$(mktemp /tmp/param_map_missing11.XXXXXX)
python3 - "$MODELS" "$DATASETS" "$FRACTIONS" "$TRIALS" "$MAP_LOCAL" <<'PY'
import sys
models = sys.argv[1].split()
datasets = sys.argv[2].split()
fracs = [int(x) for x in sys.argv[3].split()]
trials = [int(x) for x in sys.argv[4].split()]
out = sys.argv[5]
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
$SCP_CMD "$MAP_LOCAL" $REMOTE:$REMOTE_DIR/expes/results/enhanced/param_map_missing11.txt

N_JOBS=$(wc -l < "$MAP_LOCAL" | tr -d '[:space:]')
if ! [[ "$N_JOBS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not count parameter-map lines" >&2
    rm -f "$MAP_LOCAL"
    exit 1
fi
echo "Total jobs for 11 missing datasets: $N_JOBS"

echo "=== Generating SLURM array script with OFFSET support ==="
SLURM_LOCAL=$(mktemp /tmp/submit_missing11_array.XXXXXX)
cat > "$SLURM_LOCAL" << 'SLURM'
#!/bin/bash
#SBATCH --job-name=tfni_m11
#SBATCH --output=slurm_logs/missing11_%A_%a.out
#SBATCH --error=slurm_logs/missing11_%A_%a.err
#SBATCH --array=0-1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --partition=SallesInfo

mkdir -p slurm_logs

OFFSET=${OFFSET:-0}
MAPFILE="results/enhanced/param_map_missing11.txt"
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

$SCP_CMD "$SLURM_LOCAL" $REMOTE:$REMOTE_DIR/expes/submit_missing11_array.sh
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
    $SSH_CMD "cd $REMOTE_DIR/expes && OFFSET=$BATCH_START sbatch --array=0-$LAST_IDX submit_missing11_array.sh"
    BATCH_START=$((BATCH_END + 1))
done

echo "=== Done! Total jobs submitted: $N_JOBS ==="
rm -f "$MAP_LOCAL"
