#!/bin/bash
# Submit improved GTTensorFieldNetworkV2 + improved 3-branch HybridGTTFN to dindon
# GTTensorFieldNetworkV2 now has: use_cov_features=True, use_attention_pool=True, better rho head
# HybridGTTFN now has 3 branches: GT-TFN + DistanceMatrix + PointNet
set -e

REMOTE="ten.nguyen-hanaoka@dindon.polytechnique.fr"
REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-"
SSH_CMD="sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no $REMOTE"
SCP_CMD="sshpass -p 'RHs2K.q-RsE4' scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no"

echo "=== Syncing code to cluster ==="
$SSH_CMD "cd $REMOTE_DIR && git pull 2>&1" || true

echo "=== Generating parameter map (local) ==="
MODELS="GTTensorFieldNetworkV2 HybridGTTFN"
DATASETS="CBF ECG200 ECG5000 GunPoint Plane PowerCons SonyAIBORobotSurface1 SonyAIBORobotSurface2 TwoLeadECG UMD ChlorineConcentration DistalPhalanxOutlineCorrect ItalyPowerDemand MedicalImages MiddlePhalanxOutlineAgeGroup MiddlePhalanxOutlineCorrect MiddlePhalanxTW PhalangesOutlinesCorrect ProximalPhalanxOutlineAgeGroup ProximalPhalanxTW GunPointOldVersusYoung"
FRACTIONS="10 20 30 50 70 100"
TRIALS="0 1 2 3"

MAP_LOCAL=$(mktemp /tmp/param_map_improved.XXXXXX)
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
$SCP_CMD "$MAP_LOCAL" $REMOTE:$REMOTE_DIR/expes/results/enhanced/param_map_improved.txt

N_JOBS=$(wc -l < "$MAP_LOCAL" | tr -d '[:space:]')
if ! [[ "$N_JOBS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not count parameter-map lines" >&2
    rm -f "$MAP_LOCAL"
    exit 1
fi
echo "Total improved-model jobs: $N_JOBS"

echo "=== Generating SLURM array script (local) ==="
SLURM_LOCAL=$(mktemp /tmp/submit_improved_array.XXXXXX)
cat > "$SLURM_LOCAL" << 'SLURM'
#!/bin/bash
#SBATCH --job-name=tfni_new
#SBATCH --output=slurm_logs/improved_%A_%a.out
#SBATCH --error=slurm_logs/improved_%A_%a.err
#SBATCH --array=0-NJOBS_PLACEHOLDER
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --partition=SallesInfo

mkdir -p slurm_logs

MAPFILE="results/enhanced/param_map_improved.txt"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAPFILE")
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

echo "=== Job $SLURM_ARRAY_TASK_ID: $DS $MODEL ${PCT}% trial=$TRIAL clf=$CLF aug=$AUG ms=$MS ==="
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
if [ -f "$RESULT" ] && grep -q '"pipeline_version": 2' "$RESULT" \
        && grep -q '"save_artifacts": true' "$RESULT"; then
    echo "Result already in v2 format with artifacts, skipping"
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
EXTRA_ARGS="$EXTRA_ARGS --save-artifacts"

python3 train_enhanced.py "$DS" "$MODEL" "$PCT" "$TRIAL" 100 try1 $EXTRA_ARGS

echo "=== Done at $(date) ==="
SLURM

perl -pi -e "s/NJOBS_PLACEHOLDER/$((N_JOBS-1))/g" "$SLURM_LOCAL"
$SCP_CMD "$SLURM_LOCAL" $REMOTE:$REMOTE_DIR/expes/submit_improved_array.sh
rm -f "$SLURM_LOCAL"

echo "=== Submitting (splitting for MaxArraySize=1000) ==="
MAX_ARRAY=1000
BATCH_START=0
BATCH_NUM=0
while [ $BATCH_START -lt $N_JOBS ]; do
    BATCH_END=$((BATCH_START + MAX_ARRAY - 1))
    if [ $BATCH_END -ge $N_JOBS ]; then
        BATCH_END=$((N_JOBS - 1))
    fi
    BATCH_SIZE=$((BATCH_END - BATCH_START + 1))
    BATCH_NUM=$((BATCH_NUM + 1))
    echo "Submitting batch $BATCH_NUM: jobs $BATCH_START-$BATCH_END ($BATCH_SIZE jobs)"
    $SSH_CMD "cd $REMOTE_DIR/expes && sed 's/#SBATCH --array=0-$((N_JOBS-1))/#SBATCH --array=$BATCH_START-$BATCH_END/' submit_improved_array.sh | sbatch"
    BATCH_START=$((BATCH_END + 1))
done

echo "=== Done! Total jobs submitted: $N_JOBS ==="
