#!/bin/bash
# Submit ONLY the combos missing bars in the enhanced 100% plots.
# PersNet collapsed on 6 datasets; OnEquivTFN missing mlp on 2 + never ran on Sony2.
# Usage: bash submit_enhanced_fill_gaps.sh
set -e

REMOTE="ten.nguyen-hanaoka@dindon.polytechnique.fr"
REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-"
SSH_CMD="sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no $REMOTE"
SCP_CMD="sshpass -p 'RHs2K.q-RsE4' scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no"

echo "=== Syncing code to cluster ==="
$SCP_CMD tfn_enhancements.py $REMOTE:$REMOTE_DIR/tfn_enhancements.py
$SCP_CMD expes/train_enhanced.py $REMOTE:$REMOTE_DIR/expes/train_enhanced.py

echo "=== Generating targeted parameter map (local) ==="
MODELS="PersNet OnEquivariantTensorFieldNetwork"
DATASETS="ECG200 ECG5000 GunPoint Plane PowerCons SonyAIBORobotSurface1 SonyAIBORobotSurface2"
FRACTIONS="100"
TRIALS="0 1 2 3"

MAP_LOCAL=$(mktemp /tmp/param_map_fill.XXXXXX)
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
    ms = '-'; nsc = '-'; sf = '-'
    for ds in datasets:
        for m in models:
            for f in fracs:
                for t in trials:
                    lines.append(f'{i} {ds} {m} {f} {t} {clf} {augment} {ms} {nsc} {sf}')
                    i += 1
with open(out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print(f'Generated {len(lines)} parameter combinations')
PY

$SSH_CMD "mkdir -p $REMOTE_DIR/expes/results/enhanced"
$SCP_CMD "$MAP_LOCAL" $REMOTE:$REMOTE_DIR/expes/results/enhanced/param_map_fill.txt

N_JOBS=$(wc -l < "$MAP_LOCAL" | tr -d '[:space:]')
echo "Total fill-gap jobs: $N_JOBS"

echo "=== Generating SLURM array script (local) ==="
SLURM_LOCAL=$(mktemp /tmp/submit_fill_array.XXXXXX)
cat > "$SLURM_LOCAL" << 'SLURM'
#!/bin/bash
#SBATCH --job-name=tfn_fill
#SBATCH --output=slurm_logs/fill_%A_%a.out
#SBATCH --error=slurm_logs/fill_%A_%a.err
#SBATCH --array=0-NJOBS_PLACEHOLDER
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=SallesInfo

mkdir -p slurm_logs

MAPFILE="results/enhanced/param_map_fill.txt"
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

echo "=== Job $SLURM_ARRAY_TASK_ID: $DS $MODEL ${PCT}% trial=$TRIAL clf=$CLF ==="
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

EXTRA_ARGS="--classifier $CLF"
if [ "$AUG" = "--augment" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --augment"
fi
EXTRA_ARGS="$EXTRA_ARGS --save-artifacts"

python3 train_enhanced.py "$DS" "$MODEL" "$PCT" "$TRIAL" 100 try1 $EXTRA_ARGS

echo "=== Done at $(date) ==="
SLURM

perl -pi -e "s/NJOBS_PLACEHOLDER/$((N_JOBS-1))/g" "$SLURM_LOCAL"
$SCP_CMD "$SLURM_LOCAL" $REMOTE:$REMOTE_DIR/expes/submit_fill_array.sh
rm -f "$SLURM_LOCAL"

echo "=== Submitting ==="
$SSH_CMD "cd $REMOTE_DIR/expes && sbatch submit_fill_array.sh"

echo "=== Done! ==="
