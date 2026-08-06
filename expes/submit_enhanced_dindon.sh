#!/bin/bash
# Submit enhanced TFN experiments to dindon cluster
# Usage: bash submit_enhanced_dindon.sh [batch_id]

set -e

REMOTE="ten.nguyen-hanaoka@dindon.polytechnique.fr"
REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-"
SSH_CMD="sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no $REMOTE"
SCP_CMD="sshpass -p 'RHs2K.q-RsE4' scp -o ConnectTimeout=20 -o StrictHostKeyChecking=no"

echo "=== Syncing code to cluster ==="
$SCP_CMD tfn_enhancements.py $REMOTE:$REMOTE_DIR/tfn_enhancements.py
$SCP_CMD expes/train_enhanced.py $REMOTE:$REMOTE_DIR/expes/train_enhanced.py
$SCP_CMD expes/ensemble_evaluate.py $REMOTE:$REMOTE_DIR/expes/ensemble_evaluate.py
$SSH_CMD "cd $REMOTE_DIR && git pull || true"

echo "=== Submitting enhanced experiments ==="

# Generate parameter map for enhanced experiments
# Models to test with improvements: the best TFN variants
MODELS="TensorFieldNetwork OnEquivariantTensorFieldNetwork AttentionTensorFieldNetwork HybridOnEquivariantTensorFieldNetwork"
DATASETS="CBF ECG200 ECG5000 GunPoint Plane PowerCons SonyAIBORobotSurface1 SonyAIBORobotSurface2 TwoLeadECG UMD"
FRACTIONS="10 20 30 50 70 100"
TRIALS="0 1 2 3"

# Classifier modes: mlp (end-to-end) + xgboost (paper comparison)
CLASSIFIERS="mlp xgboost"
# Augmentation: on and off
AUGMENT_FLAGS="--augment "
# Hidden channels: default and large
HIDDEN_FLAGS=""

# Build parameter map
MAPFILE="results/enhanced/param_map_enhanced.txt"
$SSH_CMD "mkdir -p $REMOTE_DIR/expes/results/enhanced"

$SSH_CMD "cd $REMOTE_DIR/expes && python3 -c \"
import itertools
models = '$MODELS'.split()
datasets = '$DATASETS'.split()
fracs = [int(x) for x in '$FRACTIONS'.split()]
trials = [int(x) for x in '$TRIALS'.split()]
classifiers = '$CLASSIFIERS'.split()

lines = []
i = 0
for clf in classifiers:
    augment = '--augment' if clf == 'mlp' else ''
    for ds in datasets:
        for m in models:
            for f in fracs:
                for t in trials:
                    lines.append(f'{i} {ds} {m} {f} {t} {clf} {augment}')
                    i += 1
with open('results/enhanced/param_map_enhanced.txt', 'w') as fh:
    fh.write(chr(10).join(lines) + chr(10))
print(f'Generated {len(lines)} parameter combinations')
\""

# Count jobs
N_JOBS=$($SSH_CMD "wc -l < $REMOTE_DIR/expes/results/enhanced/param_map_enhanced.txt")
echo "Total enhanced jobs: $N_JOBS"

# Create SLURM script
$SSH_CMD "cat > $REMOTE_DIR/expes/submit_enhanced_array.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=tfn_enh
#SBATCH --output=slurm_logs/enhanced_%A_%a.out
#SBATCH --error=slurm_logs/enhanced_%A_%a.err
#SBATCH --array=0-NJOBS_PLACEHOLDER
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=SallesInfo

mkdir -p slurm_logs

MAPFILE=\"results/enhanced/param_map_enhanced.txt\"
LINE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" \"\$MAPFILE\")
IDX=\$(echo \"\$LINE\" | awk '{print \$1}')
DS=\$(echo \"\$LINE\" | awk '{print \$2}')
MODEL=\$(echo \"\$LINE\" | awk '{print \$3}')
PCT=\$(echo \"\$LINE\" | awk '{print \$4}')
TRIAL=\$(echo \"\$LINE\" | awk '{print \$5}')
CLF=\$(echo \"\$LINE\" | awk '{print \$6}')
AUG=\$(echo \"\$LINE\" | awk '{print \$7}')

echo \"=== Job \$SLURM_ARRAY_TASK_ID: \$DS \$MODEL \${PCT}% trial=\$TRIAL clf=\$CLF aug=\$AUG ===\"
echo \"Node: \$(hostname)\"
echo \"Time: \$(date)\"

export CUDA_VISIBLE_DEVICES=\"\"

RESULT=\"results/enhanced/train_\${DS}_\${MODEL}_\${PCT}pct_t\${TRIAL}.json\"
if [ -f \"\$RESULT\" ]; then
    echo \"Result already exists, skipping\"
    exit 0
fi

EXTRA_ARGS=\"\"
if [ \"\$CLF\" = \"mlp\" ]; then
    EXTRA_ARGS=\"--classifier mlp\"
elif [ \"\$CLF\" = \"xgboost\" ]; then
    EXTRA_ARGS=\"--classifier xgboost\"
fi
if [ \"\$AUG\" = \"--augment\" ]; then
    EXTRA_ARGS=\"\$EXTRA_ARGS --augment\"
fi

python3 train_enhanced.py \"\$DS\" \"\$MODEL\" \"\$PCT\" \"\$TRIAL\" 100 try1 \$EXTRA_ARGS

echo \"=== Done at \$(date) ===\"
SLURM

# Replace N_JOBS placeholder
$SSH_CMD "cd $REMOTE_DIR/expes && sed -i 's/NJOBS_PLACEHOLDER/$((N_JOBS-1))/g' submit_enhanced_array.sh"

# Submit
$SSH_CMD "cd $REMOTE_DIR/expes && sbatch submit_enhanced_array.sh"

echo "=== Done! ==="
