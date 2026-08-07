#!/bin/bash
# Split 2880 enhanced jobs into 3 SLURM array batches and submit all

set -e

REMOTE_DIR="/users/eleves-a/2023/ten.nguyen-hanaoka/Ripsnet-/expes"

# Generate split parameter maps on cluster
sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no ten.nguyen-hanaoka@dindon.polytechnique.fr "cd $REMOTE_DIR && python3 -c \"
lines = open('results/enhanced/param_map_enhanced.txt').readlines()
total = len(lines)
batch_size = 960
n_batches = (total + batch_size - 1) // batch_size
for b in range(n_batches):
    start = b * batch_size
    end = min(start + batch_size, total)
    with open(f'results/enhanced/param_map_enhanced_b{b}.txt', 'w') as f:
        for i, line in enumerate(lines[start:end]):
            f.write(line)
    print(f'Batch {b}: lines {start}-{end-1} ({end-start} jobs)')
print(f'Total: {total} jobs in {n_batches} batches')
\""

# Create and submit each batch
for BATCH in 0 1 2; do
    MAX=$(sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no ten.nguyen-hanaoka@dindon.polytechnique.fr "wc -l < $REMOTE_DIR/results/enhanced/param_map_enhanced_b${BATCH}.txt")
    MAX=$((MAX - 1))

    echo "Submitting batch $BATCH with $((MAX+1)) jobs (array 0-$MAX)..."

    sshpass -p 'RHs2K.q-RsE4' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no ten.nguyen-hanaoka@dindon.polytechnique.fr "cd $REMOTE_DIR && cat > submit_enhanced_b${BATCH}.sh << 'ENDOFSCRIPT'
#!/bin/bash
#SBATCH --job-name=tfn_bBATCHNUM
#SBATCH --output=slurm_logs/enhanced_bBATCHNUM_%A_%a.out
#SBATCH --error=slurm_logs/enhanced_bBATCHNUM_%A_%a.err
#SBATCH --array=0-ARRAYMAX
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=SallesInfo

mkdir -p slurm_logs

MAPFILE=\"results/enhanced/param_map_enhanced_bBATCHNUM.txt\"
LINE=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" \"\$MAPFILE\")
IDX=\$(echo \"\$LINE\" | awk '{print \$1}')
DS=\$(echo \"\$LINE\" | awk '{print \$2}')
MODEL=\$(echo \"\$LINE\" | awk '{print \$3}')
PCT=\$(echo \"\$LINE\" | awk '{print \$4}')
TRIAL=\$(echo \"\$LINE\" | awk '{print \$5}')
CLF=\$(echo \"\$LINE\" | awk '{print \$6}')
AUG=\$(echo \"\$LINE\" | awk '{print \$7}')
MS=\$(echo \"\$LINE\" | awk '{print \$8}')
NSC=\$(echo \"\$LINE\" | awk '{print \$9}')
SF=\$(echo \"\$LINE\" | awk '{print \$10}')

echo \"=== Job \$SLURM_ARRAY_TASK_ID: \$DS \$MODEL \${PCT}% trial=\$TRIAL clf=\$CLF aug=\$AUG ms=\$MS ===\"
echo \"Node: \$(hostname)\"
echo \"Time: \$(date)\"

export CUDA_VISIBLE_DEVICES=\"\"

AUG_TAG=\"\"
if [ \"\$AUG\" = \"--augment\" ]; then
    AUG_TAG=\"_aug\"
fi
MS_TAG=\"\"
if [ \"\$MS\" = \"--multi-scale\" ]; then
    MS_TAG=\"_ms\"
fi
RESULT=\"results/enhanced/train_\${DS}_\${MODEL}_\${PCT}pct_t\${TRIAL}_\${CLF}\${AUG_TAG}\${MS_TAG}.json\"
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
if [ \"\$MS\" = \"--multi-scale\" ]; then
    EXTRA_ARGS=\"\$EXTRA_ARGS --multi-scale\"
    if [ \"\$NSC\" != \"-\" ]; then
        EXTRA_ARGS=\"\$EXTRA_ARGS --num-scales \$NSC\"
    fi
    if [ \"\$SF\" != \"-\" ]; then
        EXTRA_ARGS=\"\$EXTRA_ARGS --scale-factor \$SF\"
    fi
fi

python3 train_enhanced.py \"\$DS\" \"\$MODEL\" \"\$PCT\" \"\$TRIAL\" 100 try1 \$EXTRA_ARGS

echo \"=== Done at \$(date) ===\"
ENDOFSCRIPT

# Replace placeholders
sed -i \"s/BATCHNUM/$BATCH/g\" submit_enhanced_b${BATCH}.sh
sed -i \"s/ARRAYMAX/$MAX/g\" submit_enhanced_b${BATCH}.sh

echo '  Batch $BATCH script ready, submitting...'
sbatch submit_enhanced_b${BATCH}.sh"
done

echo "=== All enhanced batches submitted ==="
