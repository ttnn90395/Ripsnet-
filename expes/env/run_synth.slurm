#!/bin/bash
#SBATCH --job-name=ripsnet-synth
#SBATCH --partition=qc-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/synth_%A_%a.out
#SBATCH --error=logs/synth_%A_%a.err
#SBATCH --array=0-9

# Run synthetic experiments from the Ripsnet- repo on RCC Cloud.
# Usage: sbatch expes/env/run_synth.slurm
#
# Runs 10 array tasks (try0..try9), each doing:
#   1. Generate synthetic data (PI and LS)
#   2. Train neural networks
#   3. Analyze results

set -euo pipefail

# --- Configuration ---
REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
CONDA_PREFIX="/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-$(uname -m)"
ENV_NAME="ripsnet"
IDENTIFIER="try${SLURM_ARRAY_TASK_ID}"

# --- Setup ---
cd "$REPO_DIR/expes"
mkdir -p logs

# Activate conda
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== $(date) Starting synth experiments: $IDENTIFIER ==="
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

# --- Run the experiments (mode=cluster generate train) ---
# PI experiment
bash expe_synth.sh \
  synth _train_PI_1${IDENTIFIER} _train_PI_2${IDENTIFIER} \
  _test_PI_clean_1${IDENTIFIER} _test_PI_noise_1${IDENTIFIER} \
  - - - 1 - - - - 50 - - - - - 1 1000 \
  PI 0 10 1 cluster generate train

# LS experiment
bash expe_synth.sh \
  synth _train_LS_1${IDENTIFIER} _train_LS_2${IDENTIFIER} \
  _test_LS_clean_1${IDENTIFIER} _test_LS_noise_1${IDENTIFIER} \
  - - - 1 - - - - 300 - - - - - 1 1000 \
  PL 5 - - cluster generate train

echo "=== $(date) Done: $IDENTIFIER ==="
