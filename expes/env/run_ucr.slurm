#!/bin/bash
#SBATCH --job-name=ripsnet-ucr
#SBATCH --partition=qc-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/ucr_%A_%a.out
#SBATCH --error=logs/ucr_%A_%a.err
#SBATCH --array=0-9

# Run UCR experiments from the Ripsnet- repo on RCC Cloud.
# Usage: sbatch expes/env/run_ucr.slurm
#
# Runs 10 array tasks (try0..try9), each running the full UCR experiment loop.

set -euo pipefail

# --- Configuration ---
REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
CONDA_PREFIX="/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-$(uname -m)"
ENV_NAME="ripsnet"
IDENTIFIER="try${SLURM_ARRAY_TASK_ID}"

# IMPORTANT: Set UCR_PATH to where UCRArchive_2018 is located on the cluster.
# Download from: https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/
export UCR_PATH="${UCR_PATH:-/hs/work0/home/users/${USER}/data/UCRArchive_2018}"

# --- Setup ---
cd "$REPO_DIR/expes"
mkdir -p logs

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "=== $(date) Starting UCR experiments: $IDENTIFIER ==="
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

# --- Run via launch_expe.py ---
python launch_expe.py ucr cluster generate train "$IDENTIFIER"

echo "=== $(date) Done: $IDENTIFIER ==="
