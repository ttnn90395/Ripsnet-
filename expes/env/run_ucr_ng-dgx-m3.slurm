#!/bin/bash
#SBATCH --job-name=ripsnet-ucr-dgx
#SBATCH --partition=ng-dgx-m3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/ucr_%A_%a.out
#SBATCH --error=logs/ucr_%A_%a.err
#SBATCH --array=0-20

set -euo pipefail

REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
CONDA_PREFIX="/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-$(uname -m)"
ENV_NAME="ripsnet"
IDENTIFIER="try${SLURM_ARRAY_TASK_ID}"

export UCR_PATH="${UCR_PATH:-/hs/work0/home/users/${USER}/data/UCRArchive_2018}"
export TFN_MODEL_TAG="task_${SLURM_ARRAY_TASK_ID}"

cd "$REPO_DIR/expes"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LD_LIBRARY_PATH="$CONDA_PREFIX/envs/$ENV_NAME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== $(date) Starting UCR experiments: $IDENTIFIER ==="
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

python launch_expe.py ucr cluster generate train "$IDENTIFIER" 21 "$SLURM_ARRAY_TASK_ID"

echo "=== $(date) Done: $IDENTIFIER ==="
