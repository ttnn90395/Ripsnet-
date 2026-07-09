#!/bin/bash
#SBATCH --job-name=ripsnet-ucr-h100
#SBATCH --partition=ai-h100l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/ucr_split_%A_%a.out
#SBATCH --error=logs/ucr_split_%A_%a.err
#SBATCH --array=0-20

# Optimized UCR experiment: 21 array tasks, 1 dataset each.
# Each task trains all models on its assigned dataset.
# With --gres=gpu:1, multiple tasks share nodes on multi-GPU partitions.

set -euo pipefail

REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
CONDA_PREFIX="/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-$(uname -m)"
ENV_NAME="ripsnet"
IDENTIFIER="try0"

export UCR_PATH="${UCR_PATH:-/hs/work0/home/users/${USER}/data/UCRArchive_2018}"

# Isolate checkpoints per task to avoid overwrites
export TFN_MODEL_TAG="task_${SLURM_ARRAY_TASK_ID}"

cd "$REPO_DIR/expes"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LD_LIBRARY_PATH="$CONDA_PREFIX/envs/$ENV_NAME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "=== $(date) Starting UCR fast: task $SLURM_ARRAY_TASK_ID ==="
echo "Host: $(hostname)  Arch: $(uname -m)"
python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

# 21 datasets, 1 per task (round-robin split)
python launch_expe.py ucr cluster generate train "$IDENTIFIER" 21 "$SLURM_ARRAY_TASK_ID"

echo "=== $(date) Done: task $SLURM_ARRAY_TASK_ID ==="
