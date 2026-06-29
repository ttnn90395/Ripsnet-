#!/bin/bash
#SBATCH --job-name=ripsnet-ucr-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --output=logs/small_%x_%j_%a.out
#SBATCH --error=logs/small_%x_%j_%a.err
#SBATCH --array=0-20

set -euo pipefail

REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
CONDA_PREFIX="/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-$(uname -m)"
ENV_NAME="ripsnet"
IDENTIFIER="try0"

export UCR_PATH="${UCR_PATH:-/hs/work0/home/users/${USER}/data/UCRArchive_2018}"

# Smaller model hyperparams
export TFN_HPARAMS='{"hidden_channels":16,"k_neighbors":8,"num_layers":2,"num_rbf":32,"max_order":0,"classifier_dims":[32,16]}'

cd "$REPO_DIR/expes"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LD_LIBRARY_PATH="$CONDA_PREFIX/envs/$ENV_NAME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "=== $(date) Starting small UCR split: task $SLURM_ARRAY_TASK_ID ==="
echo "Host: $(hostname)  Arch: $(uname -m)  Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "TFN_HPARAMS: $TFN_HPARAMS"
python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

python launch_expe.py ucr cluster generate train "$IDENTIFIER" 21 "$SLURM_ARRAY_TASK_ID"

echo "=== $(date) Done: task $SLURM_ARRAY_TASK_ID ==="
