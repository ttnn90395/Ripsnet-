#!/bin/bash
#SBATCH --job-name=tfn-sweep-h100
#SBATCH --partition=ai-h100l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/tfn_sweep_%A_%a.out
#SBATCH --error=logs/tfn_sweep_%A_%a.err
#SBATCH --array=0-4

# Sweep: train TensorFieldNetwork on UCR with different hyperparams.
# Each array task tests one hparam config on all 21 UCR datasets.
# Set TFN_HPARAMS via sweep_configs below.

set -euo pipefail

REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
CONDA_PREFIX="/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-$(uname -m)"
ENV_NAME="ripsnet"
export UCR_PATH="${UCR_PATH:-/hs/work0/home/users/${USER}/data/UCRArchive_2018}"

cd "$REPO_DIR/expes"
mkdir -p logs

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/envs/$ENV_NAME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== $(date) TFN sweep task $SLURM_ARRAY_TASK_ID ==="
echo "Host: $(hostname)  Arch: $(uname -m)"
python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------
# Each entry: name, json_args
configs=(
  'default   {}'
  'hC64      {"hidden_channels":64}'
  'hC128     {"hidden_channels":128}'
  'deep      {"num_layers":4,"hidden_channels":64}'
  'cosine    {"lr_schedule":"cosine","lr":0.005}'
)

readarray -t cfglines <<<"${configs[*]}"
line="${cfglines[$SLURM_ARRAY_TASK_ID]}"
cfg_name=$(echo "$line" | awk '{print $1}')
cfg_json=$(echo "$line" | cut -d' ' -f2-)

export TFN_HPARAMS="$cfg_json"
export TFN_MODEL_NAME="TensorFieldNetwork"
export TFN_MODEL_TAG="sweep_${cfg_name}"
echo "Config: $cfg_name  TFN_HPARAMS=$TFN_HPARAMS  MODEL=$TFN_MODEL_NAME  TAG=$TFN_MODEL_TAG"

# ---------------------------------------------------------------------------
# Run on all UCR datasets (clean + noise)
# ---------------------------------------------------------------------------
python launch_expe.py ucr cluster generate train "sweep_${cfg_name}"

echo "=== $(date) Done: $cfg_name ==="
