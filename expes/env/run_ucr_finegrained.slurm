#!/bin/bash
#SBATCH --job-name=ripsnet-fine-gh200
#SBATCH --partition=qc-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/ucr_fine_%A_%a.out
#SBATCH --error=logs/ucr_fine_%A_%a.err
#SBATCH --array=180-269%16

# Rescue: b300 range (180-269, 90 tasks) on qc-gh200 (16 GH200 GPUs).

set -euo pipefail

# ---- Dataset list (order = DataSummary.csv row order, matches launch_expe.py modulo split) ----
DATASETS=(
  CBF ChlorineConcentration DistalPhalanxOutlineCorrect ECG200 ECG5000
  GunPoint ItalyPowerDemand MedicalImages MiddlePhalanxOutlineAgeGroup
  MiddlePhalanxOutlineCorrect MiddlePhalanxTW PhalangesOutlinesCorrect Plane
  ProximalPhalanxOutlineAgeGroup ProximalPhalanxTW SonyAIBORobotSurface1
  SonyAIBORobotSurface2 TwoLeadECG GunPointOldVersusYoung PowerCons UMD
)

# ---- Model list (order matches MODEL_NAMES in train_nn.py) ----
MODELS=(
  TensorFieldNetwork GTTensorFieldNetwork GTTensorFieldNetworkV2
  HierarchicalGTTFN HierarchicalTensorFieldNetwork
  OnEquivariantTensorFieldNetwork PointNet3D
  ScalarDistanceDeepSet PointNetTutorial ScalarInputMLP MultiInputModel
  RaggedPersistenceModel DistanceMatrixRaggedModel
  AttentionTensorFieldNetwork StochasticTensorFieldNetwork
  CrossAttentionTensorFieldNetwork
  RelaxedOnEquivariantTensorFieldNetwork
  HybridOnEquivariantTensorFieldNetwork
)

N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODELS[@]}

DATASET_ID=$(( SLURM_ARRAY_TASK_ID / N_MODELS ))
MODEL_ID=$(( SLURM_ARRAY_TASK_ID % N_MODELS ))

DATASET="${DATASETS[$DATASET_ID]}"
MODEL="${MODELS[$MODEL_ID]}"

export TFN_MODEL_NAME="$MODEL"
export TFN_MODEL_TAG="fine_${DATASET}_${MODEL}"
export UCR_PATH="${UCR_PATH:-/hs/work0/home/users/${USER}/data/UCRArchive_2018}"

REPO_DIR="/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-"
CONDA_PREFIX="/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-$(uname -m)"
ENV_NAME="ripsnet"

cd "$REPO_DIR/expes"
mkdir -p logs

trap 'rm -f "$REPO_DIR/expes/cache/geom_"*.pth' EXIT

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LD_LIBRARY_PATH="$CONDA_PREFIX/envs/$ENV_NAME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "=== $(date) Fine task $SLURM_ARRAY_TASK_ID: dataset=$DATASET model=$MODEL ==="
echo "Host: $(hostname)  Arch: $(uname -m)"
python -c "import torch; print('GPU:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

# Call launch_expe.py with 21-way split to get exactly this dataset
python launch_expe.py ucr cluster generate train "try0" 21 "$DATASET_ID"

echo "=== $(date) Done: $DATASET / $MODEL ==="
