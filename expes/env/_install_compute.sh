#!/usr/bin/env bash
# Run inside an RCC GPU allocation (qc-gh200 aarch64 or qc-a100 x86_64).
# Installs the ripsnet conda environment with PyTorch, Gudhi, and Velour.
set -euo pipefail

ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" ]]; then
  DEFAULT_PREFIX="${RIPSNET_CONDA_PREFIX:-/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-aarch64}"
elif [[ "$ARCH" == "x86_64" ]]; then
  DEFAULT_PREFIX="${RIPSNET_CONDA_PREFIX:-/hs/work0/home/users/${USER}/exp/ripsnet/miniforge3-x86_64}"
else
  echo "ERROR: unsupported architecture: $ARCH" >&2
  exit 2
fi

PREFIX="$DEFAULT_PREFIX"
ENV_NAME="${CONDA_ENV_NAME:-ripsnet}"
REPO_DIR="${RIPSNET_REPO_DIR:-/hs/work0/home/users/${USER}/exp/ripsnet/Ripsnet-}"

# Find the YAML
YAML="${REPO_DIR}/expes/env/ripsnet-env.yaml"
if [[ ! -f "$YAML" ]]; then
  echo "ERROR: $YAML not found – rsync repo first." >&2
  exit 3
fi

echo "### arch=$ARCH PREFIX=$PREFIX env=$ENV_NAME"

# Install Miniforge3 if needed
if [[ ! -x "$PREFIX/bin/python" ]]; then
  echo "### Installing Miniforge3 ($ARCH) -> $PREFIX"
  if [[ "$ARCH" == "aarch64" ]]; then
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  else
    URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  fi
  _tmp="/tmp/miniforge3-${ARCH}.$$.sh"
  /usr/bin/curl -fsSL -o "$_tmp" "$URL"
  bash "$_tmp" -b -p "$PREFIX"
  rm -f "$_tmp"
fi

# shellcheck source=/dev/null
source "$PREFIX/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -f "$YAML" -n "$ENV_NAME" --prune
else
  conda env create -f "$YAML" -n "$ENV_NAME"
fi

PY="$PREFIX/envs/$ENV_NAME/bin/python"
TORCH_VER="${RIPSNET_TORCH_VERSION:-2.5.1}"
TORCH_INDEX="${RIPSNET_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

"$PY" -m pip install -U pip
"$PY" -m pip install "torch==${TORCH_VER}" --index-url "$TORCH_INDEX"

# Install Velour from GitHub
"$PY" -m pip install git+https://github.com/raphaeltinarrage/velour.git

# Verify
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$PY" -c "import gudhi; print('gudhi', gudhi.__version__)"
"$PY" -c "import velour; print('velour ok')"
"$PY" -c "import sklearn, scipy, numpy, pandas, matplotlib, dill, xgboost; print('all deps ok')"

echo ""
echo "### Done ($ARCH). Python: $PY"
