#!/bin/bash
set -e
export PATH=/hs/work0/home/users/u0001943/miniforge3_aarch64/bin:$PATH
SP=/hs/work0/home/users/u0001943/miniforge3_aarch64/lib/python3.13/site-packages
echo "=== 0. Remove partial install remnants ==="
rm -rf "$SP"/~* 2>/dev/null
echo "=== 1. Force reinstall torch+cu126 ==="
pip install torch torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu126 2>&1
echo "=== 2. Pin nccl for torch compat ==="
pip install nvidia-nccl-cu12==2.29.3 2>&1
echo "=== 3. Install remaining pkgs ==="
pip install scikit-learn dill xgboost --force-reinstall 2>&1
echo "=== 4. Verify ==="
python3 -c "
import torch; print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
import numpy, sklearn, xgboost, dill
print('numpy:', numpy.__version__)
print('sklearn:', sklearn.__version__)
print('xgboost:', xgboost.__version__)
print('dill:', dill.__version__)
"
echo "FIX_ENV_DONE"
