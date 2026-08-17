#!/bin/bash
# Run shape classification experiments for all working models
# Usage: bash run_all_models.sh [dataset] [epochs] [trial]

DATASET=${1:-circles}
EPOCHS=${2:-50}
TRIAL=${3:-0}

echo "============================================"
echo "Shape classification: ${DATASET} (epochs=${EPOCHS}, trial=${TRIAL})"
echo "============================================"

for model in \
    PersNet \
    RipsPointNet \
    ScalarDistanceDeepSet \
    ScalarInputMLP \
    TensorFieldNetwork \
    GTTensorFieldNetworkV2; do
    echo ""
    echo ">>> $model"
    python train_shape.py "$DATASET" "$model" "$EPOCHS" "$TRIAL"
done

echo ""
echo "============================================"
echo "All models complete."
echo "============================================"
