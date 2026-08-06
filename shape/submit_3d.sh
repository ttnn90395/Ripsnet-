#!/bin/bash
# Submit 3D shape classification experiments to RIKEN cluster
# Usage: bash submit_3d.sh [epochs] [trials]
set -euo pipefail

REMOTE="u0001943@login.cloud.r-ccs.riken.jp"
REMOTE_DIR="/hs/work0/home/users/u0001943/exp/ripsnet/Ripsnet-/shape"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EPOCHS=${1:-100}
TRIALS=${2:-3}

MODELS=(
    PointNet3D
    ScalarDistanceDeepSet
    ScalarInputMLP
    TensorFieldNetwork
    GTTensorFieldNetworkV2
    HierarchicalGTTFN
    OnEquivariantTensorFieldNetwork
    AttentionTensorFieldNetwork
    CrossAttentionTensorFieldNetwork
    StochasticTensorFieldNetwork
    RelaxedOnEquivariantTensorFieldNetwork
    HybridOnEquivariantTensorFieldNetwork
)

# 3D dataset configs:
# shapes3d_topology: sphere, torus, double_torus, two_spheres  (4 classes, topology)
# shapes3d_geometry: sphere, torus, cylinder, cube  (4 classes, geometry)
# shapes3d_complex: 6 classes (sphere, torus, double_torus, two_spheres, knot, tetrahedron)
# shapes3d_8way: all 8 shapes
DATASETS=(
    shapes3d_topology
    shapes3d_geometry
    shapes3d_complex
    shapes3d_8way
)

PARTITIONS=("ai-l40s" "ai-h200-brc" "qc-a100")

# Copy scripts + shapes3d.py
echo "=== Copying scripts to cluster ==="
scp "$SCRIPT_DIR/train_shape.py" "$REMOTE:$REMOTE_DIR/"
scp "$SCRIPT_DIR/../datasets/shapes3d.py" "$REMOTE:$REMOTE_DIR/../datasets/"
scp "$SCRIPT_DIR/run_all_models.sh" "$REMOTE:$REMOTE_DIR/"

# Submit jobs
echo "=== Submitting jobs ==="
ssh "$REMOTE" bash << JOBS
cd "$REMOTE_DIR"
mkdir -p results models logs

EPOCHS=$EPOCHS
TRIALS=$TRIALS
MODELS=(${MODELS[*]})
DATASETS=(${DATASETS[*]})
PARTITIONS=(${PARTITIONS[*]})

total=0
submitted=0
for ds in "\${DATASETS[@]}"; do
    for model in "\${MODELS[@]}"; do
        for ((t=0; t<TRIALS; t++)); do
            pi=$(( total % \${#PARTITIONS[@]} ))
            PART="\${PARTITIONS[\$pi]}"
            jname="shape_\${ds}_\${model}_t\${t}"
            sbatch --job-name="\$jname" --partition="\$PART" \
                --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
                --time=03:00:00 \
                --output="logs/\${jname}.out" \
                --error="logs/\${jname}.err" \
                --wrap="cd $REMOTE_DIR && python train_shape.py \$ds \$model \$EPOCHS \$t" 2>&1 | grep -q "Submitted batch" && submitted=\$(( submitted + 1 ))
            total=\$(( total + 1 ))
        done
    done
done
echo "Submitted \$submitted/\$total jobs"
JOBS

echo ""
echo "=== Done ==="
echo "Monitor: ssh $REMOTE 'squeue -u u0001943 | grep shape'"
