# RipsNet: Equivariant Neural Networks for Shape Classification

Comparative evaluation of Tensor Field Network variants, PointNet baselines, and DeepSet architectures on 2D and 3D point cloud classification tasks.

## Overview

This project benchmarks **12 neural network architectures** across **6 datasets** (2D circles and 3D synthetic shapes) with **3 trials each** (162 total experiments). The goal is to evaluate whether equivariant architectures (Tensor Field Networks) provide advantages over permutation-invariant baselines (PointNet, DeepSets) for geometric shape classification.

### Architectures

| Category | Model | Params | Description |
|----------|-------|--------|-------------|
| **Baseline** | PersNet | ~0.5M | Plain PointNet/Deep-Sets baseline on raw 3D coordinates (no persistence structure) |
| **Baseline** | RipsPointNet | ~0.8M | PointNet + persistence diagram fusion (H1 topology) |
| **Baseline** | ScalarInputMLP | ~50K | MLP on flattened point coordinates |
| **Baseline** | ScalarDistanceDeepSet | ~50K | DeepSet on pairwise distance features |
| **TFN** | TensorFieldNetwork | ~1M | SO(3)-equivariant TFN (kNN + GT basis) |
| **TFN** | GTTensorFieldNetworkV2 | ~1M | GT-TFN with improved radial networks |
| **TFN** | HierarchicalTensorFieldNetwork | ~1M | TFN with PointNet++-style hierarchical pooling |
| **TFN** | StochasticTensorFieldNetwork | ~1M | TFN with stochastic depth/regularization |
| **TFN** | OnEquivariantTensorFieldNetwork | ~1M | SO(n)→O(n) parity-equivariant wrapper |
| **TFN** | AttentionTensorFieldNetwork | ~0.6M | TFN with attention-based pooling |
| **TFN** | RelaxedOnEquivariantTensorFieldNetwork | ~1M | Relaxed parity constraints on O(n)-TFN |
| **TFN** | HybridOnEquivariantTensorFieldNetwork | ~1.2M | TFN + non-equivariant feature fusion |

### Datasets

| Dataset | Classes | Points | Description |
|---------|---------|--------|-------------|
| `circles` | 3 | 600 | 2D concentric circles (clean) |
| `circles_noisy` | 3 | 600 | 2D concentric circles with noise |
| `shapes3d_topology` | 4 | 600 | 3D shapes differing by topology (sphere, torus, etc.) |
| `shapes3d_geometry` | 4 | 600 | 3D shapes differing by geometry (radii, sizes) |
| `shapes3d_complex` | 6 | 600 | 3D shapes, complex mix of topological/geometric variation |
| `shapes3d_8way` | 8 | 600 | 8-class 3D shape classification |

## Project Structure

```
Ripsnet-/
├── models.py                 # All 18+ model architectures (central registry)
├── gt_tfn_layer.py           # SE(n)-equivariant TFN layer with kNN geometry
├── gt_basis.py               # Gelfand-Tsetlin basis for SO(n) representations
├── gt_improvements.py        # Hierarchical pooling, O(n) parity, FPS, hybrid models
├── TFN.py                    # Original SO(3) TFN implementation
├── tfn_model.py              # Model wrappers and ragged batching
├── tfn_enhancements.py       # MLP head, multi-scale persistence, augmentation
├── utils.py                  # DenseRagged, PermopRagged utilities
│
├── datasets/
│   ├── utils.py              # Circle generation + persistence diagram computation
│   └── shapes3d.py           # 3D shape generators (sphere, torus, cylinder, etc.)
│
├── shape/                    # Shape classification experiments
│   ├── train_shape.py        # Main training script (all models, all datasets)
│   ├── check_missing.py      # Audit which experiments are missing
│   ├── consolidate_results.py # Download + merge all results → CSV
│   ├── results_all.csv       # Per-trial results
│   ├── results_summary.csv   # Mean±std summary
│   ├── results/              # JSON result files per experiment
│   └── submit_*.sh           # SLURM job submission scripts
│
├── expes/                    # Original RipsNet experiments (time series → persistence)
│   ├── train_nn.py           # Training script for persistence-based pipeline
│   ├── analysis_nn.py        # Post-training analysis
│   ├── train_ablation.py     # Training data fraction ablation
│   ├── density_ablation.py   # Test-time density ablation
│   ├── isometry_ablation.py  # Rotation/translation robustness
│   └── results/              # UCR benchmark results
│
├── tutorial.ipynb            # (Legacy) Original RipsNet TF tutorial — see shape_classification.ipynb
├── tutorial_pytorch_ragged.ipynb  # (Legacy) Original RipsNet PyTorch tutorial
├── shape_classification.ipynb # Main results notebook: benchmark plots + analysis
└── comprehensive_model_testing.py  # All-model test harness
```

## Quick Start

### Shape Classification (main experiments)

```bash
# Run a single experiment
python shape/train_shape.py circles TensorFieldNetwork 50 0

# Check which results are missing
cd shape && python check_missing.py

# Consolidate all results into CSV
python shape/consolidate_results.py --no-download  # use local results/
python shape/consolidate_results.py                 # download from cluster first
```

### Original RipsNet Pipeline (time series → persistence → classification)

```bash
# Generate synthetic data and run all models
for i in 0 1 2 3 4 5 6 7 8 9; do
    python expes/launch_expe.py synth laptop generate train try$i
done
```

### Cluster Submission (Polytechnique)

```bash
# Submit all shape experiments to SLURM
cd shape
sbatch submit_resubmit.sh     # resubmit missing experiments
```

## Training Script Arguments

```bash
python shape/train_shape.py <dataset> <model> [num_epochs] <trial> [batch_size]
```

- `dataset`: circles, circles_noisy, shapes3d_topology, shapes3d_geometry, shapes3d_complex, shapes3d_8way
- `model`: Any model name from the table above (e.g., TensorFieldNetwork, PersNet)
- `num_epochs`: Training epochs (default: 50)
- `trial`: Random seed trial index (0, 1, or 2)
- `batch_size`: Batch size (default: 32)

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- Gudhi (persistence diagrams)
- tqdm

## Results

Results are stored in `shape/results/` as JSON files and consolidated into:
- `shape/results_all.csv` — per-trial raw results
- `shape/results_summary.csv` — mean±std across trials

Run `python shape/consolidate_results.py` to regenerate these files from the cluster.

## Cluster Infrastructure

- **Polytechnique** (dindon): `ten.nguyen-hanaoka@dindon.polytechnique.fr`, partition `SallesInfo`
  - CPU-only jobs (GPU not exposed via SLURM GRES)
  - 48GB RAM, 4 CPUs per job
  - 3-day max time limit

## Acknowledgments

Built on the RipsNet framework by Tinarrage et al. for persistence-vectorization of time series.
Extended with equivariant architectures (GT-TFN, O(n)-parity, hybrid fusion) and 3D shape benchmarks.
