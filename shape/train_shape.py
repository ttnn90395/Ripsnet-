"""
Shape Classification Training Script
=====================================
Trains and evaluates 12 neural network architectures on 6 point cloud
classification tasks (2D circles and 3D synthetic shapes).

Models:
    PointNet3D, RipsPointNet, ScalarInputMLP, ScalarDistanceDeepSet,
    TensorFieldNetwork, GTTensorFieldNetworkV2, HierarchicalTensorFieldNetwork,
    StochasticTensorFieldNetwork, OnEquivariantTensorFieldNetwork,
    AttentionTensorFieldNetwork, RelaxedOnEquivariantTensorFieldNetwork,
    HybridOnEquivariantTensorFieldNetwork

Datasets:
    circles, circles_noisy, shapes3d_topology, shapes3d_geometry,
    shapes3d_complex, shapes3d_8way

Usage:
    python train_shape.py <dataset> <model> <num_epochs> <trial> [batch_size]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm
import gc

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import (
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    HierarchicalGTTFN, HierarchicalTensorFieldNetwork,
    OnEquivariantTensorFieldNetwork, PointNet3D, RipsPointNet,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DenseRagged, PermopRagged, RaggedPersistenceModel, DistanceMatrixRaggedModel,
    AttentionTensorFieldNetwork, StochasticTensorFieldNetwork,
    CrossAttentionTensorFieldNetwork,
    RelaxedOnEquivariantTensorFieldNetwork,
    HybridOnEquivariantTensorFieldNetwork,
    _move_basis_tensors, compute_persistence_diagram,
)
from shape.robust_utils import (
    augment_with_outliers, filter_by_dtm, dtm_scores,
    AttentionPooling, PersistenceFusion,
    denoise_batch, StatisticalOutlierRemover, LocalMeanShiftDenoiser,
    GeometricRegularization, ModelEnsemble, FeatureLevelPDFusion,
)

from datasets.utils import (
    create_multiple_circles,
    create_1_circle_clean,
    create_2_circle_clean,
    create_3_circle_clean,
    create_1_circle_noisy,
    create_2_circle_noisy,
    create_3_circle_noisy,
    augment_isometries,
)
from datasets.shapes3d import generate_dataset, DATASET_CONFIGS

# -------------------------------------------------------------------------
# Model categories
# -------------------------------------------------------------------------
TFN_MODELS = {
    'TensorFieldNetwork', 'GTTensorFieldNetwork',
    'GTTensorFieldNetworkV2', 'HierarchicalGTTFN',
    'HierarchicalTensorFieldNetwork', 'OnEquivariantTensorFieldNetwork',
    'AttentionTensorFieldNetwork', 'StochasticTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork',
    'HybridOnEquivariantTensorFieldNetwork',
    'CrossAttentionTensorFieldNetwork',
}

def _ensure_output_dir(path: str) -> None:
    """Create *path* as a directory, or raise a clear error if a *file*
    sits in the way (e.g. on case-insensitive filesystems where the tracked
    `Results` file collides with the `results/` output directory)."""
    if os.path.isfile(path):
        raise FileExistsError(
            f"A file named {path!r} blocks the output directory. Remove or "
            f"rename the file (e.g. `mv {path} {path}.txt`) and retry."
        )
    os.makedirs(path, exist_ok=True)

_ensure_output_dir('results')
_ensure_output_dir('models')

# -------------------------------------------------------------------------
# CLI arguments
# -------------------------------------------------------------------------
dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'circles'
model_name   = sys.argv[2] if len(sys.argv) > 2 else 'PointNet3D'
num_epochs   = int(sys.argv[3]) if len(sys.argv) > 3 else 50
trial        = int(sys.argv[4]) if len(sys.argv) > 4 else 0
quick_mode   = '--quick' in sys.argv
batch_size   = 16 if quick_mode else 32
for a in sys.argv[5:]:
    if a.isdigit():
        batch_size = int(a)
        break

# Robustness improvements (flags)
use_noise_aug    = '--noise-aug' in sys.argv
use_dtm_filter   = '--dtm-filter' in sys.argv
use_attn_pool    = '--attn-pool' in sys.argv
use_multiscale   = '--multiscale' in sys.argv
use_pd_fusion    = '--pd-fusion' in sys.argv
use_denoise      = '--denoise' in sys.argv
use_geom_reg     = '--geom-reg' in sys.argv
use_feat_pd      = '--feat-pd' in sys.argv
use_robust_knn   = '--robust-knn' in sys.argv
use_train_robust_knn = '--train-robust-knn' in sys.argv
use_dtm_readout  = '--dtm-readout' in sys.argv
use_norm_readout = '--norm-readout' in sys.argv
readout_pool     = 'sum'
for a in sys.argv:
    if a.startswith('--pool='):
        readout_pool = a.split('=')[1]
noise_rate       = 0.2
noise_scale      = 2.0
aug_frac         = 1.0
denoise_method   = 'statistical'
dtm_keep         = 0.8
dtm_m            = 10
robust_alpha     = 1.0
robust_gamma     = 2.0
for a in sys.argv:
    if a.startswith('--noise-rate='):
        noise_rate = float(a.split('=')[1])
    if a.startswith('--noise-scale='):
        noise_scale = float(a.split('=')[1])
    if a.startswith('--aug-frac='):
        aug_frac = float(a.split('=')[1])
    if a.startswith('--denoise-method='):
        denoise_method = a.split('=')[1]
    if a.startswith('--dtm-keep='):
        dtm_keep = float(a.split('=')[1])
    if a.startswith('--dtm-m='):
        dtm_m = int(a.split('=')[1])
    if a.startswith('--robust-alpha='):
        robust_alpha = float(a.split('=')[1])
    if a.startswith('--dtm-gamma='):
        robust_gamma = float(a.split('=')[1])
best_on = 'clean'
for a in sys.argv:
    if a.startswith('--best-on='):
        best_on = a.split('=')[1]
        if best_on not in ('clean', 'noisy'):
            raise ValueError(f"--best-on must be 'clean' or 'noisy', got {best_on}")

print(f"shape/train_shape.py: {dataset_name} {model_name} epochs={num_epochs} trial={trial} bs={batch_size} quick={quick_mode} best_on={best_on}")
if any([use_noise_aug, use_dtm_filter, use_attn_pool, use_multiscale, use_pd_fusion,
        use_denoise, use_geom_reg, use_feat_pd, use_robust_knn, use_dtm_readout,
        use_train_robust_knn]):
    flags = []
    if use_noise_aug:  flags.append(f'noise_aug(r={noise_rate},s={noise_scale},frac={aug_frac})')
    if use_dtm_filter: flags.append(f'dtm_filter(keep={dtm_keep},m={dtm_m})')
    if use_robust_knn: flags.append(f'robust_knn(alpha={robust_alpha})')
    if use_train_robust_knn: flags.append(f'train_robust_knn(alpha={robust_alpha})')
    if use_dtm_readout: flags.append(f'dtm_readout(m={dtm_m},gamma={robust_gamma})')
    if readout_pool != 'sum': flags.append(f'pool={readout_pool}')
    if use_norm_readout: flags.append('norm_readout')
    if use_attn_pool:  flags.append('attn_pool')
    if use_multiscale: flags.append('multiscale')
    if use_pd_fusion:  flags.append('pd_fusion')
    if use_denoise:    flags.append(f'denoise({denoise_method})')
    if use_geom_reg:   flags.append('geom_reg')
    if use_feat_pd:    flags.append('feat_pd')
    print(f"  Robustness flags: {', '.join(flags)}")

# -------------------------------------------------------------------------
# Device
# -------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = device.type == "cuda"
print(f"Device: {device}  AMP: {USE_AMP}")

# -------------------------------------------------------------------------
# Data generation
# -------------------------------------------------------------------------
N_TRAIN = 300 if quick_mode else 900
N_TEST  = 100 if quick_mode else 300
N_POINTS = 100 if quick_mode else 600
N_NOISE  = 20 if quick_mode else 200

IS_3D = dataset_name.startswith('shapes3d')

print(f"Generating {dataset_name} data...")
if dataset_name == 'circles':
    data_train, label_train = create_multiple_circles(N_TRAIN, N_POINTS, noisy=False, N_noise=N_NOISE)
    data_test, label_test   = create_multiple_circles(N_TEST, N_POINTS, noisy=False, N_noise=N_NOISE)
    data_noisy, label_noisy = create_multiple_circles(N_TEST, N_POINTS, noisy=True, N_noise=N_NOISE)
    data_noisy_val, label_noisy_val = create_multiple_circles(N_TEST, N_POINTS, noisy=True, N_noise=N_NOISE)
elif dataset_name == 'circles_noisy':
    data_train, label_train = create_multiple_circles(N_TRAIN, N_POINTS, noisy=True, N_noise=N_NOISE)
    data_test, label_test   = create_multiple_circles(N_TEST, N_POINTS, noisy=True, N_noise=N_NOISE)
    data_noisy, label_noisy = data_test, label_test
    data_noisy_val, label_noisy_val = create_multiple_circles(N_TEST, N_POINTS, noisy=True, N_noise=N_NOISE)
elif IS_3D:
    noise_sigma_train = 0.05 if 'noisy' in dataset_name else 0.0
    noise_sigma_test  = 0.0
    base_name = dataset_name.replace('_noisy', '')
    n_per_class_train = N_TRAIN // len(DATASET_CONFIGS[base_name]['classes'])
    n_per_class_test  = N_TEST  // len(DATASET_CONFIGS[base_name]['classes'])
    data_train, y_train_raw, data_test, y_test_raw, class_names = generate_dataset(
        base_name, n_per_class_train, n_per_class_test, N_POINTS,
        noise_sigma=noise_sigma_train, seed=42)
    data_noisy, y_noisy_raw, _, _, _ = generate_dataset(
        base_name, n_per_class_test, 0, N_POINTS,
        noise_sigma=0.15, seed=123)
    label_train = y_train_raw
    label_test  = y_test_raw
    label_noisy = y_noisy_raw
    data_noisy_val, y_noisy_val_raw, _, _, _ = generate_dataset(
        base_name, n_per_class_test, 0, N_POINTS,
        noise_sigma=0.15, seed=456)
    label_noisy_val = y_noisy_val_raw
else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(label_train)
y_test  = le.transform(label_test)
y_noisy = le.transform(label_noisy)
y_noisy_val = le.transform(label_noisy_val)
num_classes = len(le.classes_)

print(f"  Train: {len(data_train)} samples, Test: {len(data_test)} samples")
print(f"  Classes: {num_classes}  Points: {N_POINTS}")

# -------------------------------------------------------------------------
# Convert to tensors
# -------------------------------------------------------------------------
data_train_t = [torch.FloatTensor(x).to(device) for x in data_train]
data_test_t  = [torch.FloatTensor(x).to(device) for x in data_test]
data_noisy_t = [torch.FloatTensor(x).to(device) for x in data_noisy]
data_noisy_val_t = [torch.FloatTensor(x).to(device) for x in data_noisy_val]

# -------------------------------------------------------------------------
# Model building (following expes patterns)
# -------------------------------------------------------------------------
_npts = N_POINTS
_dim  = 3 if IS_3D else 2

def build_model(name):
    """Build model following expes/train_ablation.py patterns."""
    _hp = {'max_order': 0, 'hidden_channels': 16, 'num_layers': 2,
           'classifier_dims': [32], 'num_rbf': 64, 'k_neighbors': 16}

    if name == 'PointNetTutorial':
        return PointNetTutorial(output_dim=num_classes)
    if name == 'PointNet3D':
        return PointNet3D(output_dim=num_classes, input_dim=_dim)
    if name == 'RipsPointNet':
        return RipsPointNet(output_dim=num_classes, input_dim=_dim)
    if name == 'DistanceMatrixRaggedModel':
        return DistanceMatrixRaggedModel(output_dim=num_classes, num_points=_npts)
    if name == 'RaggedPersistenceModel':
        return RaggedPersistenceModel(output_dim=num_classes)
    if name == 'ScalarDistanceDeepSet':
        return ScalarDistanceDeepSet(output_dim=num_classes)
    if name == 'ScalarInputMLP':
        return ScalarInputMLP(output_dim=num_classes)
    if name == 'MultiInputModel':
        return MultiInputModel(target_output_dim=num_classes, scalar_input_dim=1)
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=num_classes, n=_dim, **_hp)
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=_dim, num_classes=num_classes, radial_hidden=128, **_hp)
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=_dim, num_classes=num_classes, radial_hidden=128,
                                      use_attention_pool=use_attn_pool, **_hp)
    if name == 'HierarchicalGTTFN':
        return HierarchicalGTTFN(n=_dim, num_classes=num_classes,
            max_order=_hp['max_order'], hidden_channels=_hp['hidden_channels'],
            stage_sizes=[64, 32], num_rbf=_hp['num_rbf'],
            cutoff=1.0, classifier_dims=_hp['classifier_dims'])
    if name == 'HierarchicalTensorFieldNetwork':
        return HierarchicalTensorFieldNetwork(num_classes=num_classes, n=_dim,
            max_order=_hp['max_order'], hidden_channels=_hp['hidden_channels'],
            stage_sizes=[64, 32], num_rbf=_hp['num_rbf'],
            cutoff=1.0, classifier_dims=_hp['classifier_dims'])
    if name == 'OnEquivariantTensorFieldNetwork':
        return OnEquivariantTensorFieldNetwork(num_classes=num_classes, n=_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64,32])
    if name == 'AttentionTensorFieldNetwork':
        return AttentionTensorFieldNetwork(num_classes=num_classes, n=_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_heads=4,
            num_rbf=64, classifier_dims=[64,32], radial_hidden=64)
    if name == 'CrossAttentionTensorFieldNetwork':
        return CrossAttentionTensorFieldNetwork(num_classes=num_classes, n=_dim,
            max_order=1, hidden_channels=32, num_layers=2, num_heads=4,
            transformer_layers=2, num_rbf=64, classifier_dims=[64,32], radial_hidden=64)
    if name == 'StochasticTensorFieldNetwork':
        return StochasticTensorFieldNetwork(num_classes=num_classes, n=_dim,
            num_mixtures=3, max_order=0, hidden_channels=16, num_layers=2,
            num_rbf=64, encoder_dims=[64,32])
    if name == 'RelaxedOnEquivariantTensorFieldNetwork':
        return RelaxedOnEquivariantTensorFieldNetwork(num_classes=num_classes, n=_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64,32])
    if name == 'HybridOnEquivariantTensorFieldNetwork':
        return HybridOnEquivariantTensorFieldNetwork(num_classes=num_classes, n=_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64,32], non_eq_dim=128)
    raise ValueError(f"Unknown model: {name}")

model = build_model(model_name).to(device)
print(f"  Model: {model_name} ({sum(p.numel() for p in model.parameters())} params)")

# -------------------------------------------------------------------------
# Data preparation (following expes patterns)
# -------------------------------------------------------------------------
def prepare_data(data_list, name):
    """Prepare data in the format expected by each model."""
    if name in TFN_MODELS:
        return data_list
    if name == 'PointNet3D':
        return data_list
    if name == 'PointNetTutorial':
        return data_list
    if name == 'RipsPointNet':
        pd_list = []
        for x in tqdm(data_list, desc='Computing persistence diagrams'):
            pd = compute_persistence_diagram(x)
            pd_list.append(pd.to(device))
        return list(zip(data_list, pd_list))
    if name in ('ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel'):
        out = []
        for x in data_list:
            a = x.cpu().numpy()
            m = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
            out.append(torch.FloatTensor(m).to(device))
        return out
    if name == 'ScalarInputMLP':
        out = []
        for x in data_list:
            a = x.cpu().numpy()
            m = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
            out.append(torch.FloatTensor([[m.mean()]]).to(device))
        return out
    if name == 'MultiInputModel':
        out = []
        for x in data_list:
            a = x.cpu().numpy()
            m = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
            out.append((x, torch.FloatTensor([[m.mean()]]).to(device)))
        return out
    return data_list

# -------------------------------------------------------------------------
# Input denoising (improvement 6)
# -------------------------------------------------------------------------
if use_denoise:
    print(f"Applying input denoising ({denoise_method})...")
    data_train_t = denoise_batch(data_train_t, method=denoise_method, k=10, alpha=2.0)
    data_test_t  = denoise_batch(data_test_t, method=denoise_method, k=10, alpha=2.0)
    data_noisy_t = denoise_batch(data_noisy_t, method=denoise_method, k=10, alpha=2.0)
    print(f"  After denoising: train={len(data_train_t)} test={len(data_test_t)} noisy={len(data_noisy_t)}")

# -------------------------------------------------------------------------
# DTM filtering (improvement 3)
# Outlier points are removed from the noisy evaluation set only; the model is
# trained on the full clean clouds.  This restores TFN k-NN geometry (which
# collapses to chance accuracy when outlier points corrupt the neighborhood)
# at test time without shifting the training distribution.
# -------------------------------------------------------------------------
if use_dtm_filter:
    print(f"Applying DTM pre-filtering (keep={dtm_keep}, m={dtm_m}) to noisy eval data...")
    data_noisy_t = filter_by_dtm(data_noisy_t, keep_ratio=dtm_keep, m=dtm_m)
    data_noisy_val_t = filter_by_dtm(data_noisy_val_t, keep_ratio=dtm_keep, m=dtm_m)
    print(f"  After filtering: noisy={len(data_noisy_t)} clouds")

# Initialize PD lists (used by pd-fusion, feat-pd, and RipsPointNet)
train_pd = None
test_pd  = None
noisy_pd = None

train_in = prepare_data(data_train_t, model_name)
test_in  = prepare_data(data_test_t, model_name)
noisy_in = prepare_data(data_noisy_t, model_name)
noisy_val_in = prepare_data(data_noisy_val_t, model_name)

# -------------------------------------------------------------------------
# PD fusion module (improvement 5)
# -------------------------------------------------------------------------
pd_fusion = None
if use_pd_fusion:
    print("Setting up persistence diagram fusion branch...")
    pd_fusion = PersistenceFusion(output_dim=128).to(device)
    # Precompute PDs for all data
    print("Precomputing persistence diagrams...")
    train_pd = [compute_persistence_diagram(x) for x in tqdm(data_train, desc='Train PDs')]
    test_pd  = [compute_persistence_diagram(x) for x in tqdm(data_test, desc='Test PDs')]
    noisy_pd = [compute_persistence_diagram(x) for x in tqdm(data_noisy, desc='Noisy PDs')]
    # Move to device
    train_pd = [p.to(device) for p in train_pd]
    test_pd  = [p.to(device) for p in test_pd]
    noisy_pd = [p.to(device) for p in noisy_pd]

# -------------------------------------------------------------------------
# Feature-level PD fusion (improvement 9)
# -------------------------------------------------------------------------
feat_pd_fusion = None
if use_feat_pd:
    print("Setting up feature-level PD fusion...")
    # Precompute PDs if not already done
    if train_pd is None:
        print("Precomputing persistence diagrams...")
        train_pd = [p.to(device) for p in [compute_persistence_diagram(x) for x in tqdm(data_train, desc='Train PDs')]]
        test_pd  = [p.to(device) for p in [compute_persistence_diagram(x) for x in tqdm(data_test, desc='Test PDs')]]
        noisy_pd = [p.to(device) for p in [compute_persistence_diagram(x) for x in tqdm(data_noisy, desc='Noisy PDs')]]
    # geo_dim must be divisible by n_heads; pick nearest multiple of 4 >= num_classes
    _nheads = 4
    _geo_dim = max(num_classes, _nheads) * 2
    _geo_dim = (_geo_dim // _nheads + 1) * _nheads  # round up to multiple of n_heads
    feat_pd_fusion = FeatureLevelPDFusion(
        geo_dim=_geo_dim, input_dim=num_classes, pd_dim=128, hidden_dim=64, n_heads=_nheads
    ).to(device)
    print(f"  Feature-level PD fusion: geo_dim={_geo_dim}, pd_dim=128")

# -------------------------------------------------------------------------
# TFN geometry precomputation (following expes patterns)
# -------------------------------------------------------------------------
def _unwrap_tfn(m):
    """Find the model with k_neighbors/rbf/gt_basis."""
    for _ in range(5):
        if hasattr(m, 'k_neighbors') and hasattr(m, 'rbf') and hasattr(m, 'gt_basis'):
            return m
        child = getattr(m, '_inner', None) or getattr(m, 'base', None) or getattr(m, 'tfn_backbone', None)
        if child is None or child is m:
            break
        m = child
    return m

def precompute_geom(model, data_list):
    """Precompute k-NN geometry for TFN models."""
    if model_name not in TFN_MODELS:
        return None
    try:
        from gt_tfn_layer import knn_geometry
        inner = _unwrap_tfn(model)
        _move_basis_tensors(inner, device)
        k = getattr(inner, 'k_neighbors', 16)
        rbfs, gts, nbrs = [], [], []
        for pc in tqdm(data_list, desc="Precomputing geometry", leave=False):
            r, g, n = knn_geometry(pc, inner.rbf, inner.gt_basis, k,
                                   robust_alpha=robust_alpha if use_robust_knn else 0.0,
                                   robust_m=dtm_m)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            rbfs.append(r.detach())
            gts.append(g.detach())
            nbrs.append(n.detach())
        if all(r.shape == rbfs[0].shape for r in rbfs):
            return {'rbf': torch.stack(rbfs), 'gt_edge': torch.stack(gts),
                    'nbr_idx': torch.stack(nbrs), 'uniform': True}
        return {'list': list(zip(rbfs, gts, nbrs)), 'uniform': False}
    except Exception as e:
        print(f"  precompute_geom failed: {e}")
        return None

print("Precomputing geometry...")
train_geom = precompute_geom(model, train_in)
test_geom  = precompute_geom(model, test_in)
noisy_geom = precompute_geom(model, noisy_in)

if use_dtm_readout:
    inner = _unwrap_tfn(model)
    inner.robust_readout = False
    inner.robust_m       = dtm_m
    inner.robust_gamma   = robust_gamma
    print(f"  DTM-robust readout enabled for final inference (m={dtm_m}, gamma={robust_gamma}); OFF during training")
if readout_pool != 'sum':
    inner = _unwrap_tfn(model)
    inner.readout_pool = readout_pool
    print(f"  Readout pooling: {readout_pool}")
if use_norm_readout:
    inner = _unwrap_tfn(model)
    inner.norm_readout = True
    print("  Per-point L2-normalized readout ON")

# -------------------------------------------------------------------------
# Forward functions (following expes patterns)
# -------------------------------------------------------------------------
def _is_hybrid(m):
    inner = getattr(m, '_inner', None)
    return inner is not None and hasattr(inner, 'neq_phi')

def _find_encoder(m):
    for _ in range(5):
        if (hasattr(m, '_encode_single') or hasattr(m, '_encode_batch')) and hasattr(m, 'rho'):
            return m
        child = getattr(m, '_inner', None) or getattr(m, 'base', None)
        if child is None or child is m:
            break
        m = child
    return m

def get_geom(g, bix):
    if g is None:
        return None
    if isinstance(bix, range):
        bix = list(bix)
    if g.get('uniform', False):
        return {k: g[k][bix] if k in ('rbf', 'gt_edge', 'nbr_idx') else g[k]
                for k in g}
    return [g['list'][i] for i in bix]

def forward_batch(model, batch_data, mname, geom=None, pd_batch=None):
    """Forward a batch through the model, with optional PD fusion."""
    if mname == 'MultiInputModel':
        logits = model([x[0] for x in batch_data],
                     torch.cat([x[1] for x in batch_data]))
    elif mname == 'RipsPointNet':
        pcs = [x[0] for x in batch_data]
        pds = [x[1] for x in batch_data]
        logits = model(pcs, precomputed_pd=pds)
    elif mname == 'ScalarInputMLP':
        logits = model(torch.cat([x.reshape(1, -1) for x in batch_data]))
    elif mname == 'CrossAttentionTensorFieldNetwork':
        geom_list = None
        if geom is not None:
            if isinstance(geom, dict) and geom.get('uniform', False):
                geom_list = list(zip(geom['rbf'], geom['gt_edge'], geom['nbr_idx']))
            elif isinstance(geom, dict):
                geom_list = geom['list']
            else:
                geom_list = geom
        logits = model(batch_data, precomputed_geom=geom_list)
    elif geom is not None and mname in TFN_MODELS:
        if _is_hybrid(model):
            hybrid_inner = _find_encoder(model)
            tfn_inner = getattr(hybrid_inner, 'tfn_backbone', hybrid_inner)
            tfn_enc = getattr(tfn_inner, '_inner', tfn_inner)
            _move_basis_tensors(tfn_enc, device)
            if isinstance(geom, dict) and geom.get('uniform', False):
                geom_list = list(zip(geom['rbf'], geom['gt_edge'], geom['nbr_idx']))
            elif isinstance(geom, dict):
                geom_list = geom['list']
            else:
                geom_list = geom
            descs = []
            for x, (r, g, n) in zip(batch_data, geom_list):
                r = r.squeeze(0) if r.ndim == 4 else r
                g = g.squeeze(0) if g.ndim == 4 else g
                n = n.squeeze(0) if n.ndim == 3 else n
                descs.append(tfn_enc._encode_single(x, precomputed_geom=(r, g, n)))
            eq_feats = torch.stack(descs)
            input_dim = getattr(hybrid_inner, '_input_dim', 3)
            neq_feats = []
            for pc in batch_data:
                raw = pc[:, :input_dim] if pc.shape[1] >= input_dim else pc
                h = hybrid_inner.neq_phi(raw)
                h = h.max(dim=0).values
                neq_feats.append(h)
            neq_feats = torch.stack(neq_feats)
            combined = torch.cat([eq_feats, neq_feats], dim=1)
            logits = hybrid_inner.fusion(combined)
        else:
            inner = _find_encoder(model)
            _move_basis_tensors(inner, device)
            if isinstance(geom, dict) and geom.get('uniform', False):
                if hasattr(inner, '_encode_batch'):
                    logits = inner._encode_batch(torch.stack(batch_data),
                        precomputed_geom=(geom['rbf'], geom['gt_edge'], geom['nbr_idx']))
                else:
                    geom_list = list(zip(geom['rbf'], geom['gt_edge'], geom['nbr_idx']))
                    descs = []
                    for x, (r, g, n) in zip(batch_data, geom_list):
                        r = r.squeeze(0) if r.ndim == 4 else r
                        g = g.squeeze(0) if g.ndim == 4 else g
                        n = n.squeeze(0) if n.ndim == 3 else n
                        descs.append(inner._encode_single(x, precomputed_geom=(r, g, n)))
                    logits = inner.rho(torch.stack(descs))
            elif isinstance(geom, dict):
                geom_list = geom['list']
                descs = []
                for x, (r, g, n) in zip(batch_data, geom_list):
                    r = r.squeeze(0) if r.ndim == 4 else r
                    g = g.squeeze(0) if g.ndim == 4 else g
                    n = n.squeeze(0) if n.ndim == 3 else n
                    descs.append(inner._encode_single(x, precomputed_geom=(r, g, n)))
                logits = inner.rho(torch.stack(descs))
            else:
                geom_list = geom
                descs = []
                for x, (r, g, n) in zip(batch_data, geom_list):
                    r = r.squeeze(0) if r.ndim == 4 else r
                    g = g.squeeze(0) if g.ndim == 4 else g
                    n = n.squeeze(0) if n.ndim == 3 else n
                    descs.append(inner._encode_single(x, precomputed_geom=(r, g, n)))
                logits = inner.rho(torch.stack(descs))
    else:
        logits = model(batch_data)

    # PD fusion: concatenate PD features and re-classify
    if pd_batch is not None and pd_fusion is not None:
        pd_feats = pd_fusion(pd_batch)  # (B, 128)
        B = logits.shape[0]
        combined = torch.cat([logits, pd_feats], dim=1)  # (B, num_classes + 128)
        logits = pd_fusion_fuse(combined)  # (B, num_classes)

    # Feature-level PD fusion: cross-attention between logits and PD features
    if pd_batch is not None and feat_pd_fusion is not None:
        logits = feat_pd_fusion(logits, pd_batch)

    return logits

# -------------------------------------------------------------------------
# Training loop (following expes patterns)
# -------------------------------------------------------------------------
# PD fusion head: re-classify from (model_logits, pd_features)
pd_fusion_fuse = None
if pd_fusion is not None:
    # Get num_classes from model output
    _test_out = forward_batch(model, [train_in[0]], model_name,
                              geom=get_geom(train_geom, [0]))
    _logit_dim = _test_out.shape[1]
    pd_fusion_fuse = nn.Sequential(
        nn.Linear(_logit_dim + 128, 128), nn.GELU(), nn.LayerNorm(128),
        nn.Linear(128, num_classes),
    ).to(device)
    print(f"  PD fusion head: {_logit_dim}+128 → 128 → {num_classes}")

# Collect all trainable parameters
params = list(model.parameters())
if pd_fusion is not None:
    params += list(pd_fusion.parameters()) + list(pd_fusion_fuse.parameters())
if feat_pd_fusion is not None:
    params += list(feat_pd_fusion.parameters())

optimizer = optim.Adam(params, lr=5e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
scaler = amp.GradScaler(enabled=USE_AMP)

# Geometric regularization (improvement 7)
geom_reg = None
if use_geom_reg:
    print("Setting up geometric regularization (noise_scale=0.01, kl_weight=1.0)")
    geom_reg = GeometricRegularization(noise_scale=0.01, kl_weight=1.0)

best_val_acc = 0.0
best_model_state = None
best_epoch = 0
patience = 15
patience_counter = 0

print(f"\nTraining {model_name} for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    perm = np.random.permutation(len(train_in))
    epoch_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, len(train_in), batch_size):
        idx = perm[start:start + batch_size]
        batch_data = [train_in[i] for i in idx]
        batch_targets = torch.LongTensor(y_train[idx]).to(device)
        batch_geom = get_geom(train_geom, idx)

        # Noise augmentation: inject outlier points during training.
        # Non-TFN models consume the augmented points directly.  TFN models
        # were previously excluded because their k-NN geometry is precomputed
        # offline on clean clouds and goes stale once points are corrupted;
        # they are now augmented too, with geometry recomputed on the fly for
        # the corrupted batch so the model learns robust representations
        # instead of collapsing on noisy evaluation sets.
        # `aug_frac` < 1.0 keeps a fraction of the batch clean so the model
        # does not forget the clean-cloud representation while learning
        # robustness to corruption.
        if use_noise_aug and aug_frac > 0:
            n_aug = int(round(aug_frac * len(batch_data)))
            n_aug = max(1, n_aug)
            rng_sel = np.random.RandomState(epoch * 1000 + start)
            aug_idx = set(rng_sel.choice(len(batch_data), n_aug, replace=False).tolist())
            aug_data = augment_with_outliers([batch_data[i] for i in aug_idx],
                                             corruption_rate=noise_rate,
                                             scale=noise_scale, seed=epoch * 1000 + start)
            for j, i in enumerate(aug_idx):
                batch_data[i] = aug_data[j]
            if model_name in TFN_MODELS:
                from gt_tfn_layer import knn_geometry as _knn
                inner = _find_encoder(model)
                _move_basis_tensors(inner, device)
                k = getattr(inner, 'k_neighbors', 16)
                geom_items = []
                for i in range(len(batch_data)):
                    if i in aug_idx:
                        r, g, n = _knn(batch_data[i], inner.rbf, inner.gt_basis, k,
                                       robust_alpha=robust_alpha if (use_robust_knn or use_train_robust_knn) else 0.0,
                                       robust_m=dtm_m)
                        geom_items.append((r.detach(), g.detach(), n.detach()))
                    else:
                        if isinstance(batch_geom, dict):
                            geom_items.append((batch_geom['rbf'][i], batch_geom['gt_edge'][i], batch_geom['nbr_idx'][i]))
                        else:
                            geom_items.append(batch_geom[i])
                batch_geom = geom_items

        # PD batch for fusion (pd-fusion or feat-pd)
        pd_batch = None
        if pd_fusion is not None or feat_pd_fusion is not None:
            pd_batch = [train_pd[i] for i in idx]

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type=device.type, enabled=USE_AMP):
            output = forward_batch(model, batch_data, model_name,
                                   geom=batch_geom, pd_batch=pd_batch)

            if geom_reg is not None:
                total_loss, ce_loss, reg_loss = geom_reg(
                    model,
                    lambda m, d, **kw: forward_batch(m, d, model_name, **kw),
                    batch_data, batch_geom=batch_geom,
                    batch_targets=batch_targets, criterion=criterion,
                    pd_batch=pd_batch)
                loss = total_loss
            else:
                loss = criterion(output, batch_targets)

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += loss.item() * len(batch_data)
        correct += output.argmax(1).eq(batch_targets).sum().item()
        total += len(batch_targets)

    train_loss = epoch_loss / len(train_in)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for start in range(0, len(test_in), batch_size):
            end = min(start + batch_size, len(test_in))
            batch_data = test_in[start:end]
            batch_geom = get_geom(test_geom, list(range(start, end)))
            pd_batch = [test_pd[i] for i in range(start, end)] if (pd_fusion is not None or feat_pd_fusion is not None) else None
            out = forward_batch(model, batch_data, model_name, geom=batch_geom, pd_batch=pd_batch)
            val_preds.append(out.argmax(1).cpu())
    val_acc = 100 * accuracy_score(y_test, torch.cat(val_preds).numpy())

    # Noisy-domain validation (model selection target when --best-on=noisy).
    # Geometry is computed on the fly (no large stacked tensors); the split is
    # separate from the final noisy test set so selection does not leak.
    nval_preds = []
    with torch.no_grad():
        for start in range(0, len(noisy_val_in), batch_size):
            end = min(start + batch_size, len(noisy_val_in))
            out = forward_batch(model, noisy_val_in[start:end], model_name, geom=None)
            nval_preds.append(out.argmax(1).cpu())
    nval_acc = 100 * accuracy_score(y_noisy_val, torch.cat(nval_preds).numpy())

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{num_epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%  noisy_val={nval_acc:.1f}%")

    select_acc = nval_acc if best_on == 'noisy' else val_acc
    if select_acc > best_val_acc:
        best_val_acc = select_acc
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

# Restore best model (selected by clean or noisy-domain validation)
if best_model_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    print(f"  Restored best {best_on}-val model (epoch {best_epoch}, {best_on}_val_acc={best_val_acc:.2f}%)")

# Enable DTM-robust readout for final inference (trained unweighted on clean
# clouds; the soft DTM mask suppresses outlier pollution at inference time).
if use_dtm_readout:
    _unwrap_tfn(model).robust_readout = True
    print("  DTM-robust readout ENABLED for final evaluation")

# -------------------------------------------------------------------------
# Final evaluation
# -------------------------------------------------------------------------
print("\nFinal evaluation...")
model.eval()

# Clean test
test_preds = []
with torch.no_grad():
    for start in range(0, len(test_in), batch_size):
        end = min(start + batch_size, len(test_in))
        batch_data = test_in[start:end]
        batch_geom = get_geom(test_geom, list(range(start, end)))
        pd_batch = [test_pd[i] for i in range(start, end)] if (pd_fusion is not None or feat_pd_fusion is not None) else None
        out = forward_batch(model, batch_data, model_name, geom=batch_geom, pd_batch=pd_batch)
        test_preds.append(out.argmax(1).cpu())
clean_acc = accuracy_score(y_test, torch.cat(test_preds).numpy())

# Noisy test
noisy_preds = []
with torch.no_grad():
    for start in range(0, len(noisy_in), batch_size):
        end = min(start + batch_size, len(noisy_in))
        batch_data = noisy_in[start:end]
        batch_geom = get_geom(noisy_geom, list(range(start, end)))
        pd_batch = [noisy_pd[i] for i in range(start, end)] if (pd_fusion is not None or feat_pd_fusion is not None) else None
        out = forward_batch(model, batch_data, model_name, geom=batch_geom, pd_batch=pd_batch)
        noisy_preds.append(out.argmax(1).cpu())
noisy_acc = accuracy_score(y_noisy, torch.cat(noisy_preds).numpy())

print(f"\n{'='*60}")
print(f"Results: {model_name} on {dataset_name}")
print(f"{'='*60}")
print(f"  Clean test accuracy: {clean_acc:.4f} ({100*clean_acc:.2f}%)")
print(f"  Noisy test accuracy: {noisy_acc:.4f} ({100*noisy_acc:.2f}%)")
print(f"  Best val accuracy:   {best_val_acc:.2f}%")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
if IS_3D:
    base = dataset_name.replace('_noisy', '')
    print(f"  Classes: {DATASET_CONFIGS[base]['classes']}")
    print(f"  {DATASET_CONFIGS[base]['description']}")

# -------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------
result = {
    'dataset': dataset_name,
    'model': model_name,
    'trial': trial,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'clean_accuracy': float(clean_acc),
    'noisy_accuracy': float(noisy_acc),
    'best_val_accuracy': float(best_val_acc),
    'num_classes': num_classes,
    'n_train': len(data_train),
    'n_test': len(data_test),
    'n_points': N_POINTS,
    'params': sum(p.numel() for p in model.parameters()),
    'noise_aug': use_noise_aug,
    'noise_rate': noise_rate if use_noise_aug else None,
    'noise_scale': noise_scale if use_noise_aug else None,
    'aug_frac': aug_frac if use_noise_aug else None,
    'dtm_filter': use_dtm_filter,
    'dtm_keep': dtm_keep if use_dtm_filter else None,
    'dtm_m': dtm_m if use_dtm_filter else None,
    'robust_knn': use_robust_knn,
    'robust_alpha': robust_alpha if use_robust_knn else None,
    'train_robust_knn': use_train_robust_knn,
    'dtm_readout': use_dtm_readout,
    'dtm_readout_m': dtm_m if use_dtm_readout else None,
    'dtm_readout_gamma': robust_gamma if use_dtm_readout else None,
    'readout_pool': readout_pool,
    'norm_readout': use_norm_readout,
    'attn_pool': use_attn_pool,
    'multiscale': use_multiscale,
    'pd_fusion': use_pd_fusion,
    'denoise': use_denoise,
    'geom_reg': use_geom_reg,
    'feat_pd': use_feat_pd,
}

flag_suffix = ""
if use_attn_pool:
    flag_suffix += "_attn-pool"
if use_noise_aug:
    flag_suffix += "_noise-aug"
    if aug_frac != 1.0:
        flag_suffix += f"-frac{aug_frac}"
if use_dtm_filter:
    flag_suffix += "_dtm-filter"
if use_multiscale:
    flag_suffix += "_multiscale"
if use_pd_fusion:
    flag_suffix += "_pd-fusion"
if use_denoise:
    flag_suffix += f"_denoise-{denoise_method}"
if use_geom_reg:
    flag_suffix += "_geom-reg"
if use_feat_pd:
    flag_suffix += "_feat-pd"
if use_robust_knn:
    flag_suffix += "_robust-knn"
if use_train_robust_knn:
    flag_suffix += "_train-robust-knn"
if use_dtm_readout:
    flag_suffix += "_dtm-readout"
if readout_pool != 'sum':
    flag_suffix += f"_pool-{readout_pool}"
if use_norm_readout:
    flag_suffix += "_norm-readout"

out_path = f"results/shape_{dataset_name}_{model_name}_t{trial}{flag_suffix}.json"
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}")

# Save model checkpoint
ckpt_path = f"models/shape_{dataset_name}_{model_name}_t{trial}{flag_suffix}.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'result': result,
}, ckpt_path)
print(f"Model saved to {ckpt_path}")
