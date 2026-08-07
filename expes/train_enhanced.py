"""
Enhanced TFN Training Script
============================
Improved training with:
  - MLP head (end-to-end) OR XGBoost (baseline comparison)
  - Multi-scale persistence diagrams
  - Data augmentation
  - Configurable hidden dimensions
  - Attention pooling

Usage:
    python train_enhanced.py <dataset> <model> <fraction_pct> <trial> <epochs> <identifier> [options]

Options:
    --classifier xgboost|mlp     (default: mlp; xgboost for baseline comparison)
    --augment                    Enable data augmentation
    --multi-scale                Use multi-scale persistence
    --scale-factor FLOAT         Scale factor for multi-scale (default: 0.5)
    --num-scales INT             Number of scales (default: 3)
    --hidden-channels INT        Override hidden channels (default: per-model)
    --dropout FLOAT              Dropout rate (default: 0.1)
    --lr FLOAT                   Learning rate (default: 5e-3)
    --ensemble FILE [FILE ...]   Ensemble mode: average predictions from checkpoints
"""
import os, sys, json, argparse

# torch and xgboost bundle conflicting OpenMP runtimes; loading both in one
# process segfaults (pthread_mutex_init failed). These env vars must be set
# before either library is imported.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import dill as pck
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models import (
    _move_basis_tensors,
    PointNetTutorial, PointNet3D, DistanceMatrixRaggedModel, ScalarDistanceDeepSet,
    ScalarInputMLP, MultiInputModel,
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    HierarchicalGTTFN, HierarchicalTensorFieldNetwork, RaggedPersistenceModel,
    OnEquivariantTensorFieldNetwork, AttentionTensorFieldNetwork,
    StochasticTensorFieldNetwork, CrossAttentionTensorFieldNetwork,
    RelaxedOnEquivariantTensorFieldNetwork, HybridOnEquivariantTensorFieldNetwork,
    GraphMambaTensorFieldNetwork,
)
from tfn_enhancements import (
    MLPClassifierHead, MultiScalePersistenceEncoder, PointCloudAugmenter,
)

os.makedirs('results', exist_ok=True)
os.makedirs('results/enhanced', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Parse arguments ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Enhanced TFN Training')
parser.add_argument('dataset', type=str)
parser.add_argument('model', type=str)
parser.add_argument('fraction_pct', type=int)
parser.add_argument('trial', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('identifier', type=str, nargs='?', default='run1')
parser.add_argument('--classifier', type=str, default='mlp',
                    choices=['xgboost', 'mlp'],
                    help='Classifier type: mlp (end-to-end) or xgboost (baseline comparison)')
parser.add_argument('--augment', action='store_true',
                    help='Enable data augmentation during training')
parser.add_argument('--multi-scale', action='store_true',
                    help='Use multi-scale point clouds (subsample at several scales, '
                         'concatenate per-scale PVs before classification)')
parser.add_argument('--scale-factor', type=float, default=0.5,
                    help='Scale factor for multi-scale: scale s keeps fraction '
                         'scale_factor**s of the points')
parser.add_argument('--num-scales', type=int, default=3,
                    help='Number of persistence scales')
parser.add_argument('--hidden-channels', type=int, default=None,
                    help='Override hidden channels')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate for MLP head')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5,
                    help='Weight decay')
parser.add_argument('--ensemble', nargs='+', default=None,
                    help='Ensemble: list of checkpoint paths to average')
parser.add_argument('--gnn', action='store_true',
                    help='Use GNN classifier on persistence diagram points')

args = parser.parse_args()

dataset_name = args.dataset
model_name = args.model
fraction_pct = args.fraction_pct
trial = args.trial
num_epochs = args.epochs
identifier = args.identifier
fraction = fraction_pct / 100.0
use_mlp = args.classifier == 'mlp'
use_augment = args.augment
use_multiscale = args.multi_scale
use_gnn = args.gnn

print(f"train_enhanced: {dataset_name} {model_name} {fraction_pct}% trial={trial}")
print(f"  classifier={args.classifier}  augment={use_augment}  "
      f"multi_scale={use_multiscale}  gnn={use_gnn}  "
      f"hidden_ch={args.hidden_channels}")

# ─── Load data ───────────────────────────────────────────────────────────────
train_sfx = f"_train_TDE311LS_5{identifier}"
test_sfx = f"_test_TDE311LS_clean_3{identifier}"

train_data = pck.load(open(f"datasets/{dataset_name}{train_sfx}.pkl", 'rb'))
test_data = pck.load(open(f"datasets/{dataset_name}{test_sfx}.pkl", 'rb'))

data_train = train_data["data_train"]
PVs_train = train_data["PV_train"]
label_train = train_data["label_train"]
homdim = train_data["hdims"]
dim = data_train[0].shape[1]
output_dim = sum(PVs_train[h].shape[1] for h in range(len(homdim)))
N_full = len(data_train)

data_test = test_data["data_test"]
label_test = test_data["label_test"]
N_test = len(data_test)

# ─── Subsample ───────────────────────────────────────────────────────────────
rng = np.random.RandomState(trial * 42 + 999)
n_use = max(1, int(N_full * fraction))
idx = rng.choice(N_full, size=n_use, replace=False)
idx.sort()
data_train = [data_train[i] for i in idx]
PVs_train = [PVs_train[h][idx] for h in range(len(homdim))]
label_train = label_train[idx]
print(f"  {n_use}/{N_full} train samples ({fraction_pct}%)  test={N_test}  dim={dim}  out={output_dim}")

# ─── Labels ──────────────────────────────────────────────────────────────────
le = LabelEncoder().fit(np.concatenate([label_train, label_test]))
y_train, y_test = le.transform(label_train), le.transform(label_test)
n_classes = len(le.classes_)
print(f"  classes={n_classes}")

# ─── Augmenter ───────────────────────────────────────────────────────────────
augmenter = PointCloudAugmenter(
    jitter_std=0.01, rotate=True, scale_range=(0.9, 1.1),
    dropout_prob=0.1, permute=True
) if use_augment else None

# ─── Torch tensors ───────────────────────────────────────────────────────────
data_train_t = [torch.FloatTensor(x).to(device) for x in data_train]
data_test_t = [torch.FloatTensor(x).to(device) for x in data_test]
targets_t = torch.FloatTensor(np.concatenate(PVs_train, axis=1)).to(device)

# ─── Model building ─────────────────────────────────────────────────────────
TFN_MODELS = {
    'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
    'HierarchicalGTTFN', 'HierarchicalTensorFieldNetwork',
    'OnEquivariantTensorFieldNetwork', 'AttentionTensorFieldNetwork',
    'StochasticTensorFieldNetwork', 'CrossAttentionTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork', 'HybridOnEquivariantTensorFieldNetwork',
    'GraphMambaTensorFieldNetwork',
}

_npts = data_train[0].shape[0]

_hp = {'max_order': 0, 'hidden_channels': 8, 'num_layers': 2,
       'classifier_dims': [16], 'num_rbf': 64, 'k_neighbors': 8}

DS_HP = {
    'SonyAIBORobotSurface2': {'max_order': 2, 'hidden_channels': 16, 'num_layers': 3},
    'MiddlePhalanxOutlineCorrect': {'max_order': 1, 'hidden_channels': 8},
    'PowerCons': {'max_order': 1, 'hidden_channels': 16},
    'ProximalPhalanxTW': {'max_order': 1, 'hidden_channels': 8},
    'ECG5000': {'max_order': 1, 'hidden_channels': 16},
    'CBF': {'max_order': 1, 'hidden_channels': 8},
    'ItalyPowerDemand': {'max_order': 1, 'hidden_channels': 8},
    'TwoLeadECG': {'max_order': 1, 'hidden_channels': 8},
}
if dataset_name in DS_HP:
    _hp.update(DS_HP[dataset_name])

if args.hidden_channels is not None:
    _hp['hidden_channels'] = args.hidden_channels


def build_backbone(name, hp):
    if name == 'PointNetTutorial':
        return PointNetTutorial(output_dim=output_dim)
    if name == 'PointNet3D':
        return PointNet3D(output_dim=output_dim)
    if name == 'DistanceMatrixRaggedModel':
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=_npts)
    if name == 'RaggedPersistenceModel':
        return RaggedPersistenceModel(output_dim=output_dim)
    if name == 'ScalarDistanceDeepSet':
        return ScalarDistanceDeepSet(output_dim=output_dim)
    if name == 'ScalarInputMLP':
        return ScalarInputMLP(output_dim=output_dim)
    if name == 'MultiInputModel':
        return MultiInputModel(target_output_dim=output_dim, scalar_input_dim=1)
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim, **hp)
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=dim, num_classes=output_dim, radial_hidden=128, **hp)
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=dim, num_classes=output_dim, radial_hidden=128, **hp)
    if name == 'HierarchicalGTTFN':
        return HierarchicalGTTFN(n=dim, num_classes=output_dim,
            max_order=hp.get('max_order', 0), hidden_channels=hp.get('hidden_channels', 8),
            stage_sizes=[64, 32], num_rbf=hp.get('num_rbf', 64),
            cutoff=1.0, classifier_dims=hp.get('classifier_dims', [16]))
    if name == 'HierarchicalTensorFieldNetwork':
        return HierarchicalTensorFieldNetwork(num_classes=output_dim,
            max_order=hp.get('max_order', 0), hidden_channels=hp.get('hidden_channels', 8),
            stage_sizes=[64, 32], num_rbf=hp.get('num_rbf', 64),
            cutoff=1.0, classifier_dims=hp.get('classifier_dims', [16]))
    if name == 'OnEquivariantTensorFieldNetwork':
        return OnEquivariantTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64, 32])
    if name == 'AttentionTensorFieldNetwork':
        return AttentionTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_heads=4,
            num_rbf=64, classifier_dims=[64, 32], radial_hidden=64)
    if name == 'CrossAttentionTensorFieldNetwork':
        return CrossAttentionTensorFieldNetwork(num_classes=output_dim, n=dim,
            max_order=1, hidden_channels=8, num_layers=2, num_heads=4,
            transformer_layers=2, num_rbf=64, classifier_dims=[16], radial_hidden=64)
    if name == 'StochasticTensorFieldNetwork':
        return StochasticTensorFieldNetwork(num_classes=output_dim,
            num_mixtures=3, max_order=0, hidden_channels=8, num_layers=2,
            num_rbf=64, encoder_dims=[64, 32])
    if name == 'RelaxedOnEquivariantTensorFieldNetwork':
        return RelaxedOnEquivariantTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64, 32])
    if name == 'HybridOnEquivariantTensorFieldNetwork':
        return HybridOnEquivariantTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64, 32], non_eq_dim=128)
    if name == 'GraphMambaTensorFieldNetwork':
        return GraphMambaTensorFieldNetwork(num_classes=output_dim,
            max_order=hp.get('max_order', 1), hidden_channels=hp.get('hidden_channels', 32),
            num_layers=hp.get('num_layers', 4), num_rbf=hp.get('num_rbf', 64),
            k_neighbors=hp.get('k_neighbors', 16), classifier_dims=hp.get('classifier_dims'))
    raise ValueError(f"Unknown model: {name}")


backbone = build_backbone(model_name, _hp)

# Wrap with enhancements
if use_multiscale:
    model = MultiScalePersistenceEncoder(
        backbone, num_scales=args.num_scales, num_classes=n_classes,
        classifier_dims=[256, 128], dropout=args.dropout, pv_dim=output_dim,
    ).to(device)
elif model_name in TFN_MODELS:
    model = MLPClassifierHead(
        backbone, num_classes=n_classes,
        classifier_dims=[256, 128, 64],
        dropout=args.dropout,
        pv_dim=output_dim,
    ).to(device)
else:
    # Non-TFN models: wrap with MLP head directly
    model = MLPClassifierHead(
        backbone, num_classes=n_classes,
        classifier_dims=[256, 128, 64],
        dropout=args.dropout,
        pv_dim=output_dim,
    ).to(device)

n_params = sum(p.numel() for p in model.parameters())
tag = "MultiScale" if use_multiscale else "MLPHead"
print(f"  Model: {model_name} + {tag} ({n_params} params)")

# ─── Prepare inputs per model type ──────────────────────────────────────────
def _make_scales(x, num_scales, scale_factor, seed=0):
    """Return a list of `num_scales` subsampled versions of cloud x.

    Scale s keeps a fraction `scale_factor**s` of the points, so scale 0 is
    the full cloud and each subsequent scale coarsens it. Used as the
    multi-scale input to MultiScalePersistenceEncoder.
    """
    n = x.shape[0]
    rng = np.random.RandomState(seed)
    scales = []
    for s in range(num_scales):
        frac = scale_factor ** s
        keep_n = max(3, int(round(n * frac)))
        if keep_n >= n:
            scales.append(x)
        else:
            idx = np.sort(rng.choice(n, size=keep_n, replace=False))
            scales.append(x[idx])
    return scales


def _prepare_single(x):
    """Transform a single cloud to the input format this model expects."""
    if model_name in TFN_MODELS:
        return torch.cat([x, x.new_zeros(x.shape[0], 1)], dim=1) \
            if x.shape[1] == 2 else x
    if model_name in ('PointNet3D', 'PointNetTutorial'):
        nc = 3 if model_name == 'PointNet3D' else 2
        xa = x.cpu().numpy()
        if xa.shape[1] < nc:
            xa = np.concatenate([xa, np.zeros((xa.shape[0], nc - xa.shape[1]))], axis=1)
        else:
            xa = xa[:, :nc]
        return torch.FloatTensor(xa).to(device)
    if model_name in ('ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel'):
        a = x.cpu().numpy()
        m = np.linalg.norm(a[:, None] - a[None], axis=-1)
        return torch.FloatTensor(m).to(device)
    if model_name == 'ScalarInputMLP':
        a = x.cpu().numpy()
        m = np.linalg.norm(a[:, None] - a[None], axis=-1)
        return torch.FloatTensor([[m.mean()]]).to(device)
    if model_name == 'MultiInputModel':
        a = x.cpu().numpy()
        m = np.linalg.norm(a[:, None] - a[None], axis=-1)
        return (x, torch.FloatTensor([[m.mean()]]).to(device))
    return x


def prepare(data_list, augment=False):
    if augment and augmenter is not None:
        data_list = [augmenter(x) for x in data_list]

    if use_multiscale:
        # Each sample becomes a list of `num_scales` clouds (one per scale).
        return [[_prepare_single(sc) for sc in _make_scales(x, args.num_scales, args.scale_factor)]
                for x in data_list]
    return [_prepare_single(x) for x in data_list]


train_in = prepare(data_train_t)
test_in = prepare(data_test_t)


# ─── TFN geometry helpers ───────────────────────────────────────────────────
def _unwrap_tfn(m):
    for _ in range(8):
        if hasattr(m, 'k_neighbors') and hasattr(m, 'rbf') and hasattr(m, 'gt_basis'):
            return m
        child = (getattr(m, '_inner', None) or getattr(m, 'base', None)
                 or getattr(m, 'tfn_backbone', None) or getattr(m, 'backbone', None))
        if child is None or child is m:
            break
        m = child
    return m


def _find_encoder(m):
    for _ in range(8):
        if (hasattr(m, '_encode_single') or hasattr(m, '_encode_batch')) and hasattr(m, 'rho'):
            return m
        child = (getattr(m, '_inner', None) or getattr(m, 'base', None)
                 or getattr(m, 'tfn_backbone', None) or getattr(m, 'backbone', None))
        if child is None or child is m:
            break
        m = child
    return m


def _is_hybrid(m):
    inner = getattr(m, '_inner', None)
    return inner is not None and hasattr(inner, 'neq_phi')


def precompute_geom(model, data_list):
    if model_name not in TFN_MODELS:
        return None
    try:
        from gt_tfn_layer import knn_geometry
        inner = _unwrap_tfn(model)
        _move_basis_tensors(inner, device)
        k = getattr(inner, 'k_neighbors', _hp['k_neighbors'])
        rbfs, gts, nbrs = [], [], []
        for pc in data_list:
            r, g, n = knn_geometry(pc, inner.rbf, inner.gt_basis, k)
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


# Precompute geometry using the backbone (not the MLP wrapper).
# Multi-scale inputs are subsampled per scale, so precomputed geometry would
# be stale — let the encoder compute geometry internally instead.
if use_multiscale:
    train_geom = test_geom = None
else:
    train_geom = precompute_geom(model, train_in)
    test_geom = precompute_geom(model, test_in)


def get_geom(g, bix):
    if g is None:
        return None
    if isinstance(bix, range):
        bix = list(bix)
    if g.get('uniform', False):
        return {k: g[k][bix] if k in ('rbf', 'gt_edge', 'nbr_idx') else g[k]
                for k in g}
    return [g['list'][i] for i in bix]


def _forward_backbone(scale_batch):
    """Forward one scale's batch through the backbone (geometry internal)."""
    backbone_m = getattr(model, 'backbone', model)
    if model_name == 'MultiInputModel':
        return backbone_m([x[0] for x in scale_batch],
                          torch.cat([x[1] for x in scale_batch]))
    if model_name == 'ScalarInputMLP':
        return backbone_m(torch.cat([x.reshape(1, -1) for x in scale_batch]))
    return backbone_m(scale_batch)


def forward_with_geom(model, batch_data, geom=None):
    """Forward pass using backbone's precomputed geometry, returns PV features."""
    if use_multiscale:
        # batch_data[i] is the list of per-scale clouds for sample i.
        n = len(batch_data)
        pvs = []
        for s in range(args.num_scales):
            sb = [batch_data[i][s] for i in range(n)]
            pvs.append(_forward_backbone(sb))
        return torch.cat(pvs, dim=-1)

    mname = model_name

    if mname == 'MultiInputModel':
        backbone_m = getattr(model, 'backbone', model)
        return backbone_m([x[0] for x in batch_data],
                     torch.cat([x[1] for x in batch_data]))
    if mname == 'ScalarInputMLP':
        backbone_m = getattr(model, 'backbone', model)
        return backbone_m(torch.cat([x.reshape(1, -1) for x in batch_data]))
    if mname == 'CrossAttentionTensorFieldNetwork':
        backbone_m = getattr(model, 'backbone', model)
        return backbone_m(batch_data)

    if geom is not None and mname in TFN_MODELS and not _is_hybrid(model):
        inner = _find_encoder(model)
        _move_basis_tensors(inner, device)
        if isinstance(geom, dict) and geom.get('uniform', False):
            if hasattr(inner, '_encode_batch'):
                pv = inner._encode_batch(torch.stack(batch_data),
                    precomputed_geom=(geom['rbf'], geom['gt_edge'], geom['nbr_idx']))
                return pv
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
            descs.append(inner._encode_single(x, precomputed_geom=(r, g, n)))
        return inner.rho(torch.stack(descs))

    # Fallback: direct forward through backbone
    backbone_m = getattr(model, 'backbone', model)
    return backbone_m(batch_data)


def forward_full(model, batch_data, geom=None):
    """Full forward: backbone PV -> classifier head."""
    pv = forward_with_geom(model, batch_data, geom=geom)
    if use_multiscale:
        return model.fusion_classifier(pv)
    if hasattr(model, 'classifier'):
        return model.classifier(pv)
    return pv


# ─── Training ────────────────────────────────────────────────────────────────
tr_acc = te_acc = float('nan')
try:
    if (model_name in TFN_MODELS and model_name != 'CrossAttentionTensorFieldNetwork'
            and train_geom is None and not use_multiscale):
        print("  GEOMETRY FAILED — skipping training")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        bs = min(32, len(train_in))
        n_t = len(train_in)

        for ep in range(num_epochs):
            model.train()
            perm = np.random.permutation(n_t)
            loss_sum = 0.0
            correct = 0
            total = 0

            for s in range(0, n_t, bs):
                bix = perm[s:s + bs]
                bd = [train_in[i] for i in bix]
                bg = get_geom(train_geom, bix)
                bt = torch.LongTensor(y_train[bix]).to(device)

                # Augment: re-prepare inputs with augmentation for this batch
                if use_augment and augmenter is not None and model_name in TFN_MODELS:
                    bd_aug = prepare([data_train_t[i] for i in bix], augment=True)
                    # Recompute geometry for augmented data
                    bg = None  # Can't reuse precomputed geom for augmented data

                optimizer.zero_grad(set_to_none=True)

                if use_augment and augmenter is not None and model_name in TFN_MODELS:
                    out = forward_full(model, bd_aug, geom=None)
                else:
                    out = forward_full(model, bd, geom=bg)

                loss = criterion(out, bt)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    loss_sum += loss.item() * len(bd)
                    correct += (out.argmax(-1) == bt).sum().item()
                    total += len(bd)

            if (ep + 1) % 100 == 0 or ep == 0:
                train_acc = 100.0 * correct / total if total > 0 else 0
                print(f"  ep {ep + 1}/{num_epochs}  loss={loss_sum / n_t:.4f}  "
                      f"train_acc={train_acc:.1f}%")

        # ─── Evaluate ──────────────────────────────────────────────────────────
        model.eval()
        all_logits = []
        with torch.no_grad():
            for s in range(0, len(test_in), bs):
                e = min(s + bs, len(test_in))
                bd = test_in[s:e]
                bg = get_geom(test_geom, list(range(s, e)))
                out = forward_full(model, bd, geom=bg)
                all_logits.append(out.cpu().numpy())
        logits_test = np.vstack(all_logits)
        pred_test = logits_test.argmax(axis=-1)
        te_acc = np.mean(pred_test == y_test)

        # Also run XGBoost on PV features for comparison
        all_pv_train, all_pv_test = [], []
        with torch.no_grad():
            for s in range(0, len(train_in), bs):
                e = min(s + bs, len(train_in))
                bd = train_in[s:e]
                bg = get_geom(train_geom, list(range(s, e)))
                pv = forward_with_geom(model, bd, geom=bg)
                all_pv_train.append(pv.cpu().numpy())
            for s in range(0, len(test_in), bs):
                e = min(s + bs, len(test_in))
                bd = test_in[s:e]
                bg = get_geom(test_geom, list(range(s, e)))
                pv = forward_with_geom(model, bd, geom=bg)
                all_pv_test.append(pv.cpu().numpy())

        PV_train = np.vstack(all_pv_train)
        PV_test = np.vstack(all_pv_test)

        n_classes_xgb = len(np.unique(y_train))
        xgb_tr_acc = xgb_te_acc = float('nan')
        if n_classes_xgb >= 2:
            le_xgb = LabelEncoder()
            y_train_enc = le_xgb.fit_transform(y_train)
            mask = np.isin(y_test, le_xgb.classes_)
            if mask.sum() > 0:
                y_test_enc = le_xgb.transform(y_test[mask])
                clf = XGBClassifier(eval_metric='logloss', verbosity=0)
                clf.fit(PV_train, y_train_enc)
                xgb_tr_acc = clf.score(PV_train, y_train_enc)
                xgb_te_acc = clf.score(PV_test[mask], y_test_enc)

        print(f"  MLP  test={100 * te_acc:.2f}%")
        print(f"  XGB  train={100 * xgb_tr_acc:.2f}%  test={100 * xgb_te_acc:.2f}%")

except Exception as e:
    import traceback
    print(f"  Training/evaluation failed: {e}")
    traceback.print_exc()

# ─── Save ────────────────────────────────────────────────────────────────────
result = {
    'dataset': dataset_name, 'model': model_name,
    'fraction_pct': fraction_pct, 'trial': trial,
    'n_train_used': n_use, 'n_train_full': N_full, 'n_test': N_test,
    'mlp_test_acc': te_acc if use_mlp else float('nan'),
    'xgb_test_acc': xgb_te_acc if (not use_mlp and 'xgb_te_acc' in dir()) else (
        te_acc if not use_mlp else float('nan')),
    'classifier': args.classifier,
    'augment': use_augment,
    'multi_scale': use_multiscale,
    'num_scales': args.num_scales if use_multiscale else None,
    'scale_factor': args.scale_factor if use_multiscale else None,
    'hidden_channels': args.hidden_channels,
    'dropout': args.dropout,
}
clf_tag = f"_{args.classifier}"
aug_tag = "_aug" if use_augment else ""
ms_tag = "_ms" if use_multiscale else ""
out_path = f"results/enhanced/train_{dataset_name}_{model_name}_{fraction_pct}pct_t{trial}{clf_tag}{aug_tag}{ms_tag}.json"
with open(out_path, 'w') as f:
    json.dump(result, f)
print(f"  Saved to {out_path}")
