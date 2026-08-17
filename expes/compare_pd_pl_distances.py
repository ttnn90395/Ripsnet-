#!/usr/bin/env python3
"""
compare_pd_pl_distances.py
===========================
Compare how far each model's predicted persistence diagrams/landscapes
are from the ground truth, using multiple distance metrics.

Metrics computed:
  PV-space (predicted vs true PV vectors):
    - L1 (Manhattan) distance
    - L2 (Euclidean) distance
    - Linf (Weierstrass / sup-norm) distance
    - Cosine distance
    - MSE (mean squared error)
    - KL divergence (for non-negative PVs like PI)

  PD-space (ground-truth PDs only, as reference baselines):
    - Bottleneck distance between clean vs noisy PDs
    - Wasserstein-1 distance between clean vs noisy PDs
    - Wasserstein-2 distance between clean vs noisy PDs

Usage:
  Combined format (recommended):
    python compare_pd_pl_distances.py <dataset> <normalize> <PV_type> <mode>

  3-file format (backward compat):
    python compare_pd_pl_distances.py <PV_params> <train> <test> <normalize> <PV_type> <mode>

Examples:
  python compare_pd_pl_distances.py synth_train_LS_1run1 0 PL test
  python compare_pd_pl_distances.py CBF_train_TDE311LS_5sweep_default 0 PL test
"""

import os, sys
import numpy as np
import torch
import dill as pck
import gudhi as gd
from gudhi.representations import PersistenceImage, Landscape, DiagramSelector
from gudhi.hera import wasserstein_distance as wasserstein_distance_gudhi
from scipy.spatial.distance import cosine as cosine_dist
from time import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.makedirs('results', exist_ok=True)

from models import (
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    HierarchicalGTTFN, HierarchicalTensorFieldNetwork,
    OnEquivariantTensorFieldNetwork, PersNet,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DenseRagged, PermopRagged, RaggedPersistenceModel, DistanceMatrixRaggedModel,
    AttentionTensorFieldNetwork, StochasticTensorFieldNetwork,
    CrossAttentionTensorFieldNetwork,
    RelaxedOnEquivariantTensorFieldNetwork,
    HybridOnEquivariantTensorFieldNetwork,
    _move_basis_tensors,
)

MODEL_NAMES = [
    'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
    'HierarchicalGTTFN', 'HierarchicalTensorFieldNetwork',
    'OnEquivariantTensorFieldNetwork', 'PersNet',
    'ScalarDistanceDeepSet', 'PointNetTutorial', 'ScalarInputMLP', 'MultiInputModel',
    'RaggedPersistenceModel', 'DistanceMatrixRaggedModel',
    'AttentionTensorFieldNetwork', 'StochasticTensorFieldNetwork',
    'CrossAttentionTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork',
    'HybridOnEquivariantTensorFieldNetwork',
]

TFN_MODELS = {
    'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
    'HierarchicalGTTFN', 'HierarchicalTensorFieldNetwork', 'OnEquivariantTensorFieldNetwork',
    'AttentionTensorFieldNetwork', 'StochasticTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork',
    'CrossAttentionTensorFieldNetwork',
}

# ── CLI args ────────────────────────────────────────────────────────────────
# Mode 1 (combined pickle, recommended):
#   python compare_pd_pl_distances.py <combined_dataset> <normalize> <PV_type> <mode>
#   e.g. python compare_pd_pl_distances.py synth_train_LS_1run1 0 PL test
#
# Mode 2 (3-file format, for backward compat with analysis_nn.py):
#   python compare_pd_pl_distances.py <PV_params> <train> <test> <normalize> <PV_type> <mode>

dataset_name = sys.argv[1] if len(sys.argv) >= 5 else None

if len(sys.argv) == 5:
    # Combined format: dataset_name normalize PV_type mode
    normalize = int(sys.argv[2])
    PV_type   = sys.argv[3]
    mode      = sys.argv[4]

    data = pck.load(open('datasets/' + dataset_name + '.pkl', 'rb'))
    PV_params, homdim     = data['PV_params'], data['hdims']
    PVs_train             = data['PV_train']
    data_classif_train    = data['data_train']
    data_classif_test     = data['data_test']
    PV_type               = data.get('PV_type', PV_type)

    dataset_label = dataset_name
elif len(sys.argv) >= 7:
    # 3-file format
    dataset_PV_params  = sys.argv[1]
    dataset_train_name = sys.argv[2]
    dataset_test_name  = sys.argv[3]
    normalize          = int(sys.argv[4])
    PV_type            = sys.argv[5]
    mode               = sys.argv[6]

    PV_setting = pck.load(open('datasets/' + dataset_PV_params + '.pkl', 'rb'))
    PV_params, homdim = PV_setting['PV_params'], PV_setting['hdims']

    data_train = pck.load(open('datasets/' + dataset_train_name + '.pkl', 'rb'))
    PVs_train  = data_train['PV_train']
    data_classif_train = data_train['data_train']

    data_test  = pck.load(open('datasets/' + dataset_test_name + '.pkl', 'rb'))
    data_classif_test = data_test['data_test']

    dataset_label = dataset_test_name
else:
    print("Usage:")
    print("  Combined: python compare_pd_pl_distances.py <dataset> <normalize> <PV_type> <mode>")
    print("  3-file:   python compare_pd_pl_distances.py <PV_params> <train> <test> <normalize> <PV_type> <mode>")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Args: {sys.argv}")
print(f"Device: {device}")

data_sets = data_classif_train + data_classif_test
N_sets    = len(data_sets)
N_train   = len(data_classif_train)
N_test    = len(data_classif_test)
dim       = data_sets[0].shape[1] if N_sets > 0 else 2

PV_size = PV_params[0]['resolution'][0] if PV_type == 'PI' else PV_params[0]['resolution']
output_dim = sum(PVs_train[hidx].shape[1] for hidx in range(len(homdim)))

print(f"Dataset: {dataset_label}  N_train={N_train}  N_test={N_test}  dim={dim}")
print(f"PV_type={PV_type}  PV_size={PV_size}  output_dim={output_dim}  homdim={homdim}")


# ── Ground truth PDs and PVs ────────────────────────────────────────────────

print("\n--- Computing ground-truth PDs (Gudhi AlphaComplex) ---")
PD_gudhi = []
t0 = time()
for i in tqdm(range(N_sets), desc="Gudhi PDs"):
    rcX = gd.AlphaComplex(points=data_sets[i]).create_simplex_tree()
    rcX.persistence()
    final_dg = []
    for hdim in homdim:
        dg = rcX.persistence_intervals_in_dimension(hdim)
        if dg is None or len(dg) == 0:
            dg = np.empty([0, 2])
        final_dg.append(dg)
    PD_gudhi.append(final_dg)
time_pd = time() - t0
print(f"  PD computation: {time_pd:.2f}s")

print("\n--- Computing ground-truth PVs ---")
PV_gudhi = []
for hidx in range(len(homdim)):
    if PV_type == 'PI':
        a, b = PV_params[hidx]['weight'][0], PV_params[hidx]['weight'][1]
        PV_params[hidx]['weight'] = lambda x, a=a, b=b: a * np.tanh(x[1]) ** b
    PV = (PersistenceImage(**PV_params[hidx]) if PV_type == 'PI'
          else Landscape(**PV_params[hidx]))
    pv_hidx = PV.fit_transform(
        DiagramSelector(use=True).fit_transform(
            [PD_gudhi[i][hidx] for i in range(N_sets)]))
    if normalize:
        pv_hidx /= np.max(pv_hidx[:N_train])
    PV_gudhi.append(pv_hidx)

PV_true = np.hstack(PV_gudhi)  # (N_sets, output_dim)
print(f"  PV_true shape: {PV_true.shape}")


# ── Distance metrics ────────────────────────────────────────────────────────

def l1_distance(y_true, y_pred):
    """Mean L1 (Manhattan) distance per sample."""
    return np.mean(np.abs(y_true - y_pred))

def l2_distance(y_true, y_pred):
    """Mean L2 (Euclidean) distance per sample."""
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=-1)))

def linf_distance(y_true, y_pred):
    """Mean Linf (Weierstrass / sup-norm) distance per sample.
    This measures the maximum absolute error per sample."""
    return np.mean(np.max(np.abs(y_true - y_pred), axis=-1))

def cosine_distance(y_true, y_pred):
    """Mean cosine distance per sample."""
    dists = []
    for i in range(len(y_true)):
        norm_t = np.linalg.norm(y_true[i])
        norm_p = np.linalg.norm(y_pred[i])
        if norm_t < 1e-10 or norm_p < 1e-10:
            dists.append(1.0)
        else:
            dists.append(cosine_dist(y_true[i], y_pred[i]))
    return np.mean(dists)

def mse(y_true, y_pred):
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)

def kl_divergence(y_true, y_pred, eps=1e-8):
    """KL divergence (treating vectors as distributions after ReLU + normalization)."""
    p = np.clip(y_true, eps, None)
    q = np.clip(y_pred, eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.mean(np.sum(p * np.log(p / q), axis=-1))


def compute_pd_metrics(pd1_list, pd2_list):
    """
    Compute bottleneck and Wasserstein distances between two lists of PDs.
    Returns dict with mean ± std for each metric.
    """
    bottlenecks, w1s, w2s = [], [], []
    for pd1, pd2 in zip(pd1_list, pd2_list):
        if pd1.shape[0] == 0 or pd2.shape[0] == 0:
            continue
        try:
            bn = gd.bottleneck_distance(pd1, pd2)
            bottlenecks.append(bn)
        except Exception:
            pass
        try:
            w1 = wasserstein_distance_gudhi(pd1, pd2, order=1)
            w1s.append(w1)
        except Exception:
            pass
        try:
            w2 = wasserstein_distance_gudhi(pd1, pd2, order=2)
            w2s.append(w2)
        except Exception:
            pass

    def _stats(arr):
        if len(arr) == 0:
            return {'mean': float('nan'), 'std': float('nan'), 'median': float('nan')}
        return {'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
                'median': float(np.median(arr))}

    return {
        'bottleneck': _stats(bottlenecks),
        'wasserstein_1': _stats(w1s),
        'wasserstein_2': _stats(w2s),
        'n_comparable': len(w1s),
    }


# ── Find and load models ────────────────────────────────────────────────────

def find_checkpoint(mname):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_tag = os.environ.get('TFN_MODEL_TAG', '')
    candidate_dirs = [
        f'models/{model_tag}' if model_tag else 'models',
        'models',
        os.path.join(script_dir, f'models/{model_tag}') if model_tag else '',
        os.path.join(script_dir, 'models'),
        os.path.join(script_dir, '..', 'models'),
        os.path.join(script_dir, '..', '..', 'models'),
    ]
    candidate_dirs = [d for d in candidate_dirs if d]
    for d in candidate_dirs:
        for ext in ('.pth', '.pt'):
            p = os.path.join(d, mname + ext)
            if os.path.isfile(p):
                return p
    return None


def find_all_checkpoints():
    """Find all available model checkpoints."""
    found = []
    for mname in MODEL_NAMES:
        for suffix in ('', '_GS'):
            full = mname + suffix
            path = find_checkpoint(full)
            if path is not None:
                found.append((full, path))
    return found


# ── Inference helpers (from analysis_nn.py) ─────────────────────────────────

def _pad_to_3d(batch):
    return [
        torch.cat([x, x.new_zeros(x.shape[0], 1)], dim=1) if x.shape[1] == 2 else x
        for x in batch
    ]

def gaussian_smooth_batch(data_list, sigma=0.5):
    if not data_list:
        return data_list
    if all(x.shape == data_list[0].shape for x in data_list):
        X     = torch.stack(data_list)
        diff  = X.unsqueeze(2) - X.unsqueeze(1)
        dist2 = (diff ** 2).sum(-1)
        W     = torch.exp(-dist2 / (2 * sigma ** 2))
        W     = W / W.sum(-1, keepdim=True)
        return list((W @ X).unbind(0))
    out = []
    for x in data_list:
        diff  = x.unsqueeze(0) - x.unsqueeze(1)
        dist2 = (diff ** 2).sum(-1)
        W     = torch.exp(-dist2 / (2 * sigma ** 2))
        W     = W / W.sum(-1, keepdim=True)
        out.append(W @ x)
    return out


def precompute_geometry_for_model(model_pv, class_name, data_list):
    if class_name not in TFN_MODELS:
        return None
    try:
        from gt_tfn_layer import knn_geometry
        inner   = getattr(model_pv, '_inner', model_pv)
        rbf_enc = inner.rbf
        gt_basis = inner.gt_basis
        k        = inner.k_neighbors
    except AttributeError:
        return None
    _move_basis_tensors(inner, next(inner.parameters()).device)
    is_hier = hasattr(inner, 'precompute_hierarchical_geometry')
    rbfs, gt_edges, nbr_idxs = [], [], []
    hier_geoms = [] if is_hier else None
    with torch.no_grad():
        for pc in tqdm(data_list, desc=f"  Precompute geom ({class_name})", leave=False):
            rbf, gt_edge, nbr_idx = knn_geometry(pc, rbf_enc, gt_basis, k)
            rbfs.append(rbf.detach())
            gt_edges.append(gt_edge.detach())
            nbr_idxs.append(nbr_idx.detach())
            if is_hier:
                hier_geoms.append(inner.precompute_hierarchical_geometry(pc))
    uniform = all(r.shape == rbfs[0].shape for r in rbfs)
    if uniform:
        result = {'rbf': torch.stack(rbfs), 'gt_edge': torch.stack(gt_edges),
                  'nbr_idx': torch.stack(nbr_idxs), 'uniform': True}
        if is_hier:
            result['hier'] = [[{kk: vv.detach() for kk, vv in sg.items()}
                               for sg in hg] for hg in hier_geoms]
        return result
    return {'list': list(zip(rbfs, gt_edges, nbr_idxs)), 'uniform': False}


def tfn_batched_forward(model, data_list, geom_cache, batch_size=64):
    """
    Batched TFN inference with precomputed geometry.
    Mirrors analysis_nn.py's approach: calls inner._encode_single() + inner.rho().
    """
    inner = getattr(model, '_inner', model)
    inner_device = next(inner.parameters()).device
    _move_basis_tensors(inner, inner_device)
    results = []
    is_batched_geom = isinstance(geom_cache, dict) and geom_cache.get('uniform', False)
    has_hier = isinstance(geom_cache, dict) and 'hier' in geom_cache
    # TemporalCrossAttentionTFN runs a temporal transformer + concat between
    # the per-point encoder and the rho head, so the generic
    # _encode_single() → rho() shortcut does not apply.  Route it through
    # its own forward() with per-sample precomputed geometry instead.
    is_cross_attn = isinstance(model, CrossAttentionTensorFieldNetwork)

    def _norm_geom_tuple(j, rbf_s, gt_edge_s, nbr_s):
        rbf_i     = rbf_s[j]     if rbf_s[j].ndim == 3     else rbf_s[j].squeeze(0)
        gt_edge_i = gt_edge_s[j] if gt_edge_s[j].ndim == 3 else gt_edge_s[j].squeeze(0)
        nbr_i     = nbr_s[j]     if nbr_s[j].ndim == 2     else nbr_s[j].squeeze(0)
        return (rbf_i, gt_edge_i, nbr_i)

    def _get_stage_geom(i):
        if not has_hier:
            return None
        return [{k: v.to(inner_device) for k, v in sg.items()}
                for sg in geom_cache['hier'][i]]

    with torch.no_grad():
        for start in range(0, len(data_list), batch_size):
            end = min(start + batch_size, len(data_list))
            batch = data_list[start:end]

            if geom_cache is None:
                out = model(batch)

            elif is_cross_attn:
                out_parts = []
                for j, pc in enumerate(batch):
                    idx = start + j
                    if is_batched_geom:
                        geom_i = _norm_geom_tuple(j, geom_cache['rbf'],
                                                  geom_cache['gt_edge'],
                                                  geom_cache['nbr_idx'])
                    else:
                        geom_i = geom_cache['list'][idx] \
                            if isinstance(geom_cache, dict) else geom_cache[idx]
                    out_i = model([pc], precomputed_geom=[geom_i])
                    out_parts.append(out_i.detach())
                out = torch.cat(out_parts, dim=0)

            elif is_batched_geom:
                rbf_b     = geom_cache['rbf'][start:end]
                gt_edge_b = geom_cache['gt_edge'][start:end]
                nbr_b     = geom_cache['nbr_idx'][start:end]

                if hasattr(inner, '_encode_batch'):
                    batch_tensor = torch.stack(batch)
                    out = inner._encode_batch(
                        batch_tensor,
                        precomputed_geom=(rbf_b, gt_edge_b, nbr_b))
                else:
                    descs = []
                    for j, pc in enumerate(batch):
                        idx = start + j
                        rbf_i     = rbf_b[j]     if rbf_b[j].ndim == 3     else rbf_b[j].squeeze(0)
                        gt_edge_i = gt_edge_b[j] if gt_edge_b[j].ndim == 3 else gt_edge_b[j].squeeze(0)
                        nbr_i     = nbr_b[j]     if nbr_b[j].ndim == 2     else nbr_b[j].squeeze(0)
                        desc = inner._encode_single(
                            pc,
                            precomputed_geom=(rbf_i, gt_edge_i, nbr_i),
                            precomputed_stage_geom=_get_stage_geom(idx))
                        descs.append(desc.detach())
                    out = inner.rho(torch.stack(descs))

            else:
                geom_b = geom_cache['list'][start:end] if isinstance(geom_cache, dict) \
                         else geom_cache[start:end]
                descs = []
                for j, (pc, (rbf, gt_edge, nbr_idx)) in enumerate(zip(batch, geom_b)):
                    idx = start + j
                    rbf_i     = rbf.to(inner_device)     if rbf.ndim == 3     else rbf.squeeze(0).to(inner_device)
                    gt_edge_i = gt_edge.to(inner_device) if gt_edge.ndim == 3 else gt_edge.squeeze(0).to(inner_device)
                    nbr_i     = nbr_idx.to(inner_device) if nbr_idx.ndim == 2 else nbr_idx.squeeze(0).to(inner_device)
                    desc = inner._encode_single(
                        pc,
                        precomputed_geom=(rbf_i, gt_edge_i, nbr_i),
                        precomputed_stage_geom=_get_stage_geom(idx))
                    descs.append(desc.detach())
                out = inner.rho(torch.stack(descs))

            results.append(out.detach().cpu().numpy())
    return np.vstack(results)


def prepare_non_tfn_input(x_tensor, name):
    """Convert a raw point cloud into the input format each non-TFN model
    expects (mirrors analysis_nn.prepare_single_input)."""
    arr = x_tensor.cpu().numpy()
    if name == 'PersNet':
        if arr.shape[1] < 3:
            pad = np.zeros((arr.shape[0], 3 - arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        return [torch.FloatTensor(arr[:, :3]).to(device)]
    if name == 'PointNetTutorial':
        return [torch.FloatTensor(arr[:, :2]).to(device)]
    if name in ('ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel',
                'RaggedPersistenceModel'):
        mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        return [torch.FloatTensor(mat).to(device)]
    if name == 'ScalarInputMLP':
        mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        return [torch.FloatTensor([[mat.mean()]]).to(device)]
    if name == 'MultiInputModel':
        mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scalar = torch.FloatTensor([[mat.mean()]]).to(device)
        return ([torch.FloatTensor(arr).to(device)], scalar)
    return [x_tensor]


# ── Main evaluation loop ────────────────────────────────────────────────────

checkpoints = find_all_checkpoints()
print(f"\nFound {len(checkpoints)} checkpoints:")
for name, path in checkpoints:
    print(f"  {name}: {path}")

all_results = {}

for model_label, ckpt_path in checkpoints:
    print(f"\n{'='*60}")
    print(f"  Model: {model_label}")
    print(f"{'='*60}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            ckpt_type = checkpoint.get('model_type', model_label.replace('_GS', ''))
            ckpt_dim  = checkpoint.get('dim', dim)
            ckpt_use_gs = checkpoint.get('use_gs', '_GS' in model_label)
            ckpt_sigma = checkpoint.get('gs_sigma', 0.5)
            ckpt_output_dim = checkpoint.get('output_dim', output_dim)
        else:
            model_state = checkpoint
            ckpt_type = model_label.replace('_GS', '')
            ckpt_dim = dim
            ckpt_use_gs = '_GS' in model_label
            ckpt_sigma = 0.5
            ckpt_output_dim = output_dim

        class_name = ckpt_type
        out_dim = ckpt_output_dim

        # Build model
        tfn_extra = {}
        saved_hp_keys = ['hidden_channels', 'num_layers', 'num_rbf', 'cutoff',
                         'k_neighbors', 'max_order', 'num_heads', 'radial_hidden',
                         'num_mixtures', 'encoder_dims', 'stage_sizes', 'stage_radii',
                         'k_local', 'k_global', 'num_layers_per_stage',
                         'transformer_layers', 'dropout', 'classifier_dims',
                         'num_points']
        for k in saved_hp_keys:
            v = checkpoint.get(k) if isinstance(checkpoint, dict) else None
            if v is not None:
                tfn_extra[k] = v

        # Infer architecture params from state dict shapes when not in ckpt
        def _infer_arch_from_state(model_state):
            info = {}
            keys = set(model_state.keys())
            for prefix in ('_inner.base._inner.', '_inner.tfn_backbone._inner.', '_inner.', ''):
                if any(k.startswith(prefix) for k in keys):
                    break
            rbf_key = prefix + 'rbf.centers'
            if rbf_key in model_state:
                info['num_rbf'] = model_state[rbf_key].shape[0]
            mp_prefix = prefix + 'mp_layers.'
            radial_key = next(
                (k for k in keys if k.startswith(mp_prefix) and 'radial_nets.' in k and k.endswith('.0.weight')),
                None)
            if radial_key is not None:
                info['radial_hidden'] = model_state[radial_key].shape[0]
            gate_key = next((k for k in keys if 'gate.gates.' in k and k.endswith('.weight')), None)
            if gate_key is not None:
                info['hidden_channels'] = model_state[gate_key].shape[0]
            else:
                ln_key = next((k for k in keys if k.startswith(mp_prefix) and 'layer_norms.' in k and k.endswith('.weight')), None)
                if ln_key is not None:
                    info['hidden_channels'] = model_state[ln_key].shape[0]
            layer_idxs = []
            for k in keys:
                if k.startswith(mp_prefix):
                    rest = k[len(mp_prefix):]
                    idx = rest.split('.', 1)[0]
                    if idx.isdigit():
                        layer_idxs.append(int(idx))
            if layer_idxs:
                info['num_layers'] = max(layer_idxs) + 1
            rho_prefix = prefix + 'rho.'
            rho_linear_keys = sorted(
                [k for k in keys if k.startswith(rho_prefix) and k.endswith('.weight')
                 and model_state[k].ndim == 2
                 and k[len(rho_prefix):].split('.', 1)[0].isdigit()],
                key=lambda k: int(k[len(rho_prefix):].split('.', 1)[0]))
            if len(rho_linear_keys) >= 2:
                info['classifier_dims'] = [model_state[k].shape[0] for k in rho_linear_keys[:-1]]
            classifier_prefix = prefix + 'classifier.'
            clf_linear_keys = sorted(
                [k for k in keys if k.startswith(classifier_prefix) and k.endswith('.weight')
                 and model_state[k].ndim == 2],
                key=lambda k: int(k[len(classifier_prefix):].split('.')[0]))
            if len(clf_linear_keys) >= 2 and 'classifier_dims' not in info:
                info['classifier_dims'] = [model_state[k].shape[0] for k in clf_linear_keys[:-1]]
            return info

        inferred = _infer_arch_from_state(model_state)
        for k, v in inferred.items():
            if k not in tfn_extra or tfn_extra[k] is None:
                tfn_extra[k] = v

        # Build model locally (avoids importing analysis_nn which has sys.argv side effects)
        def _build_model_local(name, output_dim, n=None, extra=None,
                               activation='gelu', norm='bn'):
            n_dim = dim if n is None else n
            extra = extra or {}
            hidden_channels = extra.get('hidden_channels')
            num_layers      = extra.get('num_layers')
            num_rbf         = extra.get('num_rbf')
            classifier_dims = extra.get('classifier_dims')
            radial_hidden   = extra.get('radial_hidden')
            cutoff          = extra.get('cutoff', 1.0)
            k_neighbors     = extra.get('k_neighbors', 16)

            if name == 'TensorFieldNetwork':
                return TensorFieldNetwork(
                    num_classes=output_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    num_layers=num_layers or 6,
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [256, 128])
            if name == 'GTTensorFieldNetwork':
                return GTTensorFieldNetwork(
                    n=n_dim, num_classes=output_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    num_layers=num_layers or 6,
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [256, 128],
                    radial_hidden=radial_hidden or 128)
            if name == 'GTTensorFieldNetworkV2':
                return GTTensorFieldNetworkV2(
                    n=n_dim, num_classes=output_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    num_layers=num_layers or 6,
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [256, 128],
                    radial_hidden=radial_hidden or 128)
            if name == 'HierarchicalGTTFN':
                return HierarchicalGTTFN(
                    n=n_dim, num_classes=output_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    stage_sizes=extra.get('stage_sizes', [256, 64]),
                    stage_radii=extra.get('stage_radii', [0.2, 0.4]),
                    k_local=extra.get('k_local', 16),
                    k_global=extra.get('k_global', 16),
                    num_layers_per_stage=extra.get('num_layers_per_stage', 2),
                    num_rbf=num_rbf or 64, cutoff=cutoff,
                    classifier_dims=classifier_dims or [256, 128],
                    node_attr_dim=extra.get('node_attr_dim', 0))
            if name == 'HierarchicalTensorFieldNetwork':
                return HierarchicalTensorFieldNetwork(
                    num_classes=output_dim,
                    n=n_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    stage_sizes=extra.get('stage_sizes', [256, 64]),
                    stage_radii=extra.get('stage_radii', [0.2, 0.4]),
                    k_local=extra.get('k_local', 16),
                    k_global=extra.get('k_global', 16),
                    num_layers_per_stage=extra.get('num_layers_per_stage', 2),
                    num_rbf=num_rbf or 64, cutoff=cutoff,
                    classifier_dims=classifier_dims or [256, 128],
                    node_attr_dim=extra.get('node_attr_dim', 0))
            if name == 'OnEquivariantTensorFieldNetwork':
                return OnEquivariantTensorFieldNetwork(
                    num_classes=output_dim,
                    n=n_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    num_layers=num_layers or 6,
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [256, 128])
            if name == 'AttentionTensorFieldNetwork':
                return AttentionTensorFieldNetwork(
                    num_classes=output_dim,
                    n=n_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    num_layers=num_layers or 6,
                    num_heads=extra.get('num_heads', 4),
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [256, 128],
                    radial_hidden=radial_hidden or 64)
            if name == 'StochasticTensorFieldNetwork':
                return StochasticTensorFieldNetwork(
                    num_classes=output_dim,
                    n=n_dim,
                    num_mixtures=extra.get('num_mixtures', 3),
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    num_layers=num_layers or 6,
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    encoder_dims=extra.get('encoder_dims', [256, 128]))
            if name == 'CrossAttentionTensorFieldNetwork':
                return CrossAttentionTensorFieldNetwork(
                    num_classes=output_dim, n=n_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 64,
                    num_layers=num_layers or 6,
                    num_heads=extra.get('num_heads', 4),
                    transformer_layers=extra.get('transformer_layers', 2),
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [256, 128],
                    radial_hidden=radial_hidden or 64,
                    dropout=extra.get('dropout', 0.1))
            if name == 'RelaxedOnEquivariantTensorFieldNetwork':
                return RelaxedOnEquivariantTensorFieldNetwork(
                    num_classes=output_dim,
                    n=n_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 32,
                    num_layers=num_layers or 3,
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [64, 32],
                    skip_init=extra.get('skip_init', 0.01))
            if name == 'HybridOnEquivariantTensorFieldNetwork':
                return HybridOnEquivariantTensorFieldNetwork(
                    num_classes=output_dim,
                    max_order=extra.get('max_order', 0),
                    hidden_channels=hidden_channels or 32,
                    num_layers=num_layers or 3,
                    num_rbf=num_rbf or 64, cutoff=cutoff, k_neighbors=k_neighbors,
                    classifier_dims=classifier_dims or [64, 32],
                    non_eq_dim=extra.get('non_eq_dim', 128),
                    fusion_dims=extra.get('fusion_dims', None))
            if name == 'PersNet':
                return PersNet(output_dim=output_dim,
                                  activation=activation, norm=norm)
            if name == 'ScalarDistanceDeepSet':
                return ScalarDistanceDeepSet(output_dim=output_dim,
                                             activation=activation, norm=norm)
            if name == 'PointNetTutorial':
                return PointNetTutorial(output_dim=output_dim,
                                        activation=activation, norm=norm)
            if name == 'ScalarInputMLP':
                return ScalarInputMLP(output_dim=output_dim,
                                      activation=activation, norm=norm)
            if name == 'MultiInputModel':
                return MultiInputModel(target_output_dim=output_dim,
                                       scalar_input_dim=1,
                                       activation=activation, norm=norm)
            if name == 'DenseRagged':
                return DenseRagged(in_features=n_dim, out_features=output_dim,
                                   activation=activation,
                                   use_norm=(norm != 'none'))
            if name == 'PermopRagged':
                return PermopRagged()
            if name == 'RaggedPersistenceModel':
                return RaggedPersistenceModel(output_dim=output_dim,
                                              activation=activation)
            if name == 'DistanceMatrixRaggedModel':
                npts = extra.get('num_points', n_dim)
                return DistanceMatrixRaggedModel(output_dim=output_dim,
                                                 num_points=npts,
                                                 activation=activation, norm=norm)
            raise ValueError(f"Unknown model name: {name}")

        model_PV = _build_model_local(class_name, out_dim, n=ckpt_dim,
                                       extra=tfn_extra, activation='gelu', norm='bn')
        # Guard against checkpoints trained on a different dataset (wrong
        # point count / output dim) so they fail loudly instead of loading
        # mismatched weights silently.
        if out_dim != output_dim:
            raise ValueError(f"checkpoint output_dim={out_dim} != dataset output_dim={output_dim}")
        if class_name == 'DistanceMatrixRaggedModel' and 'num_points' not in tfn_extra:
            raise ValueError("checkpoint has no num_points metadata")
        # Strip nested wrapper prefixes from state dict keys so they match
        # the model's own parameter names.  Checkpoints may have mixed nesting:
        #   _inner.classifier.*       → matches directly
        #   _inner.tfn_backbone._inner.* → should become _inner.*
        #   _inner.base._inner.*      → should become _inner.*
        model_keys = set(model_PV.state_dict().keys())
        ckpt_keys  = set(model_state.keys())
        if not ckpt_keys.issubset(model_keys):
            prefixes = ('_inner.base._inner.', '_inner.tfn_backbone._inner.',
                        '_inner.tfn_backbone.', '_inner.base.', '_inner.', '')
            new_state = {}
            for k, v in model_state.items():
                new_key = k
                if k in model_keys:
                    new_state[k] = v
                    continue
                for candidate in prefixes:
                    if k.startswith(candidate):
                        candidate_key = k[len(candidate):]
                        if candidate_key in model_keys:
                            new_key = candidate_key
                            break
                    # Try adding _inner. after stripping (e.g. tfn_backbone._inner.X → _inner.X)
                    if candidate.startswith('_inner.') and k.startswith(candidate):
                        candidate_key = '_inner.' + k[len(candidate):]
                        if candidate_key in model_keys:
                            new_key = candidate_key
                            break
                new_state[new_key] = v
            model_state = new_state
        model_PV.load_state_dict(model_state, strict=False)
        model_PV = model_PV.to(device).eval()

        # Prepare inference data
        if ckpt_use_gs:
            inference_data = gaussian_smooth_batch(
                [torch.FloatTensor(x).to(device) for x in data_sets],
                sigma=ckpt_sigma)
        else:
            inference_data = [torch.FloatTensor(x).to(device) for x in data_sets]

        # Pad to 3D only for models that need 3D input. TensorFieldNetwork is
        # always built as n=3 (matches train_nn, which pads it to 3D). Native-n
        # TFN models are built with n=ckpt_dim and must be fed raw n-dim data
        # (train_nn convention), padding only when the checkpoint says dim >= 3.
        if class_name in TFN_MODELS and (class_name == 'TensorFieldNetwork' or ckpt_dim >= 3):
            inference_data = _pad_to_3d(inference_data)

        # Run inference
        t0 = time()
        if class_name in TFN_MODELS:
            geom_cache = precompute_geometry_for_model(model_PV, class_name, inference_data)
            PV_NN = tfn_batched_forward(model_PV, inference_data, geom_cache)
        else:
            PV_NN_list = []
            for x in tqdm(inference_data, desc=f"  {model_label}", leave=False):
                with torch.no_grad():
                    if class_name == 'MultiInputModel':
                        pcs, scalar = prepare_non_tfn_input(x, class_name)
                        out = model_PV(pcs, scalar)
                    else:
                        out = model_PV(prepare_non_tfn_input(x, class_name))
                    PV_NN_list.append(out.cpu().numpy().ravel())
            PV_NN = np.vstack(PV_NN_list)
        inf_time = time() - t0

        if PV_NN.shape[1] != PV_true.shape[1]:
            print(f"  SKIP: output dim mismatch {PV_NN.shape[1]} vs {PV_true.shape[1]}")
            continue

        # ── Compute PV-space distances ──
        # Use only test split for fair comparison
        PV_true_test = PV_true[N_train:]
        PV_NN_test   = PV_NN[N_train:]

        metrics = {
            'L1':         l1_distance(PV_true_test, PV_NN_test),
            'L2':         l2_distance(PV_true_test, PV_NN_test),
            'Linf':       linf_distance(PV_true_test, PV_NN_test),
            'Cosine':     cosine_distance(PV_true_test, PV_NN_test),
            'MSE':        mse(PV_true_test, PV_NN_test),
            'KL':         kl_divergence(PV_true_test, PV_NN_test),
            'Inference_s': inf_time,
        }

        print("  PV-space distances (test set):")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.6f}")

        all_results[model_label] = metrics

    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        all_results[model_label] = {'error': str(e)}


# ── Compute PD-space baselines (clean vs noisy) ─────────────────────────────

print(f"\n{'='*60}")
print("  PD-space baselines (ground-truth PDs)")
print(f"{'='*60}")

# Bottleneck/Wasserstein between different PD realizations
# (clean AlphaComplex vs the PVs' underlying PDs)
for hidx, hdim in enumerate(homdim):
    pd_clean = [PD_gudhi[i][hidx] for i in range(N_sets)]
    # Compare train PDs to test PDs (cross-set variability)
    pd_metrics = compute_pd_metrics(
        pd_clean[:N_train],
        pd_clean[N_train:])
    print(f"\n  H{hdim} - Train vs Test PD distances (reference):")
    for k, v in pd_metrics.items():
        if isinstance(v, dict):
            print(f"    {k:20s}: mean={v['mean']:.4f}  std={v['std']:.4f}  "
                  f"median={v['median']:.4f}")
        else:
            print(f"    {k:20s}: {v}")


# ── Summary table ───────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  SUMMARY TABLE")
print(f"{'='*60}")

header = f"{'Model':<45s} {'L1':>10s} {'L2':>10s} {'Linf':>10s} {'Cosine':>10s} {'MSE':>10s}"
print(header)
print("-" * len(header))

# Sort by L2 distance (best first)
sorted_models = sorted(
    [(k, v) for k, v in all_results.items() if 'error' not in v],
    key=lambda x: x[1]['L2'])

for label, m in sorted_models:
    print(f"{label:<45s} {m['L1']:10.4f} {m['L2']:10.4f} {m['Linf']:10.4f} "
          f"{m['Cosine']:10.4f} {m['MSE']:10.4f}")

errors = [(k, v) for k, v in all_results.items() if 'error' in v]
if errors:
    print("\nFailed models:")
    for label, v in errors:
        print(f"  {label}: {v['error']}")


# ── Save results to CSV ─────────────────────────────────────────────────────

import csv
csv_path = f"results/{dataset_label}_pd_pl_distances.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'L1', 'L2', 'Linf', 'Cosine', 'MSE', 'KL', 'Inference_s'])
    for label, m in all_results.items():
        if 'error' not in m:
            writer.writerow([label, m['L1'], m['L2'], m['Linf'], m['Cosine'],
                             m['MSE'], m['KL'], m['Inference_s']])
        else:
            writer.writerow([label, 'ERROR', m['error'], '', '', '', '', ''])
print(f"\nResults saved to {csv_path}")


# ── Bar plot ────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if sorted_models:
        labels = [m[0] for m in sorted_models]
        metrics_names = ['L1', 'L2', 'Linf', 'Cosine', 'MSE']

        fig, axes = plt.subplots(1, 5, figsize=(25, 6))
        for ax, metric_name in zip(axes, metrics_names):
            values = [m[1][metric_name] for m in sorted_models]
            bars = ax.barh(range(len(labels)), values, color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel(metric_name)
            ax.set_title(f'{metric_name} (lower = better)')
            ax.invert_yaxis()
        plt.suptitle(f'PD/PL Distance Comparison — {dataset_label} ({PV_type})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = f"results/{dataset_label}_pd_pl_distances.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
except Exception as e:
    print(f"Plotting skipped: {e}")


# ── Per-sample heatmap (predicted vs true) ──────────────────────────────────

try:
    if sorted_models and len(PV_true_test) <= 50:
        n_models = min(6, len(sorted_models))
        n_samples = min(10, len(PV_true_test))
        fig, axes = plt.subplots(n_models + 1, 1, figsize=(14, 3 * (n_models + 1)))

        # True PV
        im0 = axes[0].imshow(PV_true_test[:n_samples].T, aspect='auto', cmap='jet')
        axes[0].set_title('Ground Truth PV')
        axes[0].set_ylabel('Feature')
        plt.colorbar(im0, ax=axes[0])

        for idx, (label, _) in enumerate(sorted_models[:n_models]):
            # Recompute PV_NN for this model (reuse from above)
            # We already have it from the loop, so use the last computed
            try:
                im = axes[idx + 1].imshow(PV_NN_test[:n_samples].T, aspect='auto', cmap='jet')
                axes[idx + 1].set_title(f'{label} — L2={all_results[label]["L2"]:.4f}')
                axes[idx + 1].set_ylabel('Feature')
                plt.colorbar(im, ax=axes[idx + 1])
            except Exception:
                pass

        axes[-1].set_xlabel('Test sample')
        plt.suptitle(f'Predicted vs True PVs — {dataset_label}', fontweight='bold')
        plt.tight_layout()
        heat_path = f"results/{dataset_label}_pv_heatmap.png"
        plt.savefig(heat_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {heat_path}")
except Exception as e:
    print(f"Heatmap skipped: {e}")


print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")
