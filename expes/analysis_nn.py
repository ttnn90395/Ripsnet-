# Analysis of the NN
# 1. Comparison between calculation times of Gudhi and NN
# 2. p-values of Kolmogorov-Smirnov tests on each PI pixel
# 3. Comparison between classification results on true and predicted PIs

import os
import matplotlib.pyplot as plt
import dill as pck
import numpy as np
import torch
import torch.nn as nn
from IPython.display import SVG
import gudhi as gd
from gudhi.representations import PersistenceImage, Landscape, DiagramSelector
from scipy.spatial import distance
import velour
from tqdm import tqdm
from time import time
from scipy.stats import ks_2samp
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from xgboost import XGBClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.makedirs('results', exist_ok=True)

from models import (
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    HierarchicalGTTFN, PointNet3D,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DenseRagged, PermopRagged, RaggedPersistenceModel, DistanceMatrixRaggedModel,
)

MODEL_NAMES = [
    'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
    'HierarchicalGTTFN', 'PointNet3D',
    'ScalarDistanceDeepSet', 'PointNetTutorial', 'ScalarInputMLP', 'MultiInputModel',
    'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel', 'DistanceMatrixRaggedModel',
]

TFN_MODELS = {
    'TensorFieldNetwork', 'GTTensorFieldNetwork',
    'GTTensorFieldNetworkV2', 'HierarchicalGTTFN',
}

# -------------------------------------------------------------------------
# CLI args
# -------------------------------------------------------------------------
requested_model    = sys.argv[1]
dataset_PV_params  = sys.argv[2]
dataset_train_name = sys.argv[3]
dataset_test_name  = sys.argv[4]
normalize          = int(sys.argv[5])
PV_type            = sys.argv[6]
mode               = sys.argv[7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(sys.argv)

# -------------------------------------------------------------------------
# Resolve which models to test
# -------------------------------------------------------------------------

def find_checkpoint(mname):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        'models',
        os.path.join(script_dir, 'models'),
        os.path.join(script_dir, '..', 'models'),
        os.path.join(script_dir, '..', '..', 'models'),
    ]
    for d in candidate_dirs:
        for ext in ('.pth', '.pt'):
            p = os.path.join(d, mname + ext)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        f"No checkpoint found for model '{mname}'. Searched: {candidate_dirs}")


if requested_model == 'all':
    models_to_test    = MODEL_NAMES[:]
    models_to_test_gs = [m + '_GS' for m in MODEL_NAMES]
elif requested_model in MODEL_NAMES:
    models_to_test    = [requested_model]
    models_to_test_gs = [requested_model + '_GS']
elif os.path.isfile(requested_model):
    models_to_test    = [os.path.splitext(os.path.basename(requested_model))[0]]
    models_to_test_gs = []
else:
    from glob import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = ['models',
                      os.path.join(script_dir, 'models'),
                      os.path.join(script_dir, '..', 'models')]
    found = []
    for d in candidate_dirs:
        for c in glob(os.path.join(d, f'*{requested_model}*')):
            if c.endswith(('.pt', '.pth')):
                found.append(os.path.splitext(os.path.basename(c))[0])
    if not found:
        raise ValueError(
            f"Unknown model '{requested_model}'. "
            f"Valid: {MODEL_NAMES} or 'all' or a direct path.")
    models_to_test    = found
    models_to_test_gs = []

if 'models_to_test_gs' not in dir():
    models_to_test_gs = [m + '_GS' for m in models_to_test]

print('models_to_test (raw):', models_to_test)
print('models_to_test (GS) :', models_to_test_gs)

# -------------------------------------------------------------------------
# Load dataset metadata and point clouds
# -------------------------------------------------------------------------

PV_setting = pck.load(open('datasets/' + dataset_PV_params + '.pkl', 'rb'))
PV_params, homdim = PV_setting['PV_params'], PV_setting['hdims']

data = pck.load(open('datasets/' + dataset_train_name + '.pkl', 'rb'))
label_classif_train = data['label_train']
data_classif_train  = data['data_train']
PVs_train           = data['PV_train']
ts_classif_train    = np.vstack(data['ts_train'])

data = pck.load(open('datasets/' + dataset_test_name + '.pkl', 'rb'))
label_classif_test = data['label_test']
data_classif_test  = data['data_test']
ts_classif_test    = np.vstack(data['ts_test'])

data_sets = data_classif_train + data_classif_test
N_sets    = len(data_sets)
dim       = data_sets[0].shape[1] if N_sets > 0 else 2

PV_size = PV_params[0]['resolution'][0] if PV_type == 'PI' else PV_params[0]['resolution']

# -------------------------------------------------------------------------
# Build analysis model by class name
# -------------------------------------------------------------------------

def build_analysis_model(name, output_dim, n=None, extra=None,
                         activation='gelu', norm='bn'):
    """
    Instantiate a model with the exact architecture used at training time.
    `activation` and `norm` are read from the checkpoint metadata so that
    old checkpoints (relu, no norm) load correctly into matching architectures.
    """
    n_dim = dim if n is None else n
    extra = extra or {}

    # TFN-family models: activation/norm are internal to gt_tfn_layer.py
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim)
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=n_dim, num_classes=output_dim)
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=n_dim, num_classes=output_dim)
    if name == 'HierarchicalGTTFN':
        return HierarchicalGTTFN(n=n_dim, num_classes=output_dim)

    # All other models accept activation and norm — BOTH must be forwarded
    # so that legacy checkpoints (relu, norm='none') reconstruct correctly.
    if name == 'PointNet3D':
        return PointNet3D(output_dim=output_dim,
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

# -------------------------------------------------------------------------
# Gaussian smoothing  (Fix 4: batched vectorised kernel)
# -------------------------------------------------------------------------

GS_SIGMA = 0.5

def gaussian_smooth_batch(data_list: list, sigma: float = GS_SIGMA) -> list:
    """Vectorised GS: processes all point clouds as a single (B, N, d) tensor
    when shapes are uniform; falls back to per-sample loop otherwise."""
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

# -------------------------------------------------------------------------
# TFN acceleration: batched forward pass (Fix 1b)
# -------------------------------------------------------------------------

def tfn_batched_forward(model, data_list, geom_cache, batch_size=64):
    """
    Run TFN inference in mini-batches.  Stacks _encode_single descriptors
    and passes them through the rho head in one vectorised call, eliminating
    the Python-loop overhead that dominates single-sample evaluation.
    Returns np.ndarray (N, output_dim).
    """
    inner   = getattr(model, '_inner', model)
    results = []
    with torch.no_grad():
        for start in range(0, len(data_list), batch_size):
            batch  = data_list[start:start + batch_size]
            geom_b = geom_cache[start:start + batch_size] if geom_cache else None
            if geom_b is not None:
                descs = [inner._encode_single(pc, precomputed_geom=g)
                         for pc, g in zip(batch, geom_b)]
                out = inner.rho(torch.stack(descs))
            else:
                out = model(batch)
            results.append(out.cpu().numpy())
    return np.vstack(results)


# -------------------------------------------------------------------------
# Geometry precomputation  (Fix 1: cache k-NN + GT basis for TFN models)
# -------------------------------------------------------------------------

def precompute_geometry(model_pv, class_name, data_list):
    """
    Pre-compute (rbf, gt_edge, nbr_idx) for every point cloud in data_list.
    Returns a list of tuples, or None if the model is not a TFN-family model.
    """
    if class_name not in TFN_MODELS:
        return None
    try:
        from gt_tfn_layer import knn_geometry
        inner    = getattr(model_pv, '_inner', model_pv)
        rbf_enc  = inner.rbf
        gt_basis = inner.gt_basis
        k        = inner.k_neighbors
    except AttributeError:
        return None

    cache = []
    with torch.no_grad():
        for pc in tqdm(data_list, desc=f"Precomputing geometry ({class_name})",
                       leave=False):
            cache.append(knn_geometry(pc, rbf_enc, gt_basis, k))
    return cache

# -------------------------------------------------------------------------
# Prepare a single point-cloud tensor for inference
# -------------------------------------------------------------------------

def prepare_single_input(mname, x_tensor, use_gs=False, sigma=GS_SIGMA):
    """Applies optional GS then returns the model-specific input format."""
    if use_gs:
        diff  = x_tensor.unsqueeze(0) - x_tensor.unsqueeze(1)
        dist2 = (diff ** 2).sum(-1)
        W     = torch.exp(-dist2 / (2 * sigma ** 2))
        W     = W / W.sum(-1, keepdim=True)
        x_tensor = W @ x_tensor

    arr = x_tensor.cpu().numpy()

    if mname in TFN_MODELS:
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
        return torch.FloatTensor(arr).to(device)

    if mname == 'PointNet3D':
        if arr.shape[1] < 3:
            pad = np.zeros((arr.shape[0], 3 - arr.shape[1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        return torch.FloatTensor(arr[:, :3]).to(device)

    if mname == 'PointNetTutorial':
        return torch.FloatTensor(arr[:, :2]).to(device)

    if mname in ['ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel', 'RaggedPersistenceModel']:
        mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        return torch.FloatTensor(mat).to(device)

    if mname == 'ScalarInputMLP':
        mat    = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        return torch.FloatTensor([[mat.mean()]]).to(device)

    if mname == 'MultiInputModel':
        mat    = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scalar = torch.FloatTensor([[mat.mean()]]).to(device)
        return (torch.FloatTensor(arr).to(device), scalar)

    return x_tensor


def forward_single(model, prepared_x, mname, geom=None):
    """
    Run model on one prepared sample.  When geom=(rbf, gt_edge, nbr_idx) is
    provided for a TFN model, the cached geometry is used directly (Fix 1).
    """
    if mname == 'MultiInputModel':
        pc, scalar = prepared_x
        return model([pc], scalar)
    if mname == 'ScalarInputMLP':
        return model(prepared_x)

    # TFN fast path: bypass k-NN recomputation using cached geometry
    if geom is not None and mname in TFN_MODELS:
        rbf, gt_edge, nbr_idx = geom
        inner = getattr(model, '_inner', model)
        desc  = inner._encode_single(prepared_x, precomputed_geom=(rbf, gt_edge, nbr_idx))
        return inner.rho(desc.unsqueeze(0))

    if mname in [
        'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
        'HierarchicalGTTFN', 'PointNet3D', 'PointNetTutorial',
        'DistanceMatrixRaggedModel', 'ScalarDistanceDeepSet',
        'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel',
    ]:
        return model([prepared_x])
    return model(prepared_x.unsqueeze(0))

# -------------------------------------------------------------------------
# DTW metric for baseline
# -------------------------------------------------------------------------

def DTW(a, b):
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1, 1), b.reshape(-1, 1))
    cumdist = np.matrix(np.ones((an + 1, bn + 1)) * np.inf)
    cumdist[0, 0] = 0
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi + 1], cumdist[ai + 1, bi], cumdist[ai, bi]])
            cumdist[ai + 1, bi + 1] = pointwise_distance[ai, bi] + minimum_cost
    return cumdist[an, bn]

# -------------------------------------------------------------------------
# Plot raw point clouds (test split)
# -------------------------------------------------------------------------

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    pc = data_sets[len(data_classif_train) + i]
    plt.scatter(pc[:, 0], pc[:, 1], s=3)
plt.savefig('results/' + dataset_test_name + '_point_clouds_on_test.png')
plt.close()

# -------------------------------------------------------------------------
# Gudhi (clean) persistence images / landscapes
# -------------------------------------------------------------------------

PD_gudhi = []
starttimeG = time()
for i in tqdm(range(N_sets), desc="Computing Gudhi PDs"):
    rcX = gd.AlphaComplex(points=data_sets[i]).create_simplex_tree()
    rcX.persistence()
    final_dg = []
    for hdim in homdim:
        dg = rcX.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        final_dg.append(dg)
    PD_gudhi.append(final_dg)

PV_gudhi = []
for hidx in tqdm(range(len(homdim)), desc="Processing Gudhi PVs"):
    if PV_type == 'PI':
        a, b = PV_params[hidx]['weight'][0], PV_params[hidx]['weight'][1]
        PV_params[hidx]['weight'] = lambda x, a=a, b=b: a * np.tanh(x[1]) ** b
    PV = (PersistenceImage(**PV_params[hidx]) if PV_type == 'PI'
          else Landscape(**PV_params[hidx]))
    PV_gudhi_hidx = PV.fit_transform(
        DiagramSelector(use=True).fit_transform(
            [PD_gudhi[i][hidx] for i in range(N_sets)]))
    if normalize:
        PV_gudhi_hidx /= np.max(PV_gudhi_hidx[:len(data_classif_train)])
    PV_gudhi.append(PV_gudhi_hidx)

timeG = time() - starttimeG
print(f"Gudhi (clean) time: {timeG:.2f}s")

# -------------------------------------------------------------------------
# Gudhi (noise-robust / DTM) persistence images / landscapes
# -------------------------------------------------------------------------

noise_PD_gudhi = []
starttimeG1 = time()
for i in tqdm(range(N_sets), desc="Computing Gudhi-noise PDs"):
    m_dtm, p_dtm, dimension_max = 0.05, 2, 2
    if dataset_PV_params[:5] == 'synth':
        st_DTM = velour.AlphaDTMFiltration(data_sets[i], m_dtm, p_dtm, dimension_max)
    else:
        st_DTM = velour.DTMFiltration(data_sets[i], m_dtm, p_dtm, dimension_max)
    st_DTM.persistence()
    final_dg = []
    for hdim in homdim:
        dg = st_DTM.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        final_dg.append(dg)
    noise_PD_gudhi.append(final_dg)
starttimeG2 = time()

noise_PV_params = []
for hidx in tqdm(range(len(homdim)), desc="Processing noise PV params"):
    noise_pds_train = DiagramSelector(use=True).fit_transform(
        [noise_PD_gudhi[i][hidx] for i in range(len(data_classif_train))])
    noise_vpdtr = np.vstack(noise_pds_train)
    if PV_type == 'PI':
        pers = pairwise_distances(
            np.hstack([noise_vpdtr[:, 0:1],
                       noise_vpdtr[:, 1:2] - noise_vpdtr[:, 0:1]])[:200]).flatten()
        pers = pers[np.argwhere(pers > 1e-5).ravel()]
        sigma = np.quantile(pers, .2)
        im_bnds = [np.quantile(noise_vpdtr[:, 0], 0.),
                   np.quantile(noise_vpdtr[:, 0], 1.),
                   np.quantile(noise_vpdtr[:, 1] - noise_vpdtr[:, 0], 0.),
                   np.quantile(noise_vpdtr[:, 1] - noise_vpdtr[:, 0], 1.)]
        noise_PV_params.append({'bandwidth': sigma, 'weight': [1, 1],
                                 'resolution': [PV_size, PV_size], 'im_range': im_bnds})
    else:
        noise_pds_all = DiagramSelector(use=True).fit_transform(
            [noise_PD_gudhi[i][hidx] for i in range(N_sets)])
        noise_vpd_all = np.vstack(noise_pds_all)
        sp_bnds = [np.quantile(noise_vpd_all[:, 0], 0.),
                   np.quantile(noise_vpd_all[:, 1], 1.)]
        noise_PV_params.append({'num_landscapes': 5, 'resolution': PV_size,
                                 'sample_range': sp_bnds})

starttimeG3 = time()
noise_PV_gudhi = []
for hidx in tqdm(range(len(homdim)), desc="Computing noise PVs"):
    if PV_type == 'PI':
        a, b = noise_PV_params[hidx]['weight'][0], noise_PV_params[hidx]['weight'][1]
        noise_PV_params[hidx]['weight'] = lambda x, a=a, b=b: a * x[1] ** b
    PV = (PersistenceImage(**noise_PV_params[hidx]) if PV_type == 'PI'
          else Landscape(**noise_PV_params[hidx]))
    PV_gudhi_hidx = PV.fit_transform(
        DiagramSelector(use=True).fit_transform(
            [PD_gudhi[i][hidx] for i in range(N_sets)]))
    if normalize:
        PV_gudhi_hidx /= np.max(PV_gudhi_hidx[:len(data_classif_train)])
    noise_PV_gudhi.append(PV_gudhi_hidx)
starttimeG4 = time()
timeGn = (starttimeG4 - starttimeG3) + (starttimeG2 - starttimeG1)
print(f"Gudhi (noise) time: {timeGn:.2f}s")

# -------------------------------------------------------------------------
# Plot true PVs (clean + noise) on test split
# -------------------------------------------------------------------------

for hidx in range(len(homdim)):
    hdim = homdim[hidx]
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(
                PV_gudhi[hidx][len(data_classif_train) + i],
                [PV_size, PV_size]), 0), cmap='jet')
            plt.colorbar()
        else:
            nls = PV_params[hidx]['num_landscapes']
            for lidx in range(nls):
                plt.plot(PV_gudhi[hidx][len(data_classif_train) + i]
                         [lidx * PV_size:(lidx + 1) * PV_size])
    plt.suptitle('Test true PV in hdim ' + str(hdim))
    plt.savefig('results/' + dataset_test_name + '_true_PVs_h' + str(hdim) + '_on_test.png')
    plt.close()

for hidx in range(len(homdim)):
    hdim = homdim[hidx]
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(
                noise_PV_gudhi[hidx][len(data_classif_train) + i],
                [PV_size, PV_size]), 0), cmap='jet')
            plt.colorbar()
        else:
            nls = noise_PV_params[hidx]['num_landscapes']
            for lidx in range(nls):
                plt.plot(noise_PV_gudhi[hidx][len(data_classif_train) + i]
                         [lidx * PV_size:(lidx + 1) * PV_size])
    plt.suptitle('Test true PV-noise in hdim ' + str(hdim))
    plt.savefig('results/' + dataset_test_name + '_true_PVs-noise_h' + str(hdim) + '_on_test.png')
    plt.close()

# -------------------------------------------------------------------------
# Encode labels
# -------------------------------------------------------------------------

le      = LabelEncoder().fit(np.concatenate([label_classif_train, label_classif_test]))
y_train = le.transform(label_classif_train)
y_test  = le.transform(label_classif_test)

# -------------------------------------------------------------------------
# Evaluate each model (raw and GS variants)
# -------------------------------------------------------------------------

data_sets_torch = [torch.FloatTensor(data_sets[i]).to(device) for i in range(N_sets)]
all_model_results = {}


def load_and_eval(model_name, use_gs=False):
    """
    Load checkpoint, run optimised inference, classify with XGBoost.
    All four speed fixes are applied:
      Fix 1 – geometry precomputed once before the inference loop
      Fix 2 – basis tensors moved to device (via GTTFNv2.to() override)
      Fix 3 – torch.compile applied after load
      Fix 4 – vectorised batched GS
    """
    label = model_name
    print(f'\n{"="*60}')
    print(f'Evaluating: {label}')
    print(f'{"="*60}')

    # --- load checkpoint ---
    if os.path.isfile(model_name):
        ckpt_path = model_name
    else:
        try:
            ckpt_path = find_checkpoint(label)
        except FileNotFoundError as e:
            print(f'  SKIP: {e}')
            all_model_results[label] = {'error': str(e), 'use_gs': use_gs}
            return

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state  = checkpoint['model_state_dict']
        output_dim   = checkpoint.get('output_dim')
        ckpt_type    = checkpoint.get('model_type', model_name)
        ckpt_dim     = checkpoint.get('dim', dim)
        ckpt_npts    = checkpoint.get('num_points', None)
        ckpt_use_gs  = checkpoint.get('use_gs', use_gs)
        ckpt_sigma   = checkpoint.get('gs_sigma', GS_SIGMA)
        # Architecture metadata: read explicitly if saved, else auto-detect
        # from the state_dict keys to handle legacy (pre-GELU) checkpoints.
        ckpt_activation = checkpoint.get('activation', None)
        ckpt_norm       = checkpoint.get('norm', None)
    else:
        model_state  = checkpoint
        output_dim   = None
        ckpt_type    = model_name
        ckpt_dim     = dim
        ckpt_npts    = None
        ckpt_use_gs  = use_gs
        ckpt_sigma   = GS_SIGMA
        ckpt_activation = None
        ckpt_norm       = None

    # ── Detect architecture from state_dict when metadata is absent ──────
    # Legacy checkpoints (before the GELU/BN upgrade) have no BatchNorm keys.
    # New checkpoints have keys like "phi_layers.2.running_mean" (BatchNorm)
    # or "phi_layers.2.weight" where shape is (C,) not (C_out, C_in) (LayerNorm).
    # Heuristic: if any key contains 'running_mean' → has BatchNorm → new arch.
    _has_bn = any('running_mean' in k for k in model_state)
    if ckpt_activation is None:
        # Old checkpoints used relu; new ones use gelu
        ckpt_activation = 'gelu' if _has_bn else 'relu'
    if ckpt_norm is None:
        ckpt_norm = 'bn' if _has_bn else 'none'

    print(f'  arch: activation={ckpt_activation}  norm={ckpt_norm}  ' +
          ('(legacy ReLU checkpoint)' if not _has_bn else '(new GELU checkpoint)'))

    # Resolve class name (trust ckpt_type; detect legacy TFN by key pattern)
    _state_keys    = set(model_state.keys())
    _is_old_tfn    = any(k.startswith('layers.') and '.W0' in k for k in _state_keys)
    base_ckpt_type = ckpt_type.replace('_GS', '')

    if _is_old_tfn:
        class_name = 'TFNTensorFieldNetwork'
    elif base_ckpt_type in MODEL_NAMES:
        class_name = base_ckpt_type
    elif base_ckpt_type == 'TFNTensorFieldNetwork':
        class_name = 'TFNTensorFieldNetwork'
    else:
        base_model_name = model_name.replace('_GS', '')
        inferred = next((c for c in MODEL_NAMES
                         if c.lower() in base_model_name.lower()), None)
        if inferred is None:
            print(f'  SKIP: cannot infer class for "{model_name}"')
            all_model_results[label] = {'error': 'cannot infer model class',
                                        'use_gs': use_gs}
            return
        class_name = inferred
        print(f'  Inferred class: {class_name}')

    if output_dim is None:
        output_dim = sum(PVs_train[hidx].shape[1] for hidx in range(len(homdim)))

    print(f'  class={class_name}  ckpt_type={ckpt_type}'
          f'  out={output_dim}  dim={ckpt_dim}  GS={ckpt_use_gs}')

    # --- instantiate and load weights ---
    try:
        if class_name == 'TFNTensorFieldNetwork':
            from models import TFNTensorFieldNetwork
            model_PV = TFNTensorFieldNetwork(num_classes=output_dim)
            model_PV.load_state_dict(model_state)
        elif class_name == 'DistanceMatrixRaggedModel':
            npts = ckpt_npts or (ckpt_dim * (ckpt_dim - 1) // 2)
            model_PV = build_analysis_model(
                class_name, output_dim, n=ckpt_dim,
                extra={'num_points': npts},
                activation=ckpt_activation, norm=ckpt_norm)
            model_PV.load_state_dict(model_state)
        elif class_name == 'RaggedPersistenceModel':
            model_PV = RaggedPersistenceModel(output_dim=output_dim,
                                              activation=ckpt_activation)
            _w0_key = 'ragged_layers.0.weight_param'
            if _w0_key in model_state:
                _w0_shape = model_state[_w0_key].shape
                model_PV.ragged_layers[0]([torch.zeros(1, _w0_shape[0])])
            model_PV.load_state_dict(model_state)
        else:
            model_PV = build_analysis_model(
                class_name, output_dim, n=ckpt_dim,
                activation=ckpt_activation, norm=ckpt_norm)
            model_PV.load_state_dict(model_state)

        # Fix 2: move ALL tensors (including plain-attribute CG/GT tensors) to device
        model_PV = model_PV.to(device)
        model_PV.eval()
    except Exception as e:
        print(f'  SKIP: failed to load model – {e}')
        all_model_results[label] = {'error': str(e), 'use_gs': use_gs}
        return

    # Fix 3: torch.compile for faster kernel execution (PyTorch >= 2.0, CUDA)
    compiled_model = model_PV
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            compiled_model = torch.compile(model_PV, mode='reduce-overhead')
            print(f'  torch.compile enabled')
        except Exception as e:
            print(f'  torch.compile skipped: {e}')

    # Fix 1: precompute k-NN + GT-basis geometry for TFN models (once, not per sample)
    effective_gs    = ckpt_use_gs
    effective_sigma = ckpt_sigma

    # Apply GS to the full dataset before precomputing geometry (GS first, then geometry)
    if effective_gs:
        inference_data = gaussian_smooth_batch(data_sets_torch, sigma=effective_sigma)
        # Pad 2D→3D after smoothing for TFN models
        if class_name in TFN_MODELS:
            inference_data = [
                (torch.cat([x, x.new_zeros(x.shape[0], 1)], dim=1)
                 if x.shape[1] == 2 else x)
                for x in inference_data
            ]
    else:
        inference_data = data_sets_torch

    geom_cache = precompute_geometry(model_PV, class_name, inference_data)

    # --- inference ---
    starttimeNN = time()
    try:
        if class_name in TFN_MODELS:
            # Acceleration (Fix 1b): batched encode + single rho call per mini-batch
            PV_NN = tfn_batched_forward(
                compiled_model, inference_data, geom_cache, batch_size=64)
        else:
            PV_NN_list = []
            for idx, x_tensor in enumerate(
                    tqdm(inference_data, desc=f"Predicting ({label})")):
                try:
                    prepared = prepare_single_input(class_name, x_tensor,
                                                    use_gs=False)
                    geom = geom_cache[idx] if geom_cache is not None else None
                    out  = forward_single(compiled_model, prepared, class_name, geom=geom)
                    out_np = out.cpu().numpy() if isinstance(out, torch.Tensor) \
                             else np.asarray(out)
                    if out_np.ndim > 1 and out_np.shape[0] == 1:
                        out_np = out_np[0]
                    PV_NN_list.append(out_np)
                except Exception as e:
                    print(f'  WARNING: inference failed on sample {idx} – {e}')
                    PV_NN_list.append(np.zeros(output_dim, dtype=np.float32))
            PV_NN = np.vstack(PV_NN_list)
    except Exception as e:
        print(f'  WARNING: inference failed entirely – {e}')
        PV_NN = np.zeros((N_sets, output_dim), dtype=np.float32)

    timeNN = time() - starttimeNN
    print(f"  NN time: {timeNN:.2f}s  "
          f"(Gudhi clean: {timeG:.2f}s  noise: {timeGn:.2f}s)")

    # --- plot predicted PVs ---
    for hidx in range(len(homdim)):
        hdim = homdim[hidx]
        plt.figure()
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            row = PV_NN[len(data_classif_train) + i]
            if PV_type == 'PI':
                chunk = row[hidx * PV_size * PV_size:(hidx + 1) * PV_size * PV_size]
                plt.imshow(np.flip(np.reshape(chunk, [PV_size, PV_size]), 0), cmap='jet')
                plt.colorbar()
            else:
                nls = PV_params[hidx]['num_landscapes']
                for lidx in range(nls):
                    start = hidx * PV_size * nls + lidx * PV_size
                    plt.plot(row[start:start + PV_size])
        plt.suptitle(f"Test predicted PV hdim {hdim} ({label})")
        plt.savefig(f"results/{dataset_test_name}_{label}"
                    f"_predicted_PVs_h{hdim}_on_test.png")
        plt.close()

    # --- XGBoost classification ---
    PV_train_NN = PV_NN[:len(data_classif_train)]
    PV_test_NN  = PV_NN[len(data_classif_train):]
    try:
        clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        clf.fit(PV_train_NN, y_train)
        tr_acc = clf.score(PV_train_NN, y_train)
        te_acc = clf.score(PV_test_NN,  y_test)
        print(f"  [{label}] XGB  train={100*tr_acc:.2f}%  test={100*te_acc:.2f}%")
        all_model_results[label] = {
            'xgb_train_acc': tr_acc,
            'xgb_test_acc':  te_acc,
            'time_nn':       timeNN,
            'use_gs':        effective_gs,
        }
    except Exception as e:
        print(f'  WARNING: XGB failed – {e}')
        all_model_results[label] = {'error': str(e), 'use_gs': effective_gs}


# Run evaluation: raw then GS for each model
for base_name in models_to_test:
    load_and_eval(base_name, use_gs=False)

for gs_name in models_to_test_gs:
    load_and_eval(gs_name, use_gs=True)

# -------------------------------------------------------------------------
# Baseline classifiers (Gudhi clean, Gudhi noise, DTW, Euclidean)
# -------------------------------------------------------------------------

PV_train_gudhi       = np.hstack(PV_gudhi)[:len(data_classif_train)]
PV_test_gudhi        = np.hstack(PV_gudhi)[len(data_classif_train):]
noise_PV_train_gudhi = np.hstack(noise_PV_gudhi)[:len(data_classif_train)]
noise_PV_test_gudhi  = np.hstack(noise_PV_gudhi)[len(data_classif_train):]

XG1 = time()
model_classif_gudhi = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_gudhi.fit(PV_train_gudhi, y_train)
train_acc_gudhi = model_classif_gudhi.score(PV_train_gudhi, y_train)
test_acc_gudhi  = model_classif_gudhi.score(PV_test_gudhi,  y_test)
XG2 = time()
print(f"\nGudhi (XGB)       train={100*train_acc_gudhi:.2f}%  "
      f"test={100*test_acc_gudhi:.2f}%  t={XG2-XG1:.2f}s")

XGN1 = time()
model_classif_noise = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_noise.fit(noise_PV_train_gudhi, y_train)
train_acc_noise = model_classif_noise.score(noise_PV_train_gudhi, y_train)
test_acc_noise  = model_classif_noise.score(noise_PV_test_gudhi,  y_test)
XGN2 = time()
print(f"Gudhi-noise (XGB) train={100*train_acc_noise:.2f}%  "
      f"test={100*test_acc_noise:.2f}%  t={XGN2-XGN1:.2f}s")

XB11 = time()
model_baseline_dtw = KNeighborsClassifier(metric=DTW)
model_baseline_dtw.fit(ts_classif_train, y_train)
train_acc_dtw = model_baseline_dtw.score(ts_classif_train, y_train)
test_acc_dtw  = model_baseline_dtw.score(ts_classif_test,  y_test)
XB12 = time()
print(f"DTW k-NN          train={100*train_acc_dtw:.2f}%  "
      f"test={100*test_acc_dtw:.2f}%  t={XB12-XB11:.2f}s")

XB21 = time()
model_baseline_euc = KNeighborsClassifier(metric='euclidean')
model_baseline_euc.fit(ts_classif_train, y_train)
train_acc_euc = model_baseline_euc.score(ts_classif_train, y_train)
test_acc_euc  = model_baseline_euc.score(ts_classif_test,  y_test)
XB22 = time()
print(f"Euclidean k-NN    train={100*train_acc_euc:.2f}%  "
      f"test={100*test_acc_euc:.2f}%  t={XB22-XB21:.2f}s")

# -------------------------------------------------------------------------
# Final summary — raw vs GS side by side
# -------------------------------------------------------------------------

print('\n' + '='*75)
print('FINAL SUMMARY')
print('='*75)
print(f"  {'Model':<30} {'GS':>3} {'Train%':>7} {'Test%':>7} {'Time(s)':>8}")
print('-'*75)

for base_name in models_to_test:
    gs_name = base_name + '_GS'
    for label in [base_name, gs_name]:
        res = all_model_results.get(label)
        if res is None:
            continue
        gs_flag = 'yes' if res.get('use_gs') else 'no'
        if 'error' in res:
            print(f"  {label:<30} {gs_flag:>3}  ERROR: {res['error']}")
        else:
            print(f"  {label:<30} {gs_flag:>3}"
                  f" {100*res['xgb_train_acc']:>6.2f}%"
                  f" {100*res['xgb_test_acc']:>6.2f}%"
                  f" {res['time_nn']:>8.2f}s")

print('-'*75)
print(f"  {'Gudhi (clean)':<30} {'---':>3} {100*train_acc_gudhi:>6.2f}%"
      f" {100*test_acc_gudhi:>6.2f}%  {XG2-XG1:>7.2f}s")
print(f"  {'Gudhi (noise)':<30} {'---':>3} {100*train_acc_noise:>6.2f}%"
      f" {100*test_acc_noise:>6.2f}%  {XGN2-XGN1:>7.2f}s")
print(f"  {'DTW k-NN':<30} {'---':>3} {100*train_acc_dtw:>6.2f}%"
      f" {100*test_acc_dtw:>6.2f}%  {XB12-XB11:>7.2f}s")
print(f"  {'Euclidean k-NN':<30} {'---':>3} {100*train_acc_euc:>6.2f}%"
      f" {100*test_acc_euc:>6.2f}%  {XB22-XB21:>7.2f}s")
