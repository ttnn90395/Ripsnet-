#!/usr/bin/env python3
"""
Ensemble analysis: load TFN checkpoints from multiple sweep configs,
average their predicted PVs, and classify with XGBoost.

Usage:
    python analysis_ensemble.py <dataset_PV_params> <dataset_train> <dataset_test> <normalize> <PV_type> <mode> [model_tag1 model_tag2 ...]

Example:
    python analysis_ensemble.py ... "sweep_default" "sweep_hC64" "sweep_hC128" "sweep_deep" "sweep_cosine"
"""

import os, sys, json, importlib.util
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.makedirs('results', exist_ok=True)

from models import (
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    HierarchicalGTTFN, HierarchicalTensorFieldNetwork,
    OnEquivariantTensorFieldNetwork, PointNet3D,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DenseRagged, PermopRagged, RaggedPersistenceModel, DistanceMatrixRaggedModel,
)

MODEL_NAMES = [
    'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
    'HierarchicalGTTFN', 'HierarchicalTensorFieldNetwork',
    'OnEquivariantTensorFieldNetwork', 'PointNet3D',
    'ScalarDistanceDeepSet', 'PointNetTutorial', 'ScalarInputMLP', 'MultiInputModel',
    'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel', 'DistanceMatrixRaggedModel',
]

TFN_MODELS = {
    'TensorFieldNetwork', 'GTTensorFieldNetwork',
    'GTTensorFieldNetworkV2', 'HierarchicalGTTFN',
}

# -------------------------------------------------------------------------
# Import functions from analysis_nn.py
# -------------------------------------------------------------------------
an = importlib.import_module('analysis_nn')

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
dataset_PV_params  = sys.argv[1]
dataset_train_name = sys.argv[2]
dataset_test_name  = sys.argv[3]
normalize          = int(sys.argv[4])
PV_type            = sys.argv[5]
mode               = sys.argv[6]
model_tags         = sys.argv[7:] if len(sys.argv) > 7 else []

if not model_tags:
    model_tags = ['sweep_default', 'sweep_hC64', 'sweep_hC128', 'sweep_deep', 'sweep_cosine']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Ensemble tags: {model_tags}')

# -------------------------------------------------------------------------
# Load datasets
# -------------------------------------------------------------------------
with open(f'datasets/{dataset_train_name}.pkl', 'rb') as f:
    data_classif_train, label_classif_train, dim, PV_params, homdim, PV_size = \
        __import__('dill').load(f)
with open(f'datasets/{dataset_test_name}.pkl', 'rb') as f:
    data_classif_test, label_classif_test = __import__('dill').load(f)

N_sets         = len(data_classif_train) + len(data_classif_test)
data_sets      = data_classif_train + data_classif_test

print(f"Train: {len(data_classif_train)}  Test: {len(data_classif_test)}  "
      f"Dim: {dim}  N_homdim: {len(homdim)}  PV_size: {PV_size}  Type: {PV_type}")

le      = LabelEncoder().fit(np.concatenate([label_classif_train, label_classif_test]))
y_train = le.transform(label_classif_train)
y_test  = le.transform(label_classif_test)

# -------------------------------------------------------------------------
# Gudhi baselines (same as analysis_nn.py)
# -------------------------------------------------------------------------
print("\n--- Computing Gudhi baselines ---")
import gudhi as gd
from gudhi.representations import PersistenceImage, Landscape, DiagramSelector
from scipy.spatial import distance
import velour

PD_gudhi = []
import gudhi as gd
for i, pc in enumerate(data_sets):
    rcX = gd.AlphaComplex(points=pc).create_simplex_tree()
    rcX.persistence()
    final_dg = []
    for hdim in homdim:
        dg = rcX.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        final_dg.append(dg)
    PD_gudhi.append(final_dg)

PV_gudhi = []
for hidx in range(len(homdim)):
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

noise_PD_gudhi = []
for i, pc in enumerate(data_sets):
    m_dtm, p_dtm, dimension_max = 0.05, 2, 2
    if dataset_PV_params[:5] == 'synth':
        st_DTM = velour.AlphaDTMFiltration(pc, m_dtm, p_dtm, dimension_max)
    else:
        st_DTM = velour.DTMFiltration(pc, m_dtm, p_dtm, dimension_max)
    st_DTM.persistence()
    final_dg = []
    for hdim in homdim:
        dg = st_DTM.persistence_intervals_in_dimension(hdim)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        final_dg.append(dg)
    noise_PD_gudhi.append(final_dg)

noise_PV_params = []
for hidx in range(len(homdim)):
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

noise_PV_gudhi = []
for hidx in range(len(homdim)):
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

# -------------------------------------------------------------------------
# Ensemble inference
# -------------------------------------------------------------------------
data_sets_torch = [torch.FloatTensor(data_sets[i]).to(device) for i in range(N_sets)]

all_pvs = []  # list of (tag, PV_NN) for each member

for tag in model_tags:
    print(f"\n{'='*60}")
    print(f'Member: {tag}')
    print(f'='*60)

    os.environ['TFN_MODEL_TAG'] = tag
    # Import find_checkpoint, load_and_eval from analysis_nn in a module-specific way
    # We need to rerun the evaluation logic for each tag.
    # The cleanest way: call analysis_nn.main() with modified env.
    # But analysis_nn is not structured as a function. Let's do it inline.

    model_name = 'TensorFieldNetwork'
    label = model_name

    # find checkpoint
    ckpt_path = an.find_checkpoint(label)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state     = checkpoint['model_state_dict']
        output_dim      = checkpoint.get('output_dim')
        ckpt_type       = checkpoint.get('model_type', model_name)
        ckpt_dim        = checkpoint.get('dim', dim)
        ckpt_npts       = checkpoint.get('num_points', None)
        ckpt_use_gs     = checkpoint.get('use_gs', False)
        ckpt_activation = checkpoint.get('activation', None)
        ckpt_norm       = checkpoint.get('norm', None)
    else:
        print(f'  SKIP: no checkpoint for {tag} ({ckpt_path})')
        continue

    hidden_channels = checkpoint.get('hidden_channels', 64)
    num_layers      = checkpoint.get('num_layers', 6)
    num_rbf         = checkpoint.get('num_rbf', 64)
    classifier_dims = checkpoint.get('classifier_dims', [256, 128])

    # Build model
    model_pv = TensorFieldNetwork(
        num_classes=output_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_rbf=num_rbf,
        cutoff=checkpoint.get('cutoff', 0.5),
        k_neighbors=checkpoint.get('k_neighbors', 16),
        classifier_dims=classifier_dims,
    ).to(device)
    model_pv.load_state_dict(model_state, strict=False)
    model_pv.eval()

    compiled = model_pv

    # Precompute geometry
    geom = an.precompute_geometry(model_pv, 'TensorFieldNetwork', data_sets_torch)
    output_dim_actual = output_dim or PV_size * PV_size * len(homdim)

    # Inference
    import time
    t0 = time.time()
    try:
        PV_NN = an.tfn_batched_forward(compiled, data_sets_torch, geom, batch_size=64)
    except Exception as e:
        print(f'  WARNING: batched inference failed – {e}, falling back to per-sample')
        PV_NN_list = []
        for idx, x_tensor in enumerate(data_sets_torch):
            geom_single = (geom['rbf'][idx], geom['gt_edge'][idx], geom['nbr_idx'][idx]) \
                if (isinstance(geom, dict) and geom.get('uniform', False)) \
                else geom[idx]
            out = an.forward_single(compiled, an.prepare_single_input(
                'TensorFieldNetwork', x_tensor, use_gs=False),
                'TensorFieldNetwork', geom=geom_single)
            PV_NN_list.append(out.detach().cpu().numpy())
        PV_NN = np.vstack(PV_NN_list)
    t1 = time.time()
    print(f'  Inference: {t1-t0:.2f}s')

    all_pvs.append((tag, PV_NN))
    print(f'  PV shape: {PV_NN.shape}')

if not all_pvs:
    print('ERROR: no models loaded')
    sys.exit(1)

# -------------------------------------------------------------------------
# Ensemble: average PVs
# -------------------------------------------------------------------------
PV_ensemble = np.mean([pv for _, pv in all_pvs], axis=0)
print(f'\nEnsemble PV shape: {PV_ensemble.shape}')

# -------------------------------------------------------------------------
# Classification: XGBoost on ensemble
# -------------------------------------------------------------------------
PV_train_ens = PV_ensemble[:len(data_classif_train)]
PV_test_ens  = PV_ensemble[len(data_classif_train):]

clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
clf.fit(PV_train_ens, y_train)
ens_train_acc = clf.score(PV_train_ens, y_train)
ens_test_acc  = clf.score(PV_test_ens,  y_test)
print(f'\n[Ensemble] XGB  train={100*ens_train_acc:.2f}%  test={100*ens_test_acc:.2f}%')

# -------------------------------------------------------------------------
# Individual + Gudhi baselines
# -------------------------------------------------------------------------
print('\n--- Individual members ---')
for tag, pv in all_pvs:
    PV_tr = pv[:len(data_classif_train)]
    PV_te = pv[len(data_classif_train):]
    clf_m = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    clf_m.fit(PV_tr, y_train)
    print(f'  [{tag}] XGB  train={100*clf_m.score(PV_tr, y_train):.2f}%  '
          f'test={100*clf_m.score(PV_te, y_test):.2f}%')

PV_train_gudhi       = np.hstack(PV_gudhi)[:len(data_classif_train)]
PV_test_gudhi        = np.hstack(PV_gudhi)[len(data_classif_train):]
noise_PV_train_gudhi = np.hstack(noise_PV_gudhi)[:len(data_classif_train)]
noise_PV_test_gudhi  = np.hstack(noise_PV_gudhi)[len(data_classif_train):]

clf_g = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
clf_g.fit(PV_train_gudhi, y_train)
print(f'\n[Gudhi (clean)] XGB  train={100*clf_g.score(PV_train_gudhi, y_train):.2f}%  '
      f'test={100*clf_g.score(PV_test_gudhi, y_test):.2f}%')

clf_gn = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
clf_gn.fit(noise_PV_train_gudhi, y_train)
print(f'[Gudhi (noise)] XGB  train={100*clf_gn.score(noise_PV_train_gudhi, y_train):.2f}%  '
      f'test={100*clf_gn.score(noise_PV_test_gudhi, y_test):.2f}%')

# Save results
res = {
    'ensemble_test_acc': float(ens_test_acc),
    'ensemble_train_acc': float(ens_train_acc),
    'members': {tag: {'test_acc': float(
        XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        .fit(pv[:len(data_classif_train)], y_train)
        .score(pv[len(data_classif_train):], y_test)
    )} for tag, pv in all_pvs},
    'gudhi_clean_test_acc': float(clf_g.score(PV_test_gudhi, y_test)),
    'gudhi_noise_test_acc': float(clf_gn.score(noise_PV_test_gudhi, y_test)),
}
with open(f'results/{dataset_test_name}_ensemble_results.json', 'w') as f:
    json.dump(res, f, indent=2)
print(f'\nResults saved to results/{dataset_test_name}_ensemble_results.json')
