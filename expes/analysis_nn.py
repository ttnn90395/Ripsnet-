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

# Ensure top repo is in path for models.py import from expes/
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

# -------------------------------------------------------------------------
# CLI args
# -------------------------------------------------------------------------
requested_model    = sys.argv[1]   # model name, 'all', or path
dataset_PV_params  = sys.argv[2]
dataset_train_name = sys.argv[3]
dataset_test_name  = sys.argv[4]
normalize          = int(sys.argv[5])
PV_type            = sys.argv[6]
mode               = sys.argv[7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(sys.argv)

# -------------------------------------------------------------------------
# Resolve which models to test and where their checkpoints live
# -------------------------------------------------------------------------

def find_checkpoint(mname):
    """Return path to checkpoint for mname, or raise FileNotFoundError."""
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
        f"No checkpoint found for model '{mname}'. "
        f"Searched: {candidate_dirs}"
    )


if requested_model == 'all':
    models_to_test = MODEL_NAMES[:]
elif requested_model in MODEL_NAMES:
    models_to_test = [requested_model]
elif os.path.isfile(requested_model):
    models_to_test = [os.path.splitext(os.path.basename(requested_model))[0]]
else:
    # fuzzy search in candidate dirs
    from glob import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        'models',
        os.path.join(script_dir, 'models'),
        os.path.join(script_dir, '..', 'models'),
    ]
    found = []
    for d in candidate_dirs:
        for c in glob(os.path.join(d, f'*{requested_model}*')):
            if c.endswith(('.pt', '.pth')):
                found.append(os.path.splitext(os.path.basename(c))[0])
    if not found:
        raise ValueError(
            f"Unknown model '{requested_model}'. "
            f"Valid names: {MODEL_NAMES} or 'all', or a direct checkpoint path."
        )
    models_to_test = found

print('models_to_test:', models_to_test)

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

def build_analysis_model(name, output_dim, n=None, extra=None):
    n_dim = dim if n is None else n
    extra = extra or {}
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim)
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=n_dim, num_classes=output_dim)
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=n_dim, num_classes=output_dim)
    if name == 'HierarchicalGTTFN':
        return HierarchicalGTTFN(n=n_dim, num_classes=output_dim)
    if name == 'PointNet3D':
        return PointNet3D(output_dim=output_dim)
    if name == 'ScalarDistanceDeepSet':
        return ScalarDistanceDeepSet(output_dim=output_dim)
    if name == 'PointNetTutorial':
        return PointNetTutorial(output_dim=output_dim)
    if name == 'ScalarInputMLP':
        return ScalarInputMLP(output_dim=output_dim)
    if name == 'MultiInputModel':
        return MultiInputModel(target_output_dim=output_dim, scalar_input_dim=1)
    if name == 'DenseRagged':
        return DenseRagged(in_features=n_dim, out_features=output_dim)
    if name == 'PermopRagged':
        return PermopRagged()
    if name == 'RaggedPersistenceModel':
        return RaggedPersistenceModel(output_dim=output_dim)
    if name == 'DistanceMatrixRaggedModel':
        npts = extra.get('num_points', n_dim)
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=npts)
    raise ValueError(f"Unknown model name: {name}")
    if name == 'DistanceMatrixRaggedModel':
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=600)
    raise ValueError(f"Unknown model name: {name}")

# -------------------------------------------------------------------------
# Prepare a single point-cloud tensor for inference
# -------------------------------------------------------------------------

def prepare_single_input(mname, x_tensor):
    """
    x_tensor : (N_pts, dim) FloatTensor on device
    Returns the input expected by forward_single().
    """
    arr = x_tensor.cpu().numpy()

    if mname in ['TensorFieldNetwork', 'GTTensorFieldNetwork',
                 'GTTensorFieldNetworkV2', 'HierarchicalGTTFN']:
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1)
        return torch.FloatTensor(arr).to(device)

    if mname == 'PointNet3D':
        # needs 3 columns; pad if necessary
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
        mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scalar = np.array([[mat.mean()]], dtype=np.float32)
        return torch.FloatTensor(scalar).to(device)

    if mname == 'MultiInputModel':
        mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
        scalar = torch.FloatTensor([[mat.mean()]]).to(device)
        return (torch.FloatTensor(arr).to(device), scalar)

    # DenseRagged, PermopRagged – raw point cloud
    return x_tensor


def forward_single(model, prepared_x, mname):
    """
    Run model on a single prepared sample and return output tensor.
    This is the ONLY place that calls model(); always returns a tensor.
    """
    if mname == 'MultiInputModel':
        pc, scalar = prepared_x
        return model([pc], scalar)

    if mname == 'ScalarInputMLP':
        return model(prepared_x)           # (1, output_dim)

    if mname in [
        'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
        'HierarchicalGTTFN', 'PointNet3D', 'PointNetTutorial',
        'DistanceMatrixRaggedModel', 'ScalarDistanceDeepSet',
        'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel',
    ]:
        return model([prepared_x])         # list-of-one convention

    return model(prepared_x.unsqueeze(0))  # fallback

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
            [PD_gudhi[i][hidx] for i in range(N_sets)]
        )
    )
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
        [noise_PD_gudhi[i][hidx] for i in range(len(data_classif_train))]
    )
    noise_vpdtr = np.vstack(noise_pds_train)
    if PV_type == 'PI':
        pers = pairwise_distances(
            np.hstack([noise_vpdtr[:, 0:1],
                       noise_vpdtr[:, 1:2] - noise_vpdtr[:, 0:1]])[:200]
        ).flatten()
        pers = pers[np.argwhere(pers > 1e-5).ravel()]
        sigma = np.quantile(pers, .2)
        im_bnds = [
            np.quantile(noise_vpdtr[:, 0], 0.),
            np.quantile(noise_vpdtr[:, 0], 1.),
            np.quantile(noise_vpdtr[:, 1] - noise_vpdtr[:, 0], 0.),
            np.quantile(noise_vpdtr[:, 1] - noise_vpdtr[:, 0], 1.),
        ]
        noise_PV_params.append(
            {'bandwidth': sigma, 'weight': [1, 1],
             'resolution': [PV_size, PV_size], 'im_range': im_bnds}
        )
    else:
        noise_pds_all = DiagramSelector(use=True).fit_transform(
            [noise_PD_gudhi[i][hidx] for i in range(N_sets)]
        )
        noise_vpd_all = np.vstack(noise_pds_all)
        sp_bnds = [np.quantile(noise_vpd_all[:, 0], 0.),
                   np.quantile(noise_vpd_all[:, 1], 1.)]
        noise_PV_params.append(
            {'num_landscapes': 5, 'resolution': PV_size, 'sample_range': sp_bnds}
        )

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
            [PD_gudhi[i][hidx] for i in range(N_sets)]
        )
    )
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
            plt.imshow(
                np.flip(np.reshape(PV_gudhi[hidx][len(data_classif_train) + i],
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
            plt.imshow(
                np.flip(np.reshape(noise_PV_gudhi[hidx][len(data_classif_train) + i],
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
# Encode labels (shared across all model evaluations)
# -------------------------------------------------------------------------

le = LabelEncoder().fit(np.concatenate([label_classif_train, label_classif_test]))
y_train = le.transform(label_classif_train)
y_test  = le.transform(label_classif_test)

# -------------------------------------------------------------------------
# Evaluate each model
# -------------------------------------------------------------------------

data_sets_torch = [torch.FloatTensor(data_sets[i]).to(device) for i in range(N_sets)]

all_model_results = {}

for model_name in models_to_test:
    print(f'\n{"="*60}')
    print(f'Evaluating model: {model_name}')
    print(f'{"="*60}')

    # --- load checkpoint ---
    if os.path.isfile(model_name):
        ckpt_path = model_name
    else:
        try:
            ckpt_path = find_checkpoint(model_name)
        except FileNotFoundError as e:
            print(f'  SKIP: {e}')
            all_model_results[model_name] = {'error': str(e)}
            continue

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        output_dim  = checkpoint.get('output_dim')
        ckpt_type   = checkpoint.get('model_type', model_name)
        ckpt_dim    = checkpoint.get('dim', dim)
        ckpt_npts   = checkpoint.get('num_points', None)
    else:
        model_state = checkpoint
        output_dim  = None
        ckpt_type   = model_name
        ckpt_dim    = dim
        ckpt_npts   = None

    # ----------------------------------------------------------------
    # Resolve class_name: trust ckpt_type first (set by train_nn.py).
    # Only fall back to key-inspection for legacy checkpoints where
    # model_type was not saved or was saved as 'best_model'.
    # ----------------------------------------------------------------
    _state_keys = set(model_state.keys())

    # Legacy TFNTensorFieldNetwork: unique key pattern "layers.X.W00..."
    _is_old_tfn = any(k.startswith('layers.') and '.W0' in k for k in _state_keys)

    if _is_old_tfn:
        # Override whatever ckpt_type says — this is the original TFN architecture
        class_name = 'TFNTensorFieldNetwork'
    elif ckpt_type in MODEL_NAMES:
        class_name = ckpt_type
    elif ckpt_type == 'TFNTensorFieldNetwork':
        class_name = 'TFNTensorFieldNetwork'
    elif model_name in MODEL_NAMES:
        class_name = model_name
    else:
        inferred = None
        for candidate in MODEL_NAMES:
            if candidate.lower() in model_name.lower():
                inferred = candidate
                break
        if inferred is None:
            print(f'  SKIP: cannot infer model class for "{model_name}" (ckpt_type="{ckpt_type}")')
            all_model_results[model_name] = {'error': 'cannot infer model class'}
            continue
        class_name = inferred
        print(f'  Inferred class: {class_name}')

    if output_dim is None:
        output_dim = sum(PVs_train[hidx].shape[1] for hidx in range(len(homdim)))

    print(f'  class_name={class_name}  ckpt_type={ckpt_type}  output_dim={output_dim}  dim={ckpt_dim}')

    try:
        if class_name == 'TFNTensorFieldNetwork':
            from models import TFNTensorFieldNetwork
            model_PV = TFNTensorFieldNetwork(num_classes=output_dim)
            model_PV.load_state_dict(model_state)
        elif class_name == 'DistanceMatrixRaggedModel':
            npts = ckpt_npts or (ckpt_dim * (ckpt_dim - 1) // 2)
            model_PV = build_analysis_model(class_name, output_dim, n=ckpt_dim,
                                            extra={'num_points': npts})
            model_PV.load_state_dict(model_state)
        elif class_name == 'RaggedPersistenceModel':
            model_PV = RaggedPersistenceModel(output_dim=output_dim)
            # Layer 0 of DenseRagged is lazy: run one dummy forward pass so
            # weight_param is registered before load_state_dict tries to fill it.
            _dummy_n = list(model_state.keys())
            # infer layer-0 weight shape from checkpoint
            _w0_key = 'ragged_layers.0.weight_param'
            if _w0_key in model_state:
                _w0_shape = model_state[_w0_key].shape  # (in_f, 30)
                _dummy_in = torch.zeros(1, _w0_shape[0])
                model_PV.ragged_layers[0](_dummy_in.unsqueeze(0) if False else [_dummy_in])
            model_PV.load_state_dict(model_state)
        else:
            model_PV = build_analysis_model(class_name, output_dim, n=ckpt_dim)
            model_PV.load_state_dict(model_state)

        model_PV = model_PV.to(device)
        model_PV.eval()
    except Exception as e:
        print(f'  SKIP: failed to load model – {e}')
        all_model_results[model_name] = {'error': str(e)}
        continue

    # --- inference ---
    starttimeNN = time()
    PV_NN_list = []
    with torch.no_grad():
        for x_tensor in tqdm(data_sets_torch, desc=f"Predicting ({model_name})"):
            try:
                prepared = prepare_single_input(class_name, x_tensor)
                out = forward_single(model_PV, prepared, class_name)
                out_np = out.cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
                if out_np.ndim > 1 and out_np.shape[0] == 1:
                    out_np = out_np[0]
                PV_NN_list.append(out_np)
            except Exception as e:
                print(f'  WARNING: inference failed on sample – {e}')
                PV_NN_list.append(np.zeros(output_dim, dtype=np.float32))

    PV_NN    = np.vstack(PV_NN_list)
    timeNN   = time() - starttimeNN
    print(f"  NN inference time: {timeNN:.2f}s  (Gudhi clean: {timeG:.2f}s, noise: {timeGn:.2f}s)")

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
        plt.suptitle(f"Test predicted PV hdim {hdim} ({model_name})")
        plt.savefig(f"results/{dataset_test_name}_{model_name}_predicted_PVs_h{hdim}_on_test.png")
        plt.close()

    # --- XGBoost classification on predicted PVs ---
    PV_train_NN = PV_NN[:len(data_classif_train)]
    PV_test_NN  = PV_NN[len(data_classif_train):]

    try:
        clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        clf.fit(PV_train_NN, y_train)
        train_acc_nn = clf.score(PV_train_NN, y_train)
        test_acc_nn  = clf.score(PV_test_NN,  y_test)
        print(f"  [{model_name}] XGB  train={100*train_acc_nn:.2f}%  test={100*test_acc_nn:.2f}%")
        all_model_results[model_name] = {
            'xgb_train_acc': train_acc_nn,
            'xgb_test_acc':  test_acc_nn,
            'time_nn':       timeNN,
        }
    except Exception as e:
        print(f'  WARNING: XGB classification failed – {e}')
        all_model_results[model_name] = {'error': str(e)}

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
print(f"\nGudhi (XGB)       train={100*train_acc_gudhi:.2f}%  test={100*test_acc_gudhi:.2f}%  t={XG2-XG1:.2f}s")

XGN1 = time()
model_classif_noise = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model_classif_noise.fit(noise_PV_train_gudhi, y_train)
train_acc_noise = model_classif_noise.score(noise_PV_train_gudhi, y_train)
test_acc_noise  = model_classif_noise.score(noise_PV_test_gudhi,  y_test)
XGN2 = time()
print(f"Gudhi-noise (XGB) train={100*train_acc_noise:.2f}%  test={100*test_acc_noise:.2f}%  t={XGN2-XGN1:.2f}s")

XB11 = time()
model_baseline_dtw = KNeighborsClassifier(metric=DTW)
model_baseline_dtw.fit(ts_classif_train, y_train)
train_acc_dtw = model_baseline_dtw.score(ts_classif_train, y_train)
test_acc_dtw  = model_baseline_dtw.score(ts_classif_test,  y_test)
XB12 = time()
print(f"DTW k-NN          train={100*train_acc_dtw:.2f}%  test={100*test_acc_dtw:.2f}%  t={XB12-XB11:.2f}s")

XB21 = time()
model_baseline_euc = KNeighborsClassifier(metric='euclidean')
model_baseline_euc.fit(ts_classif_train, y_train)
train_acc_euc = model_baseline_euc.score(ts_classif_train, y_train)
test_acc_euc  = model_baseline_euc.score(ts_classif_test,  y_test)
XB22 = time()
print(f"Euclidean k-NN    train={100*train_acc_euc:.2f}%  test={100*test_acc_euc:.2f}%  t={XB22-XB21:.2f}s")

# -------------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------------

print('\n' + '='*60)
print('FINAL SUMMARY')
print('='*60)
print(f"{'Model':<35} {'Train%':>7} {'Test%':>7} {'NN time':>9}")
print('-'*60)
for mname, res in all_model_results.items():
    if 'error' in res:
        print(f"  {mname:<33} ERROR: {res['error']}")
    else:
        print(f"  {mname:<33} {100*res['xgb_train_acc']:>6.2f}% {100*res['xgb_test_acc']:>6.2f}%  {res['time_nn']:>7.2f}s")
print('-'*60)
print(f"  {'Gudhi (clean)':<33} {100*train_acc_gudhi:>6.2f}% {100*test_acc_gudhi:>6.2f}%  {XG2-XG1:>7.2f}s")
print(f"  {'Gudhi (noise)':<33} {100*train_acc_noise:>6.2f}% {100*test_acc_noise:>6.2f}%  {XGN2-XGN1:>7.2f}s")
print(f"  {'DTW k-NN':<33} {100*train_acc_dtw:>6.2f}% {100*test_acc_dtw:>6.2f}%  {XB12-XB11:>7.2f}s")
print(f"  {'Euclidean k-NN':<33} {100*train_acc_euc:>6.2f}% {100*test_acc_euc:>6.2f}%  {XB22-XB21:>7.2f}s")
