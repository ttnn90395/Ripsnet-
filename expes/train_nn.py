# Training of the NN

import dill as pck
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import SVG
import gudhi as gd
import gudhi.representations
from tqdm import tqdm
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from sklearn.model_selection import KFold
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

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

dataset_name = sys.argv[1]
model_name   = sys.argv[2]
normalize    = int(sys.argv[3])
num_epochs   = int(sys.argv[4])
PV_type      = sys.argv[5]
mode         = sys.argv[6]

print(sys.argv)

# Load data
data = pck.load(open('datasets/' + dataset_name + ".pkl", 'rb'))
data_train = data["data_train"]
PVs_train  = data["PV_train"]
data_test  = data["data_test"]
PVs_test   = data["PV_test"]
PV_params  = data["PV_params"][0]
homdim     = data["hdims"]

N_sets_train = len(data_train)
N_sets_test  = len(data_test)
PV_size = PV_params['resolution'][0] if PV_type == 'PI' else PV_params['resolution']
dim = data_train[0].shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_train_torch = [torch.FloatTensor(data_train[i]).to(device) for i in range(len(data_train))]
data_test_torch  = [torch.FloatTensor(data_test[i]).to(device)  for i in range(len(data_test))]

if normalize:
    for hidx in range(len(homdim)):
        MPV = np.max(PVs_train[hidx])
        PVs_train[hidx] /= MPV
        PVs_test[hidx]  /= MPV

output_dim = sum([PVs_train[hidx].shape[1] for hidx in range(len(homdim))])

if dataset_name[:5] == 'synth':
    optim_lr        = 5e-4
    optimizer_class = optim.Adamax
else:
    optim_lr        = 5e-4
    optimizer_class = optim.Adam

PVs_train_torch = [torch.FloatTensor(PVs_train[hidx]).to(device) for hidx in range(len(homdim))]
PVs_test_torch  = [torch.FloatTensor(PVs_test[hidx]).to(device)  for hidx in range(len(homdim))]

# -------------------------------------------------------------------------
# Gaussian smoothing
# -------------------------------------------------------------------------

def gaussian_smooth_pointcloud(pc: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """
    Replace each point with a weighted average of all points, where weights
    are Gaussian in pairwise distance.  This smooths the point cloud without
    changing its size or ambient dimension.

    pc    : (N, d) float tensor
    sigma : bandwidth (in the same units as the point coordinates)
    returns: (N, d) smoothed point cloud
    """
    # Pairwise squared distances  (N, N)
    diff  = pc.unsqueeze(0) - pc.unsqueeze(1)          # (N, N, d)
    dist2 = (diff ** 2).sum(dim=-1)                    # (N, N)
    W     = torch.exp(-dist2 / (2 * sigma ** 2))       # (N, N)
    W     = W / W.sum(dim=-1, keepdim=True)            # row-normalise
    return W @ pc                                       # (N, d)


def gaussian_smooth_batch(data_list: list, sigma: float = 0.5) -> list:
    """Apply gaussian_smooth_pointcloud to every tensor in data_list."""
    return [gaussian_smooth_pointcloud(x, sigma=sigma) for x in data_list]


# -------------------------------------------------------------------------
# Device utility
# -------------------------------------------------------------------------

def deep_to(module: nn.Module, target_device) -> nn.Module:
    module = module.to(target_device)
    for submod in module.modules():
        for attr_name, attr_val in list(vars(submod).items()):
            if isinstance(attr_val, torch.Tensor):
                setattr(submod, attr_name, attr_val.to(target_device))
            elif isinstance(attr_val, dict):
                for k, v in list(attr_val.items()):
                    if isinstance(v, torch.Tensor):
                        attr_val[k] = v.to(target_device)
            elif isinstance(attr_val, list):
                for i, v in enumerate(attr_val):
                    if isinstance(v, torch.Tensor):
                        attr_val[i] = v.to(target_device)
    return module

# -------------------------------------------------------------------------
# Data preparation helpers
# -------------------------------------------------------------------------

def prepare_data_for_model(mname, data_list, use_gs=False, gs_sigma=0.5):
    """
    Return a list of tensors shaped correctly for the given model.
    If use_gs=True, Gaussian smoothing is applied to the raw point clouds
    before any model-specific transformation.
    """
    # Gaussian smoothing operates on raw point-cloud tensors (not on distance
    # matrices or scalar summaries), so it must come first.
    if use_gs:
        data_list = gaussian_smooth_batch(data_list, sigma=gs_sigma)

    if mname in ['TensorFieldNetwork', 'GTTensorFieldNetwork',
                 'GTTensorFieldNetworkV2', 'HierarchicalGTTFN']:
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((len(arr), 1), dtype=arr.dtype)], axis=1)
            out.append(torch.FloatTensor(arr).to(device))
        return out

    if mname in ['PointNetTutorial', 'PointNet3D']:
        ncols = 3 if mname == 'PointNet3D' else 2
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            if arr.shape[1] < ncols:
                pad = np.zeros((len(arr), ncols - arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
            out.append(torch.FloatTensor(arr[:, :ncols]).to(device))
        return out

    if mname in ['ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel', 'RaggedPersistenceModel']:
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            out.append(torch.FloatTensor(mat).to(device))
        return out

    if mname == 'ScalarInputMLP':
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            scalar = np.array([[mat.mean()]], dtype=np.float32)
            out.append(torch.FloatTensor(scalar).to(device))
        return out

    if mname == 'MultiInputModel':
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            scalar = torch.FloatTensor([[mat.mean()]]).to(device)
            out.append((torch.FloatTensor(arr).to(device), scalar))
        return out

    return data_list


def build_model_by_name(name, n=None):
    _n    = dim if n is None else n
    _npts = data_train_torch[0].shape[0] if data_train_torch else 128
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim)
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=_n, num_classes=output_dim)
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=_n, num_classes=output_dim)
    if name == 'HierarchicalGTTFN':
        return HierarchicalGTTFN(n=_n, num_classes=output_dim)
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
        return DenseRagged(in_features=_n, out_features=output_dim)
    if name == 'PermopRagged':
        return PermopRagged()
    if name == 'RaggedPersistenceModel':
        return RaggedPersistenceModel(output_dim=output_dim)
    if name == 'DistanceMatrixRaggedModel':
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=_npts)
    raise ValueError('Unknown model: ' + name)


# -------------------------------------------------------------------------
# Per-sample forward pass
# -------------------------------------------------------------------------

def forward_single(model, x, mname):
    if mname == 'MultiInputModel':
        pc, scalar = x
        return model([pc], scalar)
    if mname == 'ScalarInputMLP':
        return model(x)
    if mname in [
        'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
        'HierarchicalGTTFN', 'PointNet3D', 'PointNetTutorial',
        'DistanceMatrixRaggedModel', 'ScalarDistanceDeepSet',
        'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel',
    ]:
        return model([x])
    return model(x.unsqueeze(0))


# -------------------------------------------------------------------------
# Train / evaluate loops
# -------------------------------------------------------------------------

def get_target(targets, i):
    return torch.FloatTensor(targets[i:i + 1]).to(device)


def train_epoch(model, data, targets, optimizer, criterion, mname):
    model.train()
    total_loss = 0.0
    for i in range(len(data)):
        optimizer.zero_grad()
        output = forward_single(model, data[i], mname)
        loss = criterion(output, get_target(targets, i))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)


def evaluate(model, data, targets, criterion, mname):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(len(data)):
            output = forward_single(model, data[i], mname)
            loss = criterion(output, get_target(targets, i))
            total_loss += loss.item()
    return total_loss / len(data)


# -------------------------------------------------------------------------
# Train one model variant (raw or GS)
# -------------------------------------------------------------------------

def train_single_model(mname, use_gs=False, gs_sigma=0.5):
    tag      = '_GS' if use_gs else ''
    label    = f'{mname}{tag}'
    print(f'\n=== Training {label} (gaussian_smooth={use_gs}) ===')

    m = deep_to(build_model_by_name(mname, n=dim), device)
    train_data_model = prepare_data_for_model(mname, data_train_torch,
                                              use_gs=use_gs, gs_sigma=gs_sigma)
    test_data_model  = prepare_data_for_model(mname, data_test_torch,
                                              use_gs=use_gs, gs_sigma=gs_sigma)

    # Warm-up pass so lazily-created params exist before optimizer
    m.train()
    with torch.no_grad():
        _ = forward_single(m, train_data_model[0], mname)
    m = deep_to(m, device)

    optimizer     = optimizer_class(m.parameters(), lr=optim_lr)
    criterion     = nn.MSELoss()
    targets_train = np.hstack(PVs_train)
    targets_test  = np.hstack(PVs_test)

    for epoch in tqdm(range(num_epochs), desc=f"Epochs ({label})"):
        tr_loss  = train_epoch(m, train_data_model, targets_train,
                               optimizer, criterion, mname)
        val_loss = evaluate(m, test_data_model, targets_test, criterion, mname)
        if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
            print(f'  [{label}] epoch {epoch+1}/{num_epochs}'
                  f'  train={tr_loss:.4f}  val={val_loss:.4f}')

    extra = {}
    if mname == 'DistanceMatrixRaggedModel':
        extra['num_points'] = m._phi_inp_dim

    ckpt_path = f'models/{label}.pth'
    torch.save({
        'model_state_dict': m.state_dict(),
        'model_type':       mname,
        'output_dim':       output_dim,
        'homdim':           homdim,
        'dim':              dim,
        'use_gs':           use_gs,
        'gs_sigma':         gs_sigma,
        **extra,
    }, ckpt_path)
    print(f'  Saved → {ckpt_path}')
    return {'train_loss': tr_loss, 'val_loss': val_loss}


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

GS_SIGMA = 0.5  # can be overridden by env var if needed

def run_both(mname):
    """Train raw + GS variants of a single model, return both result dicts."""
    res = {}
    for use_gs in [False, True]:
        tag = '_GS' if use_gs else ''
        try:
            res[mname + tag] = train_single_model(mname, use_gs=use_gs,
                                                   gs_sigma=GS_SIGMA)
        except Exception as e:
            print(f'ERROR training {mname}{tag}: {e}')
            res[mname + tag] = {'error': str(e)}
    return res


if model_name == 'all':
    results = {}
    for mname in MODEL_NAMES:
        results.update(run_both(mname))
    print('\n=== ALL MODELS RESULTS ===')
    for k, v in results.items():
        print(f'  {k}: {v}')

elif model_name in MODEL_NAMES:
    run_both(model_name)

else:
    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Use one of {MODEL_NAMES} or 'all'."
    )
