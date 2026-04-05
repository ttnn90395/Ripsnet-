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
# ensure top repo is in path for models.py import from expes/
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

# Ensure models directory exists
os.makedirs('models', exist_ok=True)
# Ensure results directory exists
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

# Convert data to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_train_torch = [torch.FloatTensor(data_train[i]).to(device) for i in range(len(data_train))]
data_test_torch  = [torch.FloatTensor(data_test[i]).to(device)  for i in range(len(data_test))]

# Normalize the PVs
if normalize:
    for hidx in range(len(homdim)):
        MPV = np.max(PVs_train[hidx])
        PVs_train[hidx] /= MPV
        PVs_test[hidx]  /= MPV

output_dim = sum([PVs_train[hidx].shape[1] for hidx in range(len(homdim))])

# Learning rate and optimizer
if dataset_name[:5] == 'synth':
    optim_lr       = 5e-4
    optimizer_class = optim.Adamax
else:
    optim_lr       = 5e-4
    optimizer_class = optim.Adam

# Convert PVs to PyTorch tensors
PVs_train_torch = [torch.FloatTensor(PVs_train[hidx]).to(device) for hidx in range(len(homdim))]
PVs_test_torch  = [torch.FloatTensor(PVs_test[hidx]).to(device)  for hidx in range(len(homdim))]

# -------------------------------------------------------------------------
# Data preparation helpers
# -------------------------------------------------------------------------

def prepare_data_for_model(model_name, data_list):
    """Return a list of tensors shaped correctly for the given model."""
    if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork',
                      'GTTensorFieldNetworkV2', 'HierarchicalGTTFN']:
        # These models accept n-dim point clouds; pad 2-D to 3-D minimum
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((len(arr), 1), dtype=arr.dtype)], axis=1)
            out.append(torch.FloatTensor(arr).to(device))
        return out

    if model_name in ['PointNetTutorial', 'PointNet3D']:
        # PointNetTutorial uses 2 cols; PointNet3D uses 3 cols
        ncols = 3 if model_name == 'PointNet3D' else 2
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            if arr.shape[1] < ncols:
                pad = np.zeros((len(arr), ncols - arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
            out.append(torch.FloatTensor(arr[:, :ncols]).to(device))
        return out

    if model_name in ['ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel', 'RaggedPersistenceModel']:
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            out.append(torch.FloatTensor(mat).to(device))
        return out

    if model_name == 'ScalarInputMLP':
        # ScalarInputMLP expects a (1,) scalar per sample; use mean pairwise distance
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            scalar = np.array([[mat.mean()]], dtype=np.float32)
            out.append(torch.FloatTensor(scalar).to(device))
        return out

    if model_name == 'MultiInputModel':
        # Returns list of (point_cloud_tensor, scalar_tensor) tuples
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            scalar = torch.FloatTensor([[mat.mean()]]).to(device)
            out.append((torch.FloatTensor(arr).to(device), scalar))
        return out

    # DenseRagged, PermopRagged – pass point cloud directly
    return data_list


def build_model_by_name(name, n=None):
    _n = dim if n is None else n
    # actual point count for distance-matrix models: use first training sample
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
        # pre-init with known in_features so optimizer isn't empty
        return DenseRagged(in_features=_n, out_features=output_dim)
    if name == 'PermopRagged':
        return PermopRagged()
    if name == 'RaggedPersistenceModel':
        # pass in_features so DenseRagged weights are pre-allocated
        return RaggedPersistenceModel(output_dim=output_dim)
    if name == 'DistanceMatrixRaggedModel':
        # use actual point count from data, not a magic constant
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=_npts)
    raise ValueError('Unknown model: ' + name)


# -------------------------------------------------------------------------
# Per-sample forward pass (handles model-specific input signatures)
# -------------------------------------------------------------------------

def forward_single(model, x, mname):
    """Run model on a single prepared sample x; return output tensor."""
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
    """Return a (1, output_dim) float tensor for sample i."""
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
# Train a single named model and save checkpoint
# -------------------------------------------------------------------------

def train_single_model(mname):
    print(f'\n=== Training {mname} ===')
    m = build_model_by_name(mname, n=dim).to(device)
    train_data_model = prepare_data_for_model(mname, data_train_torch)
    test_data_model  = prepare_data_for_model(mname, data_test_torch)

    # Warm-up pass: ensures any lazily-created parameters (DenseRagged) exist
    # before we hand the model to the optimizer.
    m.train()
    with torch.no_grad():
        _ = forward_single(m, train_data_model[0], mname)

    # Move any buffers created during the warm-up to the right device
    m = m.to(device)

    optimizer     = optimizer_class(m.parameters(), lr=optim_lr)
    criterion     = nn.MSELoss()
    targets_train = np.hstack(PVs_train)
    targets_test  = np.hstack(PVs_test)

    for epoch in tqdm(range(num_epochs), desc=f"Epochs ({mname})"):
        tr_loss  = train_epoch(m, train_data_model, targets_train, optimizer, criterion, mname)
        val_loss = evaluate(m, test_data_model, targets_test, criterion, mname)
        if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
            print(f'  [{mname}] epoch {epoch+1}/{num_epochs}  train={tr_loss:.4f}  val={val_loss:.4f}')

    # Save num_points for DistanceMatrixRaggedModel so analysis can restore it
    extra = {}
    if mname == 'DistanceMatrixRaggedModel':
        extra['num_points'] = m._phi_inp_dim

    ckpt_path = f'models/{mname}.pth'
    torch.save({
        'model_state_dict': m.state_dict(),
        'model_type':       mname,
        'output_dim':       output_dim,
        'homdim':           homdim,
        'dim':              dim,
        **extra,
    }, ckpt_path)
    print(f'  Saved → {ckpt_path}')
    return {'train_loss': tr_loss, 'val_loss': val_loss}


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if model_name == 'all':
    results = {}
    for mname in MODEL_NAMES:
        try:
            results[mname] = train_single_model(mname)
        except Exception as e:
            print(f'ERROR training {mname}: {e}')
            results[mname] = {'error': str(e)}
    print('\n=== ALL MODELS RESULTS ===')
    for k, v in results.items():
        print(f'  {k}: {v}')

elif model_name in MODEL_NAMES:
    train_single_model(model_name)

else:
    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Use one of {MODEL_NAMES} or 'all'."
    )
