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
    TensorFieldNetwork, GTTensorFieldNetwork, HierarchicalGTTFN,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DenseRagged, PermopRagged, RaggedPersistenceModel, DistanceMatrixRaggedModel,
)

MODEL_NAMES = [
    'TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN',
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
PVs_train = data["PV_train"]
data_test = data["data_test"]
PVs_test = data["PV_test"]
PV_params = data["PV_params"][0]
homdim = data["hdims"]

N_sets_train = len(data_train)
N_sets_test = len(data_test)
PV_size = PV_params['resolution'][0] if PV_type == 'PI' else PV_params['resolution'] 
dim = data_train[0].shape[1]

# Convert data to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_train_torch = [torch.FloatTensor(data_train[i]).to(device) for i in range(len(data_train))]
data_test_torch = [torch.FloatTensor(data_test[i]).to(device) for i in range(len(data_test))]

# Normalize the PIs
if normalize:
    for hidx in range(len(homdim)):
        MPV = np.max(PVs_train[hidx])
        PVs_train[hidx] /= MPV
        PVs_test[hidx]  /= MPV

output_dim = sum([PVs_train[hidx].shape[1] for hidx in range(len(homdim))])

# Learning rate and optimizer class must be defined before training helpers are used
if dataset_name[:5] == 'synth':
    optim_lr = 5e-4
    optimizer_class = optim.Adamax
else:
    optim_lr = 5e-4
    optimizer_class = optim.Adam

# Mapping from model class name to checkpoint prefix (for analysis_nn.py to infer later)
# Format: checkpoint_name is always (model_prefix)_(dataset_info)
MODEL_TYPE_PREFIXES = {
    'RaggedPersistenceModel': 'ripsnet',
    'TensorFieldNetwork': 'tfn',
    'GTTensorFieldNetwork': 'gttfn',
    'HierarchicalGTTFN': 'hierarchical',
}

# Convert PVs to PyTorch tensors
PVs_train_torch = [torch.FloatTensor(PVs_train[hidx]).to(device) for hidx in range(len(homdim))]
PVs_test_torch = [torch.FloatTensor(PVs_test[hidx]).to(device) for hidx in range(len(homdim))]

# #######################
# Generic model helpers
# #######################

def to_3d_tensor_list(data_list):
    out = []
    for x in data_list:
        arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((len(arr), 1), dtype=arr.dtype)], axis=1)
        out.append(torch.FloatTensor(arr).to(device))
    return out


def prepare_data_for_model(model_name, data_list):
    # For TFN models, data may already be n-dimensional; only upgrade 2D vectors to 3D.
    if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN']:
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((len(arr), 1), dtype=arr.dtype)], axis=1)
            out.append(torch.FloatTensor(arr).to(device))
        return out
    if model_name == 'PointNetTutorial':
        # 2D only
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            out.append(torch.FloatTensor(arr[:, :2]).to(device))
        return out
    if model_name in ['ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel', 'RaggedPersistenceModel']:
        out = []
        for x in data_list:
            arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
            mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
            out.append(torch.FloatTensor(mat).to(device))
        return out
    # For simple ragged models, use point cloud list directly
    return data_list


def build_model_by_name(name, n=None):
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim)
    if name == 'GTTensorFieldNetwork':
        n_dim = dim if n is None else n
        return GTTensorFieldNetwork(n=n_dim, num_classes=output_dim)
    if name == 'HierarchicalGTTFN':
        n_dim = dim if n is None else n
        return HierarchicalGTTFN(n=n_dim, num_classes=output_dim)
    if name == 'ScalarDistanceDeepSet':
        return ScalarDistanceDeepSet(output_dim=output_dim)
    if name == 'PointNetTutorial':
        return PointNetTutorial(output_dim=output_dim)
    if name == 'ScalarInputMLP':
        return ScalarInputMLP(output_dim=output_dim)
    if name == 'MultiInputModel':
        return MultiInputModel(target_output_dim=output_dim, scalar_input_dim=1)
    if name == 'DenseRagged':
        return DenseRagged(in_features=None, out_features=output_dim)
    if name == 'PermopRagged':
        return PermopRagged()
    if name == 'RaggedPersistenceModel':
        return RaggedPersistenceModel(output_dim=output_dim)
    if name == 'DistanceMatrixRaggedModel':
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=600)
    raise ValueError('Unknown model: ' + name)


def train_epoch(model, data, targets, optimizer, criterion, model_name):
    model.train()
    total_loss = 0
    for i in range(len(data)):
        optimizer.zero_grad()
        x = data[i]
        if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN', 'PointNetTutorial', 'DistanceMatrixRaggedModel']:
            inp = [x]
        else:
            inp = [x] if isinstance(x, torch.Tensor) and x.ndim > 1 and x.shape[0] > 1 else x
        output = model(inp) if isinstance(inp, list) else model(inp)
        target = torch.FloatTensor(targets[i:i+1]).to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)


def evaluate(model, data, targets, criterion, model_name):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(len(data)):
            x = data[i]
            if model_name in ['TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN', 'PointNetTutorial', 'DistanceMatrixRaggedModel']:
                inp = [x]
            else:
                inp = [x] if isinstance(x, torch.Tensor) and x.ndim > 1 and x.shape[0] > 1 else x
            output = model(inp) if isinstance(inp, list) else model(inp)
            target = torch.FloatTensor(targets[i:i+1]).to(device)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(data)


def train_all_models():
    model_names = [
        'TensorFieldNetwork', 'GTTensorFieldNetwork', 'HierarchicalGTTFN',
        'ScalarDistanceDeepSet', 'PointNetTutorial', 'ScalarInputMLP', 'MultiInputModel',
        'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel', 'DistanceMatrixRaggedModel',
    ]
    results = {}
    for mname in tqdm(model_names, desc="Training all models"):
        try:
            print(f'=== Training {mname} ===')
            m = build_model_by_name(mname, n=dim).to(device)
            train_data_model = prepare_data_for_model(mname, data_train_torch)
            test_data_model = prepare_data_for_model(mname, data_test_torch)
            optimizer = optimizer_class(m.parameters(), lr=optim_lr)
            criterion_loc = nn.MSELoss()
            for epoch in tqdm(range(min(5, num_epochs)), desc=f"Epochs for {mname}"):
                train_loss = train_epoch(m, train_data_model, np.hstack(PVs_train), optimizer, criterion_loc, mname)
                val_loss = evaluate(m, test_data_model, np.hstack(PVs_test), criterion_loc, mname)
                print(f'[{mname}] epoch {epoch+1}/{min(5,num_epochs)} train={train_loss:.4f} val={val_loss:.4f}')
            results[mname] = {'train_loss': train_loss, 'val_loss': val_loss}
            checkpoint = {
                'model_state_dict': m.state_dict(),
                'model_type': mname,
                'output_dim': output_dim,
                'homdim': homdim,
            }
            torch.save(checkpoint, f'models/{mname}.pth')
        except Exception as e:
            print(f'ERROR {mname}:', e)
            results[mname] = {'error': str(e)}
    print('=== ALL MODELS RESULTS ===')
    for k, v in results.items():
        print(k, v)


if model_name == 'all':
    train_all_models()
    sys.exit(0)

if model_name in MODEL_NAMES:
    m = build_model_by_name(model_name, n=dim).to(device)
    train_data_model = prepare_data_for_model(model_name, data_train_torch)
    test_data_model = prepare_data_for_model(model_name, data_test_torch)
    optimizer = optimizer_class(m.parameters(), lr=optim_lr)
    criterion_loc = nn.MSELoss()

    for epoch in tqdm(range(min(5, num_epochs)), desc=f"Training {model_name}"):
        train_loss = train_epoch(m, train_data_model, np.hstack(PVs_train), optimizer, criterion_loc, model_name)
        val_loss = evaluate(m, test_data_model, np.hstack(PVs_test), criterion_loc, model_name)
        print(f'[{model_name}] epoch {epoch+1}/{min(5,num_epochs)} train={train_loss:.4f} val={val_loss:.4f}')

    torch.save({
        'model_state_dict': m.state_dict(),
        'model_type': model_name,
        'output_dim': output_dim,
        'homdim': homdim,
    }, f'models/{model_name}.pth')
    print(f'Saved model {model_name} at models/{model_name}.pth')
    sys.exit(0)

raise SystemExit("Finished model training/comparison branch in train_nn.py; legacy section skipped")
