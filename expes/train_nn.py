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

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

dataset_name = sys.argv[1]
model_name   = sys.argv[2]
normalize    = int(sys.argv[3])
num_epochs   = int(sys.argv[4])
PV_type      = sys.argv[5]
mode         = sys.argv[6]
batch_size   = int(sys.argv[7]) if len(sys.argv) > 7 else 32

print(sys.argv)
print(f'Batch size: {batch_size}')

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
    optim_lr        = 5e-3
    optimizer_class = optim.Adamax
else:
    optim_lr        = 5e-3
    optimizer_class = optim.Adam

PVs_train_torch = [torch.FloatTensor(PVs_train[hidx]).to(device) for hidx in range(len(homdim))]
PVs_test_torch  = [torch.FloatTensor(PVs_test[hidx]).to(device)  for hidx in range(len(homdim))]

# -------------------------------------------------------------------------
# Gaussian smoothing  (Fix 4: batched vectorised kernel)
# -------------------------------------------------------------------------

GS_SIGMA = 0.01

def gaussian_smooth_batch(data_list: list, sigma: float = GS_SIGMA) -> list:
    """Vectorised GS: single (B,N,d) tensor when shapes are uniform."""
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
# Fix 1: Geometry precomputation for TFN models
# -------------------------------------------------------------------------

def precompute_geometry(model, data_list, mname):
    """
    Precompute k-NN geometry for all point clouds.

    Returns either a stacked 'uniform' geometry dict (B, N, k, *) when all
    point clouds share the same size, or a list of per-sample tuples.
    """
    if mname not in TFN_MODELS:
        return None
    try:
        from gt_tfn_layer import knn_geometry
        inner    = getattr(model, '_inner', model)
        rbf_enc  = inner.rbf
        gt_basis = inner.gt_basis
        k        = inner.k_neighbors
    except AttributeError:
        return None

    rbfs, gt_edges, nbr_idxs = [], [], []
    with torch.no_grad():
        for pc in tqdm(data_list, desc="Precomputing geometry", leave=False):
            rbf, gt_edge, nbr_idx = knn_geometry(pc, rbf_enc, gt_basis, k)
            rbfs.append(rbf.detach())
            gt_edges.append(gt_edge.detach())
            nbr_idxs.append(nbr_idx.detach())

    uniform = all(r.shape == rbfs[0].shape for r in rbfs)
    if uniform:
        return {
            'uniform': True,
            'rbf': torch.stack(rbfs),
            'gt_edge': torch.stack(gt_edges),
            'nbr_idx': torch.stack(nbr_idxs),
        }
    return {
        'uniform': False,
        'list': list(zip(rbfs, gt_edges, nbr_idxs)),
    }


# -------------------------------------------------------------------------
# TFN acceleration: batched forward pass over entire dataset
# -------------------------------------------------------------------------

def tfn_batched_forward(model, data_list, geom_cache, mname, batch_size=64):
    """
    Run TFN inference over data_list in mini-batches, eliminating the Python
    loop overhead for forward passes.  Each mini-batch calls the fastest
    available geometry-aware TFN path.

    Returns np.ndarray of shape (N, output_dim).
    """
    inner = getattr(model, '_inner', model)
    results = []
    for start in range(0, len(data_list), batch_size):
        batch    = data_list[start:start + batch_size]
        batch_idx = range(start, min(start + batch_size, len(data_list)))
        geom_b   = _get_geom_batch(geom_cache, batch_idx) if geom_cache else None
        with torch.inference_mode():
            if geom_b is not None:
                if isinstance(geom_b, dict) and geom_b.get('uniform', False) and hasattr(inner, '_encode_batch'):
                    batch_tensor = torch.stack(batch)
                    out = inner._encode_batch(
                        batch_tensor,
                        precomputed_geom=(geom_b['rbf'], geom_b['gt_edge'], geom_b['nbr_idx']))
                else:
                    descs = []
                    if isinstance(geom_b, dict) and geom_b.get('uniform', False):
                        for i, pc in enumerate(batch):
                            desc = inner._encode_single(
                                pc,
                                precomputed_geom=(
                                    geom_b['rbf'][i],
                                    geom_b['gt_edge'][i],
                                    geom_b['nbr_idx'][i],
                                ),
                            )
                            descs.append(desc.detach())
                    else:
                        for pc, geom in zip(batch, geom_b):
                            desc = inner._encode_single(
                                pc, precomputed_geom=geom)
                            descs.append(desc.detach())
                    descs_t = torch.stack(descs)
                    out = inner.rho(descs_t)
            else:
                out = model(batch)
        results.append(out.detach().cpu().numpy())
    return np.vstack(results)


# -------------------------------------------------------------------------
# Data preparation helpers
# -------------------------------------------------------------------------

def _pad_to_3d(data_list):
    out = []
    for x in data_list:
        arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
        if arr.shape[1] == 2:
            arr = np.concatenate([arr, np.zeros((len(arr), 1), dtype=arr.dtype)], axis=1)
        out.append(torch.FloatTensor(arr).to(device))
    return out


def augment_point_cloud(pc, sigma=0.01, clip=0.05):
    if pc.ndim != 2 or pc.shape[1] not in (2, 3):
        return pc
    if pc.shape[1] == 2:
        theta = float(torch.rand(1, device=pc.device) * 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor([[c, -s], [s, c]], device=pc.device, dtype=pc.dtype)
        pc = pc @ R
    else:
        R = torch.randn(3, 3, device=pc.device, dtype=pc.dtype)
        q, _ = torch.linalg.qr(R)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        pc = pc @ q
    noise = torch.randn_like(pc) * sigma
    noise = noise.clamp(-clip, clip)
    return pc + noise


def prepare_data_for_model(mname, data_list, use_gs=False,
                           gs_sigma=GS_SIGMA, augment=False):
    if augment:
        data_list = [augment_point_cloud(x) for x in data_list]
    if use_gs:
        data_list = gaussian_smooth_batch(data_list, sigma=gs_sigma)

    if mname in TFN_MODELS:
        return _pad_to_3d(data_list)

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
            out.append(torch.FloatTensor([[mat.mean()]]).to(device))
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
        return TensorFieldNetwork(
            num_classes=output_dim,
            hidden_channels=64,
            num_layers=6,
            num_rbf=64,
            cutoff=2.0,
            k_neighbors=16,
            classifier_dims=[256, 128],
        )
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(
            n=_n,
            num_classes=output_dim,
            hidden_channels=64,
            num_layers=6,
            num_rbf=64,
            cutoff=2.0,
            k_neighbors=16,
            classifier_dims=[256, 128],
            radial_hidden=128,
        )
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(
            n=_n,
            num_classes=output_dim,
            hidden_channels=64,
            num_layers=6,
            num_rbf=64,
            cutoff=2.0,
            k_neighbors=16,
            classifier_dims=[256, 128],
            radial_hidden=128,
        )
    if name == 'HierarchicalGTTFN':
        return HierarchicalGTTFN(
            n=_n,
            num_classes=output_dim,
            hidden_channels=64,
            stage_sizes=[256, 64],
            num_rbf=64,
            cutoff=2.0,
            classifier_dims=[256, 128],
        )
    if name == 'HierarchicalTensorFieldNetwork':
        return HierarchicalTensorFieldNetwork(
            num_classes=output_dim,
            hidden_channels=64,
            stage_sizes=[256, 64],
            num_rbf=64,
            cutoff=2.0,
            classifier_dims=[256, 128],
        )
    if name == 'OnEquivariantTensorFieldNetwork':
        return OnEquivariantTensorFieldNetwork(
            num_classes=output_dim,
            max_order=1,
            hidden_channels=64,
            num_layers=6,
            num_rbf=64,
            cutoff=2.0,
            k_neighbors=16,
            classifier_dims=[256, 128],
        )
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

def forward_single(model, x, mname, geom=None):
    if mname == 'MultiInputModel':
        pc, scalar = x
        return model([pc], scalar)
    if mname == 'ScalarInputMLP':
        return model(x)
    if geom is not None and mname in TFN_MODELS:
        rbf, gt_edge, nbr_idx = geom
        inner = getattr(model, '_inner', model)
        desc  = inner._encode_single(x, precomputed_geom=(rbf, gt_edge, nbr_idx))
        return inner.rho(desc.unsqueeze(0))
    if mname in [
        'TensorFieldNetwork', 'GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
        'HierarchicalGTTFN', 'PointNet3D', 'PointNetTutorial',
        'DistanceMatrixRaggedModel', 'ScalarDistanceDeepSet',
        'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel',
    ]:
        return model([x])
    return model(x.unsqueeze(0))


def _get_geom_batch(geom_cache, batch_idx):
    if geom_cache is None:
        return None
    if isinstance(batch_idx, range):
        batch_idx = list(batch_idx)
    if isinstance(geom_cache, dict) and geom_cache.get('uniform', False):
        return {
            'uniform': True,
            'rbf': geom_cache['rbf'][batch_idx],
            'gt_edge': geom_cache['gt_edge'][batch_idx],
            'nbr_idx': geom_cache['nbr_idx'][batch_idx],
        }
    if isinstance(geom_cache, dict):
        return [geom_cache['list'][i] for i in batch_idx]
    return [geom_cache[i] for i in batch_idx]


def forward_batch(model, batch_data, mname, geom_batch=None):
    """Forward a mini-batch of samples through the model."""
    if len(batch_data) == 0:
        return torch.empty(0, output_dim, device=device)

    if mname == 'MultiInputModel':
        pcs = [x[0] for x in batch_data]
        scalars = torch.cat([x[1].reshape(1, -1) for x in batch_data], dim=0)
        return model(pcs, scalars)

    if mname == 'ScalarInputMLP':
        if isinstance(batch_data, list):
            batch_data = torch.cat([x.reshape(1, -1) for x in batch_data], dim=0)
        return model(batch_data)

    if geom_batch is not None and mname in TFN_MODELS:
        inner = getattr(model, '_inner', model)
        if isinstance(geom_batch, dict) and geom_batch.get('uniform', False) and hasattr(inner, '_encode_batch'):
            batch_tensor = torch.stack(batch_data)
            return inner._encode_batch(
                batch_tensor,
                precomputed_geom=(geom_batch['rbf'], geom_batch['gt_edge'], geom_batch['nbr_idx']))

        if isinstance(geom_batch, dict) and geom_batch.get('uniform', False):
            descs = []
            for i, pc in enumerate(batch_data):
                desc = inner._encode_single(
                    pc,
                    precomputed_geom=(
                        geom_batch['rbf'][i],
                        geom_batch['gt_edge'][i],
                        geom_batch['nbr_idx'][i],
                    ),
                )
                descs.append(desc)
            descs_t = torch.stack(descs)
            return inner.rho(descs_t)

        descs = [inner._encode_single(x, precomputed_geom=geom)
                 for x, geom in zip(batch_data, geom_batch)]
        descs_t = torch.stack(descs)
        return inner.rho(descs_t)

    return model(batch_data)


# -------------------------------------------------------------------------
# Early stopping helper
# -------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when validation loss has not improved for `patience` epochs.

    Parameters
    ----------
    patience  : epochs to wait before stopping after last improvement
    min_delta : minimum improvement to count as an improvement
    restore   : if True, restore the best model weights on stop
    """
    def __init__(self, patience: int = 25, min_delta: float = 1e-5,
                 restore: bool = True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.restore    = restore
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None
        self.stopped_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            if self.restore:
                # Save a deep copy of the unwrapped (non-compiled) state dict
                base = model._orig_mod if hasattr(model, '_orig_mod') else model
                import copy
                self.best_state = copy.deepcopy(base.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                if self.restore and self.best_state is not None:
                    base = model._orig_mod if hasattr(model, '_orig_mod') else model
                    base.load_state_dict(self.best_state)
                    print(f'  Early stopping at epoch {epoch+1} '
                          f'(best val={self.best_loss:.5f}, '
                          f'restored from epoch {epoch+1-self.counter})')
                return True
        return False

    @property
    def best_epoch(self) -> int:
        return self.stopped_epoch - self.counter + 1


# -------------------------------------------------------------------------
# Train / evaluate loops
# -------------------------------------------------------------------------

def get_target(targets, i):
    return torch.FloatTensor(targets[i:i + 1]).to(device)


def train_epoch(model, data, targets, optimizer, criterion, mname,
                geom_cache=None, batch_size=32, scaler=None,
                use_amp=False, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    n = len(data)
    perm = np.random.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = perm[start:start + batch_size]
        batch_data = [data[i] for i in batch_idx]
        batch_targets = targets[batch_idx]
        batch_geom = _get_geom_batch(geom_cache, batch_idx)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = forward_batch(model, batch_data, mname,
                                   geom_batch=batch_geom)
            loss = criterion(output, batch_targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * len(batch_data)
    return total_loss / n


def evaluate(model, data, targets, criterion, mname,
             geom_cache=None, batch_size=32):
    model.eval()
    total_loss = 0.0
    n = len(data)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_idx = range(start, min(start + batch_size, n))
            batch_data = [data[i] for i in batch_idx]
            batch_targets = targets[start:start + len(batch_data)]
            batch_geom = _get_geom_batch(geom_cache, batch_idx)
            output = forward_batch(model, batch_data, mname,
                                   geom_batch=batch_geom)
            loss = criterion(output, batch_targets)
            total_loss += loss.item() * len(batch_data)
    return total_loss / n


# -------------------------------------------------------------------------
# Train one model variant (raw or GS)
# -------------------------------------------------------------------------

# Early stopping defaults — can be tightened per model type
ES_PATIENCE   = 50    # epochs without improvement before stopping
ES_MIN_DELTA  = 5e-6  # minimum improvement threshold

def train_single_model(mname, use_gs=False, gs_sigma=GS_SIGMA):
    tag   = '_GS' if use_gs else ''
    label = f'{mname}{tag}'
    print(f'\n=== Training {label} (gs={use_gs}) ===')

    m = deep_to(build_model_by_name(mname, n=dim), device)

    train_data = prepare_data_for_model(
        mname, data_train_torch, use_gs=use_gs,
        gs_sigma=gs_sigma, augment=True)
    test_data  = prepare_data_for_model(
        mname, data_test_torch, use_gs=use_gs,
        gs_sigma=gs_sigma, augment=False)

    # Warm-up: trigger lazy param init before optimizer
    m.train()
    with torch.no_grad():
        forward_single(m, train_data[0], mname)
    m = deep_to(m, device)

    # Fix 1: precompute geometry for TFN models
    geom_train = precompute_geometry(m, train_data, mname)
    geom_test  = precompute_geometry(m, test_data,  mname)

    # Fix 3: torch.compile
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            m = torch.compile(m, mode='reduce-overhead')
            print(f'  torch.compile enabled')
        except Exception as e:
            print(f'  torch.compile skipped: {e}')

    optimizer     = optimizer_class(
        m.parameters(), lr=optim_lr, weight_decay=1e-5)
    # LR scheduler: reduce on plateau for stable convergence
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion     = nn.MSELoss()
    targets_train = torch.FloatTensor(np.hstack(PVs_train)).to(device)
    targets_test  = torch.FloatTensor(np.hstack(PVs_test)).to(device)
    try:
        scaler = torch.amp.GradScaler(device_type='cuda',
                                     enabled=(device.type == 'cuda'))
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    early_stop = EarlyStopping(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA,
                                restore=True)

    log_every = max(1, num_epochs // 100)
    tr_loss = val_loss = 0.0

    use_amp = device.type == 'cuda'
    for epoch in tqdm(range(num_epochs), desc=f"Epochs ({label})"):
        tr_loss  = train_epoch(
            m, train_data, targets_train, optimizer, criterion, mname,
            geom_cache=geom_train, batch_size=batch_size,
            scaler=scaler, use_amp=use_amp, max_grad_norm=1.0)
        val_loss = evaluate(
            m, test_data, targets_test, criterion, mname,
            geom_cache=geom_test, batch_size=batch_size)

        scheduler.step(val_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  [{label}] epoch {epoch+1}/{num_epochs}'
                  f'  train={tr_loss:.4f}  val={val_loss:.4f}'
                  f'  lr={lr_now:.2e}')

        if early_stop.step(val_loss, m, epoch):
            break

    print(f'  Training finished at epoch {epoch+1}  '
          f'best_val={early_stop.best_loss:.5f}')

    extra = {}
    if mname == 'DistanceMatrixRaggedModel':
        base_m = m._orig_mod if hasattr(m, '_orig_mod') else m
        extra['num_points'] = base_m._phi_inp_dim

    ckpt_path  = f'models/{label}.pth'
    base_m     = m._orig_mod if hasattr(m, '_orig_mod') else m
    torch.save({
        'model_state_dict': base_m.state_dict(),
        'model_type':       mname,
        'output_dim':       output_dim,
        'homdim':           homdim,
        'dim':              dim,
        'use_gs':           use_gs,
        'gs_sigma':         gs_sigma,
        'activation':       'gelu',
        'norm':             'bn',
        **extra,
    }, ckpt_path)
    print(f'  Saved → {ckpt_path}')
    return {'train_loss': tr_loss, 'val_loss': val_loss,
            'best_val': early_stop.best_loss,
            'stopped_epoch': epoch + 1}


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def run_both(mname):
    res = {}
    for use_gs in [False, True]:
        tag = '_GS' if use_gs else ''
        try:
            res[mname + tag] = train_single_model(mname, use_gs=use_gs)
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
