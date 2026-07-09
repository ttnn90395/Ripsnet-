"""
Experiment 2: Test-time point cloud density ablation.

For y in [10,20,30,50,70,100]%, randomly keep y% of points in each test
point cloud and evaluate using existing checkpoints.

Usage:
    python density_ablation.py <dataset> <model_label> <fraction_pct> <trial> <identifier>

model_label: e.g. 'OnEquivariantTensorFieldNetwork' or 'OnEquivariantTensorFieldNetwork_GS'
"""
import os, sys, numpy as np
import dill as pck
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models import (
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    HierarchicalGTTFN, HierarchicalTensorFieldNetwork,
    OnEquivariantTensorFieldNetwork, PointNet3D,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DenseRagged, PermopRagged, RaggedPersistenceModel, DistanceMatrixRaggedModel,
    AttentionTensorFieldNetwork, StochasticTensorFieldNetwork,
    CrossAttentionTensorFieldNetwork,
    RelaxedOnEquivariantTensorFieldNetwork,
    HybridOnEquivariantTensorFieldNetwork,
    _move_basis_tensors,
)

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name    = sys.argv[1]
model_label     = sys.argv[2]
fraction_pct    = int(sys.argv[3])
trial           = int(sys.argv[4])
identifier      = sys.argv[5] if len(sys.argv) > 5 else 'run1'
fraction        = fraction_pct / 100.0

print(f"density_ablation: {dataset_name} {model_label} {fraction_pct}% trial={trial}")

TFN_MODELS = {'TensorFieldNetwork','GTTensorFieldNetwork','GTTensorFieldNetworkV2',
    'HierarchicalGTTFN','HierarchicalTensorFieldNetwork',
    'OnEquivariantTensorFieldNetwork','AttentionTensorFieldNetwork',
    'StochasticTensorFieldNetwork','CrossAttentionTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork','HybridOnEquivariantTensorFieldNetwork'}

# ─── Load data ────────────────────────────────────────────────────────────────
train_sfx = f"_train_TDE311LS_5{identifier}"
test_sfx  = f"_test_TDE311LS_clean_3{identifier}"

train_data = pck.load(open(f"datasets/{dataset_name}{train_sfx}.pkl", 'rb'))
test_data  = pck.load(open(f"datasets/{dataset_name}{test_sfx}.pkl",  'rb'))

data_train     = train_data["data_train"]
PVs_train      = train_data["PV_train"]
label_train    = train_data["label_train"]
homdim         = train_data["hdims"]
dim            = data_train[0].shape[1]
output_dim     = sum(PVs_train[h].shape[1] for h in range(len(homdim)))

data_test      = test_data["data_test"]
label_test     = test_data["label_test"]
N_test         = len(data_test)
N_train        = len(data_train)

print(f"  train={N_train} test={N_test} dim={dim} out={output_dim}")

le = LabelEncoder().fit(np.concatenate([label_train, label_test]))
y_train, y_test = le.transform(label_train), le.transform(label_test)

# ─── Load checkpoint ──────────────────────────────────────────────────────────
def find_checkpoint(mname):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_tag = os.environ.get('TFN_MODEL_TAG', '')
    dirs = [f'models/{model_tag}' if model_tag else 'models', 'models']
    for d in dirs:
        if not d: continue
        for ext in ('.pth', '.pt'):
            p = os.path.join(d, mname + ext)
            if os.path.isfile(p): return p
    for d in [os.path.join(script_dir, 'models'), os.path.join(script_dir, '..', 'models')]:
        for ext in ('.pth', '.pt'):
            p = os.path.join(d, mname + ext)
            if os.path.isfile(p): return p
    return None

ckpt_path = find_checkpoint(model_label)
if ckpt_path is None:
    print(f"  No checkpoint for '{model_label}', training from scratch...")
    from sklearn.preprocessing import LabelEncoder
    _npts = data_train[0].shape[0]
    def _build_model(name, out_dim, n_dim):
        if name == 'OnEquivariantTensorFieldNetwork':
            return OnEquivariantTensorFieldNetwork(num_classes=out_dim, max_order=1, hidden_channels=32, num_layers=3, num_rbf=64, classifier_dims=[64,32])
        if name == 'AttentionTensorFieldNetwork':
            return AttentionTensorFieldNetwork(num_classes=out_dim, max_order=1, hidden_channels=32, num_layers=3, num_heads=4, num_rbf=64, classifier_dims=[64,32], radial_hidden=64)
        if name == 'TensorFieldNetwork':
            return TensorFieldNetwork(num_classes=out_dim, max_order=0, hidden_channels=8, num_layers=2, classifier_dims=[16], num_rbf=64, k_neighbors=8)
        if name == 'GTTensorFieldNetwork':
            return GTTensorFieldNetwork(n=n_dim, num_classes=out_dim, max_order=0, hidden_channels=8, num_layers=2, num_rbf=64, classifier_dims=[16], radial_hidden=128)
        return OnEquivariantTensorFieldNetwork(num_classes=out_dim, max_order=1, hidden_channels=32, num_layers=3, num_rbf=64, classifier_dims=[64,32])
    model = _build_model(model_label, output_dim, dim).to(device)
    from torch import nn
    targets_t_nn = torch.FloatTensor(np.concatenate(PVs_train, axis=1)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    _tfn = model_label in TFN_MODELS
    def _prepare_pts(data_list):
        if _tfn:
            return [torch.cat([x, x.new_zeros(x.shape[0],1)], dim=1) if x.shape[1]==2 else x for x in data_list]
        return data_list
    train_in_nn = _prepare_pts(data_train_t)
    bs = min(32, len(train_in_nn))
    for ep in range(5):
        model.train()
        perm = np.random.permutation(len(train_in_nn))
        ls = 0.0
        for s in range(0, len(train_in_nn), bs):
            bix = perm[s:s+bs]
            bd = [train_in_nn[i] for i in bix]
            optim.zero_grad(set_to_none=True)
            out = model(bd)
            loss = criterion(out, targets_t_nn[bix])
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                ls += loss.item() * len(bd)
        if (ep+1) % 500 == 0:
            print(f"    ep {ep+1}/5  loss={ls/len(train_in_nn):.6f}")
    model.eval()
    model_state = model.state_dict()
    ckpt_out_dim = output_dim
    base_name = model_label.replace('_GS', '')
    ckpt_npts = data_train[0].shape[0]
    ckpt_activation = 'gelu'
    ckpt_norm = 'bn'
    ckpt = {'model_state_dict': model_state, 'output_dim': output_dim, 'num_points': ckpt_npts, 'activation': 'gelu', 'norm': 'bn'}
    os.makedirs('models', exist_ok=True)
    torch.save(ckpt, f'models/{model_label}.pth')
    print(f"  Saved checkpoint to models/{model_label}.pth")
else:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    ckpt_out_dim = ckpt.get('output_dim', output_dim) if isinstance(ckpt, dict) else output_dim
    base_name = model_label.replace('_GS', '')
    ckpt_npts = ckpt.get('num_points', None) if isinstance(ckpt, dict) else None
    ckpt_activation = ckpt.get('activation', None) if isinstance(ckpt, dict) else None
    ckpt_norm = ckpt.get('norm', None) if isinstance(ckpt, dict) else None

_has_bn = any('running_mean' in k for k in model_state)
if ckpt_activation is None: ckpt_activation = 'gelu' if _has_bn else 'relu'
if ckpt_norm is None: ckpt_norm = 'bn' if _has_bn else 'none'

def build_analysis_model(name, out_dim, extra=None):
    extra = extra or {}
    hp = lambda k, d: extra.get(k, d)
    n_dim = dim
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=out_dim, max_order=hp('max_order',0),
            hidden_channels=hp('hidden_channels',8), num_layers=hp('num_layers',2),
            num_rbf=hp('num_rbf',64), cutoff=1.0, k_neighbors=hp('k_neighbors',8),
            classifier_dims=hp('classifier_dims',[16]))
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=n_dim, num_classes=out_dim,
            max_order=hp('max_order',0), hidden_channels=hp('hidden_channels',8),
            num_layers=hp('num_layers',2), num_rbf=64, cutoff=1.0,
            k_neighbors=hp('k_neighbors',8), classifier_dims=[16], radial_hidden=128)
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=n_dim, num_classes=out_dim,
            max_order=hp('max_order',0), hidden_channels=hp('hidden_channels',8),
            num_layers=hp('num_layers',2), num_rbf=64, cutoff=1.0,
            k_neighbors=hp('k_neighbors',8), classifier_dims=[16], radial_hidden=128)
    if name == 'OnEquivariantTensorFieldNetwork':
        return OnEquivariantTensorFieldNetwork(num_classes=out_dim,
            max_order=hp('max_order',1), hidden_channels=hp('hidden_channels',32),
            num_layers=hp('num_layers',3), num_rbf=64, cutoff=1.0,
            k_neighbors=hp('k_neighbors',16), classifier_dims=[64,32])
    if name == 'PointNet3D':
        return PointNet3D(output_dim=out_dim, activation=ckpt_activation, norm=ckpt_norm)
    if name == 'PointNetTutorial':
        return PointNetTutorial(output_dim=out_dim, activation=ckpt_activation, norm=ckpt_norm)
    if name == 'DistanceMatrixRaggedModel':
        npts = ckpt_npts or dim
        return DistanceMatrixRaggedModel(output_dim=out_dim, num_points=npts,
            activation=ckpt_activation, norm=ckpt_norm)
    if name == 'ScalarDistanceDeepSet':
        return ScalarDistanceDeepSet(output_dim=out_dim, activation=ckpt_activation, norm=ckpt_norm)
    if name == 'ScalarInputMLP':
        return ScalarInputMLP(output_dim=out_dim, activation=ckpt_activation, norm=ckpt_norm)
    if name == 'MultiInputModel':
        return MultiInputModel(target_output_dim=out_dim, scalar_input_dim=1,
            activation=ckpt_activation, norm=ckpt_norm)
    if name == 'AttentionTensorFieldNetwork':
        return AttentionTensorFieldNetwork(num_classes=out_dim,
            max_order=hp('max_order',1), hidden_channels=32, num_layers=3,
            num_heads=4, num_rbf=64, cutoff=1.0, k_neighbors=16,
            classifier_dims=[64,32], radial_hidden=64)
    if name == 'CrossAttentionTensorFieldNetwork':
        return CrossAttentionTensorFieldNetwork(num_classes=out_dim, n=n_dim,
            max_order=1, hidden_channels=8, num_layers=2, num_heads=4,
            transformer_layers=2, num_rbf=64, cutoff=1.0, k_neighbors=8,
            classifier_dims=[16], radial_hidden=64, dropout=0.1)
    if name == 'StochasticTensorFieldNetwork':
        return StochasticTensorFieldNetwork(num_classes=out_dim,
            num_mixtures=hp('num_mixtures',3), max_order=0, hidden_channels=8,
            num_layers=2, num_rbf=64, cutoff=1.0, k_neighbors=8,
            encoder_dims=[64,32])
    if name == 'RelaxedOnEquivariantTensorFieldNetwork':
        return RelaxedOnEquivariantTensorFieldNetwork(num_classes=out_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            cutoff=1.0, k_neighbors=16, classifier_dims=[64,32])
    if name == 'HybridOnEquivariantTensorFieldNetwork':
        return HybridOnEquivariantTensorFieldNetwork(num_classes=out_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            cutoff=1.0, k_neighbors=16, classifier_dims=[64,32], non_eq_dim=128)
    # Check for saved arch metadata
    for k in ['hidden_channels','num_layers','num_rbf','classifier_dims','k_neighbors']:
        v = None
        if isinstance(ckpt, dict):
            v = ckpt.get(k, ckpt.get(f'hp_{k}', None))
        if v: extra[k] = v
    # Try inference
    def _infer_arch(state):
        keys = set(state.keys()); info = {}
        prefix = ''
        for c in ('_inner.base._inner.','_inner.',''):
            if any(k.startswith(c) for k in keys): prefix = c; break
        rk = f'{prefix}rbf.centers'
        if rk in state: info['num_rbf'] = state[rk].shape[0]
        gk = next((k for k in keys if 'gate.gates.' in k and k.endswith('.weight')), None)
        if gk: info['hidden_channels'] = state[gk].shape[0]
        mp = f'{prefix}mp_layers.'
        lids = []
        for k in keys:
            if k.startswith(mp):
                r = k[len(mp):].split('.')[0]
                if r.isdecimal(): lids.append(int(r))
        if lids: info['num_layers'] = max(lids)+1
        rho = f'{prefix}rho.'
        lks = sorted([k for k in keys if k.startswith(rho) and k.endswith('.weight')
                      and state[k].ndim == 2],
                     key=lambda k: int(k[len(rho):].split('.')[0]))
        if lks: info['classifier_dims'] = [state[k].shape[0] for k in lks[:-1]]
        return info
    arch = _infer_arch(model_state)
    extra = {**arch, **extra}
    if name == 'AttentionTensorFieldNetwork':
        return AttentionTensorFieldNetwork(num_classes=out_dim,
            max_order=extra.get('max_order',1), hidden_channels=extra.get('hidden_channels',32),
            num_layers=extra.get('num_layers',3), num_heads=extra.get('num_heads',4),
            num_rbf=extra.get('num_rbf',64), cutoff=1.0, k_neighbors=extra.get('k_neighbors',16),
            classifier_dims=extra.get('classifier_dims',[64,32]),
            radial_hidden=extra.get('radial_hidden',64))
    raise ValueError(f"Unknown model: {name}")

# Load model
model_pv = build_analysis_model(base_name, ckpt_out_dim)
model_pv.load_state_dict(model_state)
model_pv = model_pv.to(device).eval()
print(f"  Loaded {model_label} from {ckpt_path}")

# Precompute train PVs (full, uncorrupted) for XGBoost training
data_train_t = [torch.FloatTensor(x).to(device) for x in data_train]
data_test_t  = [torch.FloatTensor(x).to(device)  for x in data_test]

def gaussian_smooth_batch(data_list, sigma=0.5):
    if not data_list: return data_list
    if all(x.shape == data_list[0].shape for x in data_list):
        X = torch.stack(data_list)
        d = X.unsqueeze(2) - X.unsqueeze(1)
        W = torch.exp(-(d**2).sum(-1) / (2*sigma**2))
        W = W / W.sum(-1, keepdim=True)
        return list((W @ X).unbind(0))
    out = []
    for x in data_list:
        d = x.unsqueeze(0) - x.unsqueeze(1)
        W = torch.exp(-(d**2).sum(-1) / (2*sigma**2))
        W = W / W.sum(-1, keepdim=True)
        out.append(W @ x)
    return out

use_gs = '_GS' in model_label
gs_sigma = ckpt.get('gs_sigma', 0.5) if isinstance(ckpt, dict) else 0.5

def prepare_for_model(mname, tensor_list):
    out = []
    for x in tensor_list:
        arr = x.cpu().numpy()
        if mname in TFN_MODELS:
            if arr.shape[1] == 2:
                arr = np.concatenate([arr, np.zeros((arr.shape[0],1))], axis=1)
            out.append(torch.FloatTensor(arr).to(device))
        elif mname == 'PointNet3D':
            if arr.shape[1] < 3:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], 3-arr.shape[1]))], axis=1)
            out.append(torch.FloatTensor(arr[:,:3]).to(device))
        elif mname == 'PointNetTutorial':
            out.append(torch.FloatTensor(arr[:,:2]).to(device))
        elif mname in ('ScalarDistanceDeepSet','DistanceMatrixRaggedModel'):
            m = np.linalg.norm(arr[:,None,:] - arr[None,:,:], axis=-1)
            out.append(torch.FloatTensor(m).to(device))
        elif mname == 'ScalarInputMLP':
            m = np.linalg.norm(arr[:,None,:] - arr[None,:,:], axis=-1)
            out.append(torch.FloatTensor([[m.mean()]]).to(device))
        elif mname == 'MultiInputModel':
            m = np.linalg.norm(arr[:,None,:] - arr[None,:,:], axis=-1)
            out.append((torch.FloatTensor(arr).to(device), torch.FloatTensor([[m.mean()]]).to(device)))
        else:
            out.append(x)
    return out

# Get train PVs (full density) for XGBoost fitting
if use_gs:
    train_pc = gaussian_smooth_batch(data_train_t, sigma=gs_sigma)
else:
    train_pc = data_train_t
train_prepared = prepare_for_model(base_name, train_pc)

# Precompute train geometry for TFN
def precompute_geom_batched(model, data_list):
    if base_name not in TFN_MODELS: return None
    try:
        from gt_tfn_layer import knn_geometry
        inner = getattr(model, '_inner', model)
        _move_basis_tensors(inner, device)
        k = inner.k_neighbors
        rbfs, gts, nbrs = [], [], []
        for pc in data_list:
            r,g,n = knn_geometry(pc, inner.rbf, inner.gt_basis, k)
            rbfs.append(r.detach()); gts.append(g.detach()); nbrs.append(n.detach())
        if all(r.shape == rbfs[0].shape for r in rbfs):
            return {'rbf':torch.stack(rbfs),'gt_edge':torch.stack(gts),
                    'nbr_idx':torch.stack(nbrs),'uniform':True}
        return {'list':list(zip(rbfs,gts,nbrs)),'uniform':False}
    except: return None

train_geom = precompute_geom_batched(model_pv, train_prepared)

def forward_batch(model, batch, mname, geom=None):
    if mname == 'MultiInputModel':
        return model([x[0] for x in batch],
                     torch.cat([x[1].reshape(1,-1) for x in batch]))
    if mname == 'ScalarInputMLP':
        return model(torch.cat([x.reshape(1,-1) for x in batch]))
    if geom is not None and mname in TFN_MODELS:
        inner = getattr(model, '_inner', model)
        _move_basis_tensors(inner, device)
        if isinstance(geom, dict) and geom.get('uniform',False) and hasattr(inner, '_encode_batch'):
            return inner._encode_batch(torch.stack(batch),
                precomputed_geom=(geom['rbf'],geom['gt_edge'],geom['nbr_idx']))
        gl = geom['list'] if isinstance(geom, dict) else geom
        descs = []
        for x,(r,g,n) in zip(batch, gl):
            r=r.squeeze(0) if r.ndim==4 else r; g=g.squeeze(0) if g.ndim==4 else g
            n=n.squeeze(0) if n.ndim==3 else n
            descs.append(inner._encode_single(x, precomputed_geom=(r,g,n)))
        return inner.rho(torch.stack(descs))
    return model(batch)

# Compute train PVs
train_preds = []
bs = 64
with torch.no_grad():
    for s in range(0, len(train_prepared), bs):
        bd = train_prepared[s:s+bs]
        bg = train_geom
        if isinstance(bg, dict) and bg.get('uniform',False):
            bg_sub = {k:(bg[k][s:s+bs] if k in ('rbf','gt_edge','nbr_idx') else bg[k]) for k in bg}
        elif bg:
            bg_sub = bg['list'][s:s+bs]
        else:
            bg_sub = None
        out = forward_batch(model_pv, bd, base_name, geom=bg_sub)
        train_preds.append(out.detach().cpu().numpy())
PV_train_nn = np.vstack(train_preds)

# Fit XGBoost on full-density train predictions (as in original analysis)
n_classes = len(np.unique(y_train))
if n_classes < 2:
    print(f"  WARNING: only {n_classes} class(es) in training set, XGBoost would fail. Recording NaN.")
    tr_acc, te_acc = float('nan'), float('nan')
    print(f"  XGB  train(full)=N/A  test({fraction_pct}% density)=N/A")
    # Save and exit early
    result = {
        'dataset': dataset_name, 'model_label': model_label,
        'fraction_pct': fraction_pct, 'trial': trial,
        'n_test': N_test, 'n_points_full': len(data_test[0]) if data_test else 0,
        'n_points_kept': 0,
        'xgb_train_acc': float('nan'), 'xgb_test_acc': float('nan'),
        'error': f'only {n_classes} class(es) in training set',
    }
    out_path = f"results/ablation_density_{dataset_name}_{model_label}_{fraction_pct}pct_t{trial}.json"
    json = __import__('json')
    json.dump(result, open(out_path, 'w'))
    print(f"  Saved to {out_path}")
    sys.exit(0)

clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)
clf.fit(PV_train_nn, y_train)
tr_acc = clf.score(PV_train_nn, y_train)

# ─── Subsampled test evaluation ────────────────────────────────────────────────
rng = np.random.RandomState(trial * 137 + 999)
subsampled_test_data_list = []
for pc in data_test:
    n = len(pc)
    n_keep = max(3, int(n * fraction))
    idx = rng.choice(n, size=n_keep, replace=False)
    idx.sort()
    subsampled_test_data_list.append(pc[idx])

sub_test_t = [torch.FloatTensor(pc).to(device) for pc in subsampled_test_data_list]
if use_gs:
    sub_test_t = gaussian_smooth_batch(sub_test_t, sigma=gs_sigma)
sub_prepared = prepare_for_model(base_name, sub_test_t)
sub_geom = precompute_geom_batched(model_pv, sub_prepared)

test_preds = []
with torch.no_grad():
    for s in range(0, len(sub_prepared), bs):
        bd = sub_prepared[s:s+bs]
        if sub_geom and isinstance(sub_geom, dict) and sub_geom.get('uniform',False):
            bg = {k:(sub_geom[k][s:s+bs] if k in ('rbf','gt_edge','nbr_idx') else sub_geom[k]) for k in sub_geom}
        elif sub_geom:
            bg = sub_geom['list'][s:s+bs]
        else:
            bg = None
        out = forward_batch(model_pv, bd, base_name, geom=bg)
        test_preds.append(out.detach().cpu().numpy())
PV_test_nn = np.vstack(test_preds)

te_acc = clf.score(PV_test_nn, y_test)
print(f"  XGB  train(full)={100*tr_acc:.2f}%  test({fraction_pct}% density)={100*te_acc:.2f}%")

# ─── Save ──────────────────────────────────────────────────────────────────────
result = {
    'dataset': dataset_name, 'model_label': model_label,
    'fraction_pct': fraction_pct, 'trial': trial,
    'n_test': N_test, 'n_points_full': len(data_test[0]),
    'n_points_kept': max(3, int(len(data_test[0]) * fraction)),
    'xgb_train_acc': tr_acc, 'xgb_test_acc': te_acc,
}
out_path = f"results/ablation_density_{dataset_name}_{model_label}_{fraction_pct}pct_t{trial}.json"
json = __import__('json')
json.dump(result, open(out_path, 'w'))
print(f"  Saved to {out_path}")
