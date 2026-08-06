"""
Experiment 3: Isometry robustness protocol (T3).

For each (dataset, model, n_augments, trial):
  1. Load trained checkpoint (or train from scratch if missing)
  2. Compute PV predictions on original test set
  3. Apply n_augments random 2D rotations + small translations to each test sample
  4. Compute PV predictions on each augmented version
  5. Measure mean/std L2 distance (isometry robustness score)
  6. Evaluate XGBoost classifier accuracy under isometry augmentation

Usage:
    python isometry_ablation.py <dataset> <model_label> <n_augments> <trial> [identifier]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, numpy as np
import dill as pck
import torch
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models import (
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    OnEquivariantTensorFieldNetwork, PointNet3D,
    ScalarDistanceDeepSet, PointNetTutorial, ScalarInputMLP, MultiInputModel,
    DistanceMatrixRaggedModel,
    AttentionTensorFieldNetwork, StochasticTensorFieldNetwork,
    CrossAttentionTensorFieldNetwork,
    RelaxedOnEquivariantTensorFieldNetwork,
    HybridOnEquivariantTensorFieldNetwork,
    GraphMambaTensorFieldNetwork,
    _move_basis_tensors,
)

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = sys.argv[1]
model_label  = sys.argv[2]
n_augments   = int(sys.argv[3])
trial        = int(sys.argv[4])
identifier   = sys.argv[5] if len(sys.argv) > 5 else 'run1'

print(f"isometry_ablation: {dataset_name} {model_label} n_aug={n_augments} trial={trial}")

TFN_MODELS = {
    'TensorFieldNetwork','GTTensorFieldNetwork','GTTensorFieldNetworkV2',
    'HierarchicalGTTFN','HierarchicalTensorFieldNetwork',
    'OnEquivariantTensorFieldNetwork','AttentionTensorFieldNetwork',
    'StochasticTensorFieldNetwork','CrossAttentionTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork','HybridOnEquivariantTensorFieldNetwork',
    'GraphMambaTensorFieldNetwork'}

# ─── Load data ────────────────────────────────────────────────────────────────
train_sfx = f"_train_TDE311LS_5{identifier}"
test_sfx  = f"_test_TDE311LS_clean_3{identifier}"

train_data = pck.load(open(f"datasets/{dataset_name}{train_sfx}.pkl", 'rb'))
test_data  = pck.load(open(f"datasets/{dataset_name}{test_sfx}.pkl",  'rb'))

data_train  = train_data["data_train"]
PVs_train   = train_data["PV_train"]
label_train = train_data["label_train"]
homdim      = train_data["hdims"]
dim         = data_train[0].shape[1]
output_dim  = sum(PVs_train[h].shape[1] for h in range(len(homdim)))

data_test  = test_data["data_test"]
label_test = test_data["label_test"]
N_test     = len(data_test)
N_train    = len(data_train)

print(f"  train={N_train} test={N_test} dim={dim} out={output_dim}")

le = LabelEncoder().fit(np.concatenate([label_train, label_test]))
y_train, y_test = le.transform(label_train), le.transform(label_test)

# ─── 2D Isometry augmentation (SO(2) rotation + translation) ─────────────
def augment_isometry_2d(pc, rng, trans_frac=0.08):
    """Apply a random 2D rotation (on first 2 coords) and small translation."""
    bbox = pc.max(axis=0) - pc.min(axis=0)
    t_max = trans_frac * np.linalg.norm(bbox)
    d = pc.shape[1]
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(d)
    R[0, 0] = c; R[0, 1] = -s
    R[1, 0] = s; R[1, 1] = c
    t = rng.uniform(-t_max, t_max, size=d)
    return pc @ R.T + t

# ─── Helpers (self-contained, no density_ablation import) ──────────────────
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

def prepare_for_model(mname, tensor_list):
    # Native-n models (built with n=dim) must get raw n-dim data; only
    # n=3-built TFN models (or dim>=3 data) get padded to 3D.
    native_n = mname in ('GTTensorFieldNetwork', 'GTTensorFieldNetworkV2',
                         'CrossAttentionTensorFieldNetwork')
    needs_3d = (not native_n) or dim >= 3
    out = []
    for x in tensor_list:
        arr = x.cpu().numpy()
        if mname in TFN_MODELS:
            if arr.shape[1] == 2 and needs_3d:
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

def find_checkpoint(mname):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_tag = os.environ.get('TFN_MODEL_TAG', '')
    for d in ([f'models/{model_tag}' if model_tag else 'models', 'models',
               os.path.join(script_dir, 'models'), os.path.join(script_dir, '..', 'models')]):
        if not d: continue
        for ext in ('.pth', '.pt'):
            p = os.path.join(d, mname + ext)
            if os.path.isfile(p): return p
    return None

# ─── Load checkpoint ────────────────────────────────────────────────────────
ckpt_path = find_checkpoint(model_label)
base_name = model_label.replace('_GS', '')
if ckpt_path is None:
    print(f"  ERROR: No checkpoint for '{model_label}'")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location=device)
model_state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
ckpt_out_dim = ckpt.get('output_dim', output_dim) if isinstance(ckpt, dict) else output_dim
ckpt_npts = ckpt.get('num_points', None) if isinstance(ckpt, dict) else None
ckpt_activation = ckpt.get('activation', None) if isinstance(ckpt, dict) else None
ckpt_norm = ckpt.get('norm', None) if isinstance(ckpt, dict) else None
print(f"  Loaded checkpoint from {ckpt_path}")

_has_bn = any('running_mean' in k for k in model_state)
if ckpt_activation is None: ckpt_activation = 'gelu' if _has_bn else 'relu'
if ckpt_norm is None: ckpt_norm = 'bn' if _has_bn else 'none'

# ─── Build model ────────────────────────────────────────────────────────────
def build_model(name, out_dim):
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=out_dim, max_order=0,
            hidden_channels=8, num_layers=2, num_rbf=64, cutoff=1.0,
            k_neighbors=8, classifier_dims=[16])
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=dim, num_classes=out_dim, max_order=0,
            hidden_channels=8, num_layers=2, num_rbf=64, cutoff=1.0,
            k_neighbors=8, classifier_dims=[16], radial_hidden=128)
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=dim, num_classes=out_dim, max_order=0,
            hidden_channels=8, num_layers=2, num_rbf=64, cutoff=1.0,
            k_neighbors=8, classifier_dims=[16], radial_hidden=128)
    if name == 'OnEquivariantTensorFieldNetwork':
        return OnEquivariantTensorFieldNetwork(num_classes=out_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            cutoff=1.0, k_neighbors=16, classifier_dims=[64,32])
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
            max_order=1, hidden_channels=32, num_layers=3, num_heads=4,
            num_rbf=64, cutoff=1.0, k_neighbors=16,
            classifier_dims=[64,32], radial_hidden=64)
    if name == 'CrossAttentionTensorFieldNetwork':
        return CrossAttentionTensorFieldNetwork(num_classes=out_dim, n=dim,
            max_order=1, hidden_channels=8, num_layers=2, num_heads=4,
            transformer_layers=2, num_rbf=64, cutoff=1.0, k_neighbors=8,
            classifier_dims=[16], radial_hidden=64, dropout=0.1)
    if name == 'StochasticTensorFieldNetwork':
        return StochasticTensorFieldNetwork(num_classes=out_dim,
            num_mixtures=3, max_order=0, hidden_channels=8, num_layers=2,
            num_rbf=64, cutoff=1.0, k_neighbors=8, encoder_dims=[64,32])
    if name == 'RelaxedOnEquivariantTensorFieldNetwork':
        return RelaxedOnEquivariantTensorFieldNetwork(num_classes=out_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            cutoff=1.0, k_neighbors=16, classifier_dims=[64,32])
    if name == 'HybridOnEquivariantTensorFieldNetwork':
        return HybridOnEquivariantTensorFieldNetwork(num_classes=out_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            cutoff=1.0, k_neighbors=16, classifier_dims=[64,32], non_eq_dim=128)
    if name == 'GraphMambaTensorFieldNetwork':
        return GraphMambaTensorFieldNetwork(num_classes=out_dim,
            max_order=1, hidden_channels=32, num_layers=4, num_rbf=64,
            cutoff=1.0, k_neighbors=16, classifier_dims=[64,32])
    raise ValueError(f"Unknown model: {name}")

model_pv = build_model(base_name, ckpt_out_dim)
model_pv.load_state_dict(model_state)
model_pv = model_pv.to(device).eval()

def precompute_geom(model, data_list):
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

def fwd(model, batch, geom=None):
    if base_name in TFN_MODELS and geom is not None:
        inner = getattr(model, '_inner', model)
        _move_basis_tensors(inner, device)
        if isinstance(geom, dict) and geom.get('uniform',False):
            if hasattr(inner, '_encode_batch'):
                return inner._encode_batch(torch.stack(batch),
                    precomputed_geom=(geom['rbf'],geom['gt_edge'],geom['nbr_idx']))
            gl = list(zip(geom['rbf'], geom['gt_edge'], geom['nbr_idx']))
        elif isinstance(geom, dict):
            gl = geom['list']
        else:
            gl = geom
        descs = []
        for x,(r,g,n) in zip(batch, gl):
            r=r.squeeze(0) if r.ndim==4 else r; g=g.squeeze(0) if g.ndim==4 else g
            n=n.squeeze(0) if n.ndim==3 else n
            descs.append(inner._encode_single(x, precomputed_geom=(r,g,n)))
        return inner.rho(torch.stack(descs))
    return model(batch)

# ─── Compute train PVs ────────────────────────────────────────────────────
use_gs = '_GS' in model_label
gs_sigma = ckpt.get('gs_sigma', 0.5) if isinstance(ckpt, dict) else 0.5

data_train_t = [torch.FloatTensor(x).to(device) for x in data_train]
data_test_t  = [torch.FloatTensor(x).to(device)  for x in data_test]

train_pc = gaussian_smooth_batch(data_train_t, sigma=gs_sigma) if use_gs else data_train_t
train_prepared = prepare_for_model(base_name, train_pc)
train_geom = precompute_geom(model_pv, train_prepared)

bs = 64
train_preds = []
with torch.no_grad():
    for s in range(0, len(train_prepared), bs):
        bd = train_prepared[s:s+bs]
        bg = None
        if train_geom and isinstance(train_geom, dict) and train_geom.get('uniform',False):
            bg = {k:(train_geom[k][s:s+bs] if k in ('rbf','gt_edge','nbr_idx') else train_geom[k]) for k in train_geom}
        elif train_geom:
            bg = train_geom['list'][s:s+bs] if isinstance(train_geom, dict) else train_geom[s:s+bs]
        out = fwd(model_pv, bd, geom=bg)
        train_preds.append(out.detach().cpu().numpy())
PV_train_nn = np.vstack(train_preds)

n_classes = len(np.unique(y_train))
if n_classes < 2:
    print(f"  WARNING: only {n_classes} class(es), skipping.")
    result = {'dataset': dataset_name, 'model_label': model_label,
              'n_augments': n_augments, 'trial': trial,
              'mean_l2_dist': float('nan'), 'std_l2_dist': float('nan'),
              'accuracy_clean': float('nan'), 'accuracy_augmented': float('nan')}
    out_path = f"results/isometry_{dataset_name}_{model_label}_aug{n_augments}_t{trial}.json"
    json.dump(result, open(out_path, 'w'))
    sys.exit(0)

clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0, n_jobs=1)
clf.fit(PV_train_nn, y_train)
train_acc = clf.score(PV_train_nn, y_train)

# ─── Compute clean test PVs ────────────────────────────────────────────────
test_pc = gaussian_smooth_batch(data_test_t, sigma=gs_sigma) if use_gs else data_test_t
test_prepared = prepare_for_model(base_name, test_pc)
test_geom = precompute_geom(model_pv, test_prepared)

test_preds = []
with torch.no_grad():
    for s in range(0, len(test_prepared), bs):
        bd = test_prepared[s:s+bs]
        bg = None
        if test_geom and isinstance(test_geom, dict) and test_geom.get('uniform', False):
            bg = {k:(test_geom[k][s:s+bs] if k in ('rbf','gt_edge','nbr_idx') else test_geom[k]) for k in test_geom}
        elif test_geom:
            bg = test_geom['list'][s:s+bs] if isinstance(test_geom, dict) else test_geom[s:s+bs]
        out = fwd(model_pv, bd, geom=bg)
        test_preds.append(out.detach().cpu().numpy())
PV_test_clean = np.vstack(test_preds)

clean_acc = clf.score(PV_test_clean, y_test)
print(f"  XGB train={100*train_acc:.2f}%  clean_test={100*clean_acc:.2f}%")

# ─── Isometry augmentation robustness ──────────────────────────────────────
rng = np.random.RandomState(trial * 271 + 42)

l2_dists = []
aug_preds_all = []

for aug_idx in range(n_augments):
    aug_test_t = []
    for pc in data_test:
        arr = pc.cpu().numpy() if isinstance(pc, torch.Tensor) else np.array(pc)
        aug_pc = augment_isometry_2d(arr, rng)
        aug_test_t.append(torch.FloatTensor(aug_pc).to(device))

    if use_gs:
        aug_test_t = gaussian_smooth_batch(aug_test_t, sigma=gs_sigma)
    aug_prepared = prepare_for_model(base_name, aug_test_t)
    aug_geom = precompute_geom(model_pv, aug_prepared)

    aug_preds = []
    with torch.no_grad():
        for s in range(0, len(aug_prepared), bs):
            bd = aug_prepared[s:s+bs]
            bg = None
            if aug_geom and isinstance(aug_geom, dict) and aug_geom.get('uniform', False):
                bg = {k:(aug_geom[k][s:s+bs] if k in ('rbf','gt_edge','nbr_idx') else aug_geom[k]) for k in aug_geom}
            elif aug_geom:
                bg = aug_geom['list'][s:s+bs] if isinstance(aug_geom, dict) else aug_geom[s:s+bs]
            out = fwd(model_pv, bd, geom=bg)
            aug_preds.append(out.detach().cpu().numpy())
    PV_test_aug = np.vstack(aug_preds)
    aug_preds_all.append(PV_test_aug)

    for i in range(N_test):
        dist = np.linalg.norm(PV_test_clean[i] - PV_test_aug[i])
        l2_dists.append(dist)

mean_l2 = float(np.mean(l2_dists)) if l2_dists else 0.0
std_l2  = float(np.std(l2_dists))  if l2_dists else 0.0

aug_accs = []
for PV_aug in aug_preds_all:
    aug_accs.append(clf.score(PV_aug, y_test))
mean_aug_acc = float(np.mean(aug_accs)) if aug_accs else float('nan')

print(f"  Isometry robustness: L2={mean_l2:.4f}±{std_l2:.4f}  aug_acc={100*mean_aug_acc:.2f}%")

result = {
    'dataset': dataset_name, 'model_label': model_label,
    'n_augments': n_augments, 'trial': trial,
    'n_test': N_test, 'dim': dim,
    'xgb_train_acc': train_acc,
    'accuracy_clean': clean_acc,
    'accuracy_augmented': mean_aug_acc,
    'mean_l2_dist': mean_l2, 'std_l2_dist': std_l2,
}
out_path = f"results/isometry_{dataset_name}_{model_label}_aug{n_augments}_t{trial}.json"
json.dump(result, open(out_path, 'w'))
print(f"  Saved to {out_path}")
