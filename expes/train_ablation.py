"""
Experiment 1: Training data fraction ablation.

Train models on x% of training data (x in [10,20,30,50,70,100]%),
multiple trials, average. Tests generalization capacity with limited data.

Usage:
    python train_ablation.py <dataset> <model> <fraction_pct> <trial> <epochs> <identifier>
"""
import os, sys, numpy as np
import dill as pck
import torch
import torch.nn as nn
import torch.optim as optim
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models import (_move_basis_tensors,
    PointNetTutorial, PointNet3D, DistanceMatrixRaggedModel, ScalarDistanceDeepSet,
    ScalarInputMLP, MultiInputModel,
    TensorFieldNetwork, GTTensorFieldNetwork, GTTensorFieldNetworkV2,
    HierarchicalGTTFN, HierarchicalTensorFieldNetwork,
    OnEquivariantTensorFieldNetwork, AttentionTensorFieldNetwork,
    StochasticTensorFieldNetwork, CrossAttentionTensorFieldNetwork,
    RelaxedOnEquivariantTensorFieldNetwork, HybridOnEquivariantTensorFieldNetwork)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

os.makedirs('results', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name  = sys.argv[1]
model_name    = sys.argv[2]
fraction_pct  = int(sys.argv[3])
trial         = int(sys.argv[4])
num_epochs    = int(sys.argv[5])
identifier    = sys.argv[6] if len(sys.argv) > 6 else 'run1'
fraction      = fraction_pct / 100.0

print(f"train_ablation: {dataset_name} {model_name} {fraction_pct}% trial={trial}")

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
N_full         = len(data_train)

data_test      = test_data["data_test"]
label_test     = test_data["label_test"]
N_test         = len(data_test)

# ─── Subsample ─────────────────────────────────────────────────────────────────
rng = np.random.RandomState(trial * 42 + 999)
n_use = max(1, int(N_full * fraction))
idx = rng.choice(N_full, size=n_use, replace=False)
idx.sort()
data_train = [data_train[i] for i in idx]
PVs_train  = [PVs_train[h][idx] for h in range(len(homdim))]
label_train = label_train[idx]
print(f"  {n_use}/{N_full} train samples ({fraction_pct}%)  test={N_test}  dim={dim}  out={output_dim}")

# ─── Torch tensors ────────────────────────────────────────────────────────────
data_train_t = [torch.FloatTensor(x).to(device) for x in data_train]
data_test_t  = [torch.FloatTensor(x).to(device)  for x in data_test]
targets_t    = torch.FloatTensor(np.concatenate(PVs_train, axis=1)).to(device)
le = LabelEncoder().fit(np.concatenate([label_train, label_test]))
y_train, y_test = le.transform(label_train), le.transform(label_test)

# ─── Model ────────────────────────────────────────────────────────────────────
TFN_MODELS = {'TensorFieldNetwork','GTTensorFieldNetwork','GTTensorFieldNetworkV2',
    'HierarchicalGTTFN','HierarchicalTensorFieldNetwork',
    'OnEquivariantTensorFieldNetwork','AttentionTensorFieldNetwork',
    'StochasticTensorFieldNetwork','CrossAttentionTensorFieldNetwork',
    'RelaxedOnEquivariantTensorFieldNetwork','HybridOnEquivariantTensorFieldNetwork'}
_npts = data_train[0].shape[0]

_hp = {'max_order': 0, 'hidden_channels': 8, 'num_layers': 2,
       'classifier_dims': [16], 'num_rbf': 64, 'k_neighbors': 8}

DS_HP = {'SonyAIBORobotSurface2': {'max_order':2,'hidden_channels':16,'num_layers':3},
    'MiddlePhalanxOutlineCorrect': {'max_order':1,'hidden_channels':8},
    'PowerCons': {'max_order':1,'hidden_channels':16},
    'ProximalPhalanxTW': {'max_order':1,'hidden_channels':8},
    'ECG5000': {'max_order':1,'hidden_channels':16},
    'CBF': {'max_order':1,'hidden_channels':8},
    'ItalyPowerDemand': {'max_order':1,'hidden_channels':8},
    'TwoLeadECG': {'max_order':1,'hidden_channels':8}}
if dataset_name in DS_HP:
    _hp.update(DS_HP[dataset_name])

def build_model(name):
    hp = _hp
    if name == 'PointNetTutorial':
        return PointNetTutorial(output_dim=output_dim)
    if name == 'PointNet3D':
        return PointNet3D(output_dim=output_dim)
    if name == 'DistanceMatrixRaggedModel':
        return DistanceMatrixRaggedModel(output_dim=output_dim, num_points=_npts)
    if name == 'ScalarDistanceDeepSet':
        return ScalarDistanceDeepSet(output_dim=output_dim)
    if name == 'ScalarInputMLP':
        return ScalarInputMLP(output_dim=output_dim)
    if name == 'MultiInputModel':
        return MultiInputModel(target_output_dim=output_dim, scalar_input_dim=1)
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim, **hp)
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=dim, num_classes=output_dim, radial_hidden=128, **hp)
    if name == 'OnEquivariantTensorFieldNetwork':
        return OnEquivariantTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64,32])
    if name == 'GTTensorFieldNetworkV2':
        return GTTensorFieldNetworkV2(n=dim, num_classes=output_dim, radial_hidden=128, **hp)
    if name == 'AttentionTensorFieldNetwork':
        return AttentionTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_heads=4,
            num_rbf=64, classifier_dims=[64,32], radial_hidden=64)
    if name == 'CrossAttentionTensorFieldNetwork':
        return CrossAttentionTensorFieldNetwork(num_classes=output_dim, n=dim,
            max_order=1, hidden_channels=8, num_layers=2, num_heads=4,
            transformer_layers=2, num_rbf=64, classifier_dims=[16], radial_hidden=64)
    if name == 'StochasticTensorFieldNetwork':
        return StochasticTensorFieldNetwork(num_classes=output_dim,
            num_mixtures=3, max_order=0, hidden_channels=8, num_layers=2,
            num_rbf=64, encoder_dims=[64,32])
    return build_model(name)  # recursive fallback works via globals?
    # Actually for safety let me just raise if unknown:
    # But TensorFieldNetwork was already handled; let's cover the rest.
    if name == 'RelaxedOnEquivariantTensorFieldNetwork':
        return RelaxedOnEquivariantTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64, classifier_dims=[64,32])
    if name == 'HybridOnEquivariantTensorFieldNetwork':
        return HybridOnEquivariantTensorFieldNetwork(num_classes=output_dim,
            max_order=1, hidden_channels=32, num_layers=3, num_rbf=64,
            classifier_dims=[64,32], non_eq_dim=128)
    raise ValueError(f"Unknown model: {name}")

model = build_model(model_name).to(device)
print(f"  Model: {model_name} ({sum(p.numel() for p in model.parameters())} params)")

# ─── Prepare inputs per model type ───────────────────────────────────────────
def prepare(data_list):
    if model_name in TFN_MODELS:
        return [torch.cat([x, x.new_zeros(x.shape[0],1)], dim=1) if x.shape[1]==2 else x
                for x in data_list]
    if model_name in ('PointNet3D', 'PointNetTutorial'):
        nc = 3 if model_name == 'PointNet3D' else 2
        return [torch.FloatTensor(
            np.concatenate([x.cpu().numpy(), np.zeros((x.shape[0], nc-x.shape[1]))], axis=1)
            if x.shape[1] < nc else x.cpu().numpy()[:,:nc]).to(device) for x in data_list]
    if model_name in ('ScalarDistanceDeepSet', 'DistanceMatrixRaggedModel'):
        out = []
        for x in data_list:
            a = x.cpu().numpy()
            m = np.linalg.norm(a[:,None] - a[None], axis=-1)
            out.append(torch.FloatTensor(m).to(device))
        return out
    if model_name == 'ScalarInputMLP':
        out = []
        for x in data_list:
            a = x.cpu().numpy()
            m = np.linalg.norm(a[:,None] - a[None], axis=-1)
            out.append(torch.FloatTensor([[m.mean()]]).to(device))
        return out
    if model_name == 'MultiInputModel':
        out = []
        for x in data_list:
            a = x.cpu().numpy()
            m = np.linalg.norm(a[:,None] - a[None], axis=-1)
            out.append((torch.FloatTensor(x).to(device),
                        torch.FloatTensor([[m.mean()]]).to(device)))
        return out
    return data_list

train_in = prepare(data_train_t)
test_in  = prepare(data_test_t)

# ─── TFN geometry ─────────────────────────────────────────────────────────────
def precompute_geom(model, data_list):
    if model_name not in TFN_MODELS: return None
    try:
        from gt_tfn_layer import knn_geometry
        inner = getattr(model, '_inner', model)
        _move_basis_tensors(inner, device)
        k = inner.k_neighbors
        rbfs, gts, nbrs = [], [], []
        for pc in data_list:
            r, g, n = knn_geometry(pc, inner.rbf, inner.gt_basis, k)
            rbfs.append(r.detach()); gts.append(g.detach()); nbrs.append(n.detach())
        if all(r.shape == rbfs[0].shape for r in rbfs):
            return {'rbf':torch.stack(rbfs),'gt_edge':torch.stack(gts),
                    'nbr_idx':torch.stack(nbrs),'uniform':True}
        return {'list':list(zip(rbfs,gts,nbrs)),'uniform':False}
    except: return None

train_geom = precompute_geom(model, train_in)
test_geom  = precompute_geom(model, test_in)

def get_geom(g, bix):
    if g is None: return None
    if isinstance(bix, range): bix = list(bix)
    if g.get('uniform', False):
        return {k:g[k][bix] if k in ('rbf','gt_edge','nbr_idx') else g[k]
                for k in g}
    return [g['list'][i] for i in bix]

def forward(model, batch_data, mname, geom=None):
    if mname == 'MultiInputModel':
        return model([x[0] for x in batch_data],
                     torch.cat([x[1] for x in batch_data]))
    if mname == 'ScalarInputMLP':
        return model(torch.cat([x.reshape(1,-1) for x in batch_data]))
    if geom is not None and mname in TFN_MODELS:
        inner = getattr(model, '_inner', model)
        _move_basis_tensors(inner, device)
        if isinstance(geom, dict) and geom.get('uniform',False) and hasattr(inner, '_encode_batch'):
            return inner._encode_batch(torch.stack(batch_data),
                precomputed_geom=(geom['rbf'],geom['gt_edge'],geom['nbr_idx']))
        geom_list = geom['list'] if isinstance(geom, dict) else geom
        descs = []
        for x, (r,g,n) in zip(batch_data, geom_list):
            r=r.squeeze(0) if r.ndim==4 else r; g=g.squeeze(0) if g.ndim==4 else g
            n=n.squeeze(0) if n.ndim==3 else n
            descs.append(inner._encode_single(x, precomputed_geom=(r,g,n)))
        return inner.rho(torch.stack(descs))
    return model(batch_data)

# ─── Training loop ────────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
bs = min(32, len(train_in))
n_t = len(train_in)

for ep in range(num_epochs):
    model.train()
    perm = np.random.permutation(n_t)
    loss_sum = 0.0
    for s in range(0, n_t, bs):
        bix = perm[s:s+bs]; bd = [train_in[i] for i in bix]; bt = targets_t[bix]
        bg = get_geom(train_geom, bix)
        optimizer.zero_grad(set_to_none=True)
        out = forward(model, bd, model_name, geom=bg)
        loss = criterion(out, bt)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(bd)
    if (ep+1) % 500 == 0:
        print(f"  ep {ep+1}/{num_epochs}  loss={loss_sum/n_t:.6f}")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
model.eval()
all_preds = []
with torch.no_grad():
    for s in range(0, len(test_in), bs):
        bd = test_in[s:s+bs]; bg = get_geom(test_geom, list(range(s,s+bs)))
        out = forward(model, bd, model_name, geom=bg)
        all_preds.append(out.detach().cpu().numpy())
PV_test = np.vstack(all_preds)

# Get train PV predictions for XGBoost
all_preds_train = []
with torch.no_grad():
    for s in range(0, len(train_in), bs):
        bd = train_in[s:s+bs]; bg = get_geom(train_geom, list(range(s,s+bs)))
        out = forward(model, bd, model_name, geom=bg)
        all_preds_train.append(out.detach().cpu().numpy())
PV_train = np.vstack(all_preds_train)

n_classes = len(np.unique(y_train))
if n_classes < 2:
    print(f"  WARNING: only {n_classes} class(es) in subsample, XGBoost would fail. Recording NaN.")
    tr_acc, te_acc = float('nan'), float('nan')
else:
    clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)
    clf.fit(PV_train, y_train)
    tr_acc = clf.score(PV_train, y_train)
    te_acc = clf.score(PV_test, y_test)
    print(f"  XGB  train={100*tr_acc:.2f}%  test={100*te_acc:.2f}%")

# ─── Save ──────────────────────────────────────────────────────────────────────
result = {
    'dataset': dataset_name, 'model': model_name,
    'fraction_pct': fraction_pct, 'trial': trial,
    'n_train_used': n_use, 'n_train_full': N_full, 'n_test': N_test,
    'xgb_train_acc': tr_acc, 'xgb_test_acc': te_acc,
}
out_path = f"results/ablation_train_{dataset_name}_{model_name}_{fraction_pct}pct_t{trial}.json"
json = __import__('json')
json.dump(result, open(out_path, 'w'))
print(f"  Saved to {out_path}")
