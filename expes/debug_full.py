"""Full pipeline debug on a failing dataset with CUDA_LAUNCH_BLOCKING."""
import os, sys, numpy as np
import dill as pck
import torch, torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from models import (_move_basis_tensors, TensorFieldNetwork, GTTensorFieldNetwork,
    GTTensorFieldNetworkV2, HierarchicalGTTFN, HierarchicalTensorFieldNetwork,
    OnEquivariantTensorFieldNetwork, AttentionTensorFieldNetwork,
    StochasticTensorFieldNetwork, CrossAttentionTensorFieldNetwork,
    RelaxedOnEquivariantTensorFieldNetwork, HybridOnEquivariantTensorFieldNetwork)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, Launch blocking: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")

ds_name = sys.argv[1] if len(sys.argv) > 1 else "ItalyPowerDemand"
model_name = sys.argv[2] if len(sys.argv) > 2 else "TensorFieldNetwork"

data = pck.load(open(f"datasets/{ds_name}_train_TDE311LS_5try1.pkl", 'rb'))
data_train = data["data_train"]
hdims = data["hdims"]
PV_train = data["PV_train"]
output_dim = sum(PV_train[h].shape[1] for h in range(len(hdims)))
dim = data_train[0].shape[1]
N_full = len(data_train)
_npts = data_train[0].shape[0]
print(f"Dataset: {ds_name}, N_full={N_full}, dim={dim}, out={output_dim}, npts={_npts}")

# Build model (same as train_ablation.py)
hp = {'max_order': 0, 'hidden_channels': 8, 'num_layers': 2,
      'classifier_dims': [16], 'num_rbf': 64, 'k_neighbors': 8}
DS_HP = {'SonyAIBORobotSurface2': {'max_order':2,'hidden_channels':16,'num_layers':3},
    'MiddlePhalanxOutlineCorrect': {'max_order':1,'hidden_channels':8},
    'PowerCons': {'max_order':1,'hidden_channels':16},
    'ProximalPhalanxTW': {'max_order':1,'hidden_channels':8},
    'ECG5000': {'max_order':1,'hidden_channels':16},
    'CBF': {'max_order':1,'hidden_channels':8},
    'ItalyPowerDemand': {'max_order':1,'hidden_channels':8},
    'TwoLeadECG': {'max_order':1,'hidden_channels':8}}
if ds_name in DS_HP:
    hp.update(DS_HP[ds_name])

def build_model(name):
    if name == 'TensorFieldNetwork':
        return TensorFieldNetwork(num_classes=output_dim, **hp)
    if name == 'GTTensorFieldNetwork':
        return GTTensorFieldNetwork(n=dim, num_classes=output_dim, radial_hidden=128, **hp)
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
    raise ValueError(f"Unknown: {name}")

model = build_model(model_name).to(device)
print(f"Model: {model_name}, params={sum(p.numel() for p in model.parameters())}")
inner = getattr(model, '_inner', model)
print(f"  inner: {type(inner).__name__}, k_neighbors={inner.k_neighbors}")

# Prepare
def prepare(data_list):
    return [torch.cat([x, x.new_zeros(x.shape[0],1)], dim=1) if x.shape[1]==2 else x
            for x in data_list]

data_train_t = [torch.FloatTensor(x).to(device) for x in data_train]
train_in = prepare(data_train_t)
print(f"Prepared {len(train_in)} samples, first shape={train_in[0].shape}")

# Precompute geometry (same as train_ablation.py)
from gt_tfn_layer import knn_geometry
try:
    train_geom = None
    print("Computing precompute_geom...")
    k = inner.k_neighbors
    rbfs, gts, nbrs = [], [], []
    for i, pc in enumerate(train_in):
        print(f"  knn_geometry sample {i}: N={pc.shape[0]}")
        r, g, n = knn_geometry(pc, inner.rbf, inner.gt_basis, k)
        print(f"    OK: max_idx={n.max().item()}")
        rbfs.append(r.detach()); gts.append(g.detach()); nbrs.append(n.detach())
    if all(r.shape == rbfs[0].shape for r in rbfs):
        train_geom = {'rbf':torch.stack(rbfs),'gt_edge':torch.stack(gts),
                      'nbr_idx':torch.stack(nbrs),'uniform':True}
    else:
        train_geom = {'list':list(zip(rbfs,gts,nbrs)),'uniform':False}
    print(f"Geometry computed: uniform={train_geom['uniform']}")
except Exception as e:
    print(f"precompute_geom FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    train_geom = None

# Test forward with geometry
if train_geom is not None:
    print("\nTesting forward with precomputed geometry...")
    from gt_tfn_layer import knn_geometry
    _move_basis_tensors(inner, device)
    if train_geom.get('uniform', False):
        if hasattr(inner, '_encode_batch'):
            batch = torch.stack(train_in[:2])
            geom = (train_geom['rbf'][:2], train_geom['gt_edge'][:2], train_geom['nbr_idx'][:2])
            print(f"  _encode_batch batch shape={batch.shape}")
            descs = inner._encode_batch(batch, precomputed_geom=geom)
            print(f"  _encode_batch OK: descs shape={descs.shape}")
        else:
            print("  no _encode_batch, trying _encode_single")
            for i in range(min(2, len(train_in))):
                x = train_in[i]
                r = train_geom['rbf'][i]; g = train_geom['gt_edge'][i]; n = train_geom['nbr_idx'][i]
                r = r.squeeze(0) if r.ndim == 4 else r
                g = g.squeeze(0) if g.ndim == 4 else g
                n = n.squeeze(0) if n.ndim == 3 else n
                desc = inner._encode_single(x, precomputed_geom=(r,g,n))
                print(f"  _encode_single[{i}] OK: desc shape={desc.shape}")
else:
    print("\nNo geometry available — testing model(batch_data) directly...")
    try:
        out = model(train_in[:2])
        print(f"  model(batch) OK: output shape={out.shape}")
    except Exception as e:
        print(f"  model(batch) FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
