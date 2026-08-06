"""Debug knn_geometry index OOB on a failing dataset."""
import os, sys, numpy as np
import dill as pck
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from gt_tfn_layer import knn_geometry

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ds_name = sys.argv[1] if len(sys.argv) > 1 else "ItalyPowerDemand"
model_name = sys.argv[2] if len(sys.argv) > 2 else "TensorFieldNetwork"

data = pck.load(open(f"datasets/{ds_name}_train_TDE311LS_5try1.pkl", 'rb'))
data_train = data["data_train"]
print(f"Dataset: {ds_name}, {len(data_train)} samples, dim={data_train[0].shape[1]}")

# Build model
if model_name == "TensorFieldNetwork":
    from models import TensorFieldNetwork
    model = TensorFieldNetwork(num_classes=sum(data["PV_train"][h].shape[1] for h in range(len(data["hdims"])))).to(device)
elif model_name == "GTTensorFieldNetwork":
    from models import GTTensorFieldNetwork
    model = GTTensorFieldNetwork(n=data_train[0].shape[1], num_classes=sum(data["PV_train"][h].shape[1] for h in range(len(data["hdims"])))).to(device)

inner = getattr(model, '_inner', model)
inner = inner.to(device)
print(f"Model: {type(inner).__name__}, k_neighbors={inner.k_neighbors}")

# Prepare data (add zero column if 2D)
def prepare(pc):
    if pc.shape[1] == 2:
        pc = np.concatenate([pc, np.zeros((pc.shape[0], 1))], axis=1)
    return torch.FloatTensor(pc).to(device)

# Test knn_geometry on each training sample
for i, pc_np in enumerate(data_train[:10]):
    pc = prepare(pc_np)
    N = pc.shape[0]
    k = min(inner.k_neighbors, N - 1)
    print(f"\nSample {i}: N={N}, n={pc.shape[1]}, k={k}")
    try:
        rbf, gt_edge, nbr_idx = knn_geometry(pc.clone(), inner.rbf, inner.gt_basis, k)
        print(f"  OK: rbf={rbf.shape}, gt={gt_edge.shape}, nbr={nbr_idx.shape}  max_idx={nbr_idx.max().item()}")
    except Exception as e:
        print(f"  FAIL: {e}")
        # Check nbr_idx from topk directly
        diff = pc.unsqueeze(1) - pc.unsqueeze(0)
        dist = diff.norm(dim=-1)
        mask = torch.eye(N, dtype=torch.bool, device=device)
        dist_masked = dist.masked_fill(mask, float('inf'))
        vals, idx = dist_masked.topk(k, dim=-1, largest=False)
        print(f"  nbr_idx from topk: max={idx.max().item()}, min={idx.min().item()}, N={N}")
        print(f"  any OOB: {(idx >= N).any().item() or (idx < 0).any().item()}")
