import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from datasets.shapes3d import generate_dataset, DATASET_CONFIGS
from models import GTTensorFieldNetworkV2, _move_basis_tensors
from gt_tfn_layer import knn_geometry

torch.set_num_threads(4)

ckpt = "models/shape_shapes3d_complex_GTTensorFieldNetworkV2_t0-seed42.pt"
ck = torch.load(ckpt, map_location="cpu")

model = GTTensorFieldNetworkV2(
    n=3, num_classes=6, radial_hidden=128,
    max_order=0, hidden_channels=16, num_layers=2,
    classifier_dims=[32], num_rbf=64, k_neighbors=16,
)
model.load_state_dict(ck["model_state_dict"])
model.eval()

N = 600
base = "shapes3d_complex"
n_test = 300
_, _, data_test, y_test_raw, _ = generate_dataset(base, 0, n_test, N, noise_sigma=0.0, seed=42)
data_noisy, y_noisy_raw, _, _, _ = generate_dataset(base, n_test, 0, N, noise_sigma=0.15, seed=123)

le = LabelEncoder().fit(y_test_raw)
y_test = le.transform(y_test_raw)
y_noisy = le.transform(y_noisy_raw)

inner = getattr(model, "_inner", model)
_move_basis_tensors(inner, "cpu")
k = inner.k_neighbors

def predict(data, geom_from):
    preds = []
    for pc in data:
        with torch.no_grad():
            r, g, n = knn_geometry(torch.tensor(pc).float(), inner.rbf, inner.gt_basis, k)
            desc = inner._encode_single(torch.tensor(pc).float(), precomputed_geom=(r, g, n))
            preds.append(inner.rho(desc).argmax().item())
    return preds

def predict_with_clean_geom(noisy_data, clean_data):
    preds = []
    for pc_n, pc_c in zip(noisy_data, clean_data):
        with torch.no_grad():
            r, g, n = knn_geometry(torch.tensor(pc_c).float(), inner.rbf, inner.gt_basis, k)
            desc = inner._encode_single(torch.tensor(pc_n).float(), precomputed_geom=(r, g, n))
            preds.append(inner.rho(desc).argmax().item())
    return preds

print("clean acc:", accuracy_score(y_test, predict(data_test, None)))
pn = predict(data_noisy, None)
print("noisy acc (noisy geometry):", accuracy_score(y_noisy, pn))
pc = predict_with_clean_geom(data_noisy, data_test)
print("noisy acc (CLEAN geometry):", accuracy_score(y_noisy, pc))

print("confusion (noisy geometry):")
cm = confusion_matrix(y_noisy, pn)
print(cm)
print("row-recall:", (cm / cm.sum(1, keepdims=True)).diagonal())
print("pred dist:", cm.sum(0))
