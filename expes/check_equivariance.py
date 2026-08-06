"""
Rotation-invariance regression check for every TFN-style model.

Each model is built with small hyperparameters, run in eval mode on a random
batch of point clouds and on the same clouds rotated by a random rotation
(SO(3) in 3D, SO(2) in 2D); the maximum output difference must stay below a
tolerance.

Run from the repo root (or with the repo on PYTHONPATH):
    python expes/check_equivariance.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

import models

TOL = 5e-3


def check(name, model, dim=3):
    model.eval()
    torch.manual_seed(0)
    batch = [torch.randn(sz, dim) for sz in [20, 24, 16]]
    if dim == 2:
        a = torch.rand(1).item() * 2 * torch.pi
        c, s = torch.cos(torch.tensor(a)), torch.sin(torch.tensor(a))
        R = torch.tensor([[c, -s], [s, c]])
    else:
        R, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(R) < 0:
            R[:, 0] *= -1
    with torch.no_grad():
        y = model(batch)
        yr = model([pc @ R.T for pc in batch])
    if isinstance(y, (list, tuple)):
        y, yr = y[0], yr[0]
    return (y - yr).abs().max().item()


CASES = [
    ("TensorFieldNetwork",
     models.TensorFieldNetwork(num_classes=4, n=3, max_order=1, hidden_channels=8,
                               num_layers=2, k_neighbors=6)),
    ("GTTensorFieldNetwork",
     models.GTTensorFieldNetwork(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                 num_layers=2, k_neighbors=6)),
    ("GTTensorFieldNetworkV2",
     models.GTTensorFieldNetworkV2(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                   num_layers=2, k_neighbors=6)),
    ("GTTensorFieldNetworkWithAttention",
     models.GTTensorFieldNetworkWithAttention(n=3, num_classes=4, max_order=1,
                                              hidden_channels=8, num_layers=2,
                                              k_neighbors=6)),
    ("EndToEndTensorFieldNetwork",
     models.EndToEndTensorFieldNetwork(num_classes=4, max_order=1, hidden_channels=8,
                                       num_layers=2, k_neighbors=6)),
    ("OnEquivariantTensorFieldNetwork",
     models.OnEquivariantTensorFieldNetwork(n=3, num_classes=4, max_order=1,
                                            hidden_channels=8, num_layers=2, k_neighbors=6)),
    ("AttentionTensorFieldNetwork",
     models.AttentionTensorFieldNetwork(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                        num_layers=2, num_heads=2, num_rbf=8, k_neighbors=6)),
    ("StochasticTensorFieldNetwork",
     models.StochasticTensorFieldNetwork(n=3, num_classes=4, num_mixtures=2, max_order=1,
                                          hidden_channels=8, num_layers=2)),
    ("HierarchicalGTTFN",
     models.HierarchicalGTTFN(n=3, num_classes=4, max_order=1, hidden_channels=8,
                              stage_sizes=[16, 8], stage_radii=[1.0, 1.0],
                              k_local=4, num_layers_per_stage=1)),
    ("CrossAttentionTensorFieldNetwork",
     models.CrossAttentionTensorFieldNetwork(num_classes=4, n=3, max_order=1,
                                             hidden_channels=8, num_layers=1, num_heads=2,
                                             transformer_layers=1, num_rbf=8, k_neighbors=6)),
    ("GraphMambaTensorFieldNetwork",
     models.GraphMambaTensorFieldNetwork(num_classes=4, max_order=1, hidden_channels=8,
                                         num_layers=2, num_rbf=8, k_neighbors=6)),
    ("RelaxedOnEquivariantTensorFieldNetwork",
     models.RelaxedOnEquivariantTensorFieldNetwork(n=3, num_classes=4, max_order=1,
                                                   hidden_channels=8, num_layers=2,
                                                   k_neighbors=6)),
    ("StochasticEquivariantTFN",
     models.StochasticEquivariantTFN(n=3, num_classes=4, num_mixtures=2, max_order=1,
                                     hidden_channels=8, num_layers=2)),
    ("TemporalCrossAttentionTFN",
     models.TemporalCrossAttentionTFN(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                      num_layers=1, num_heads=2, transformer_layers=1,
                                      num_rbf=8, k_neighbors=6)),
    # HybridOnEquivariantTensorFieldNetwork is intentionally excluded: it fuses
    # raw (non-equivariant) point coordinates through HybridTFNClassifier.neq_phi,
    # so its output is rotation-dependent by design.
]

# 2D (SO(2)) gate: the time-series datasets used here are 2D and train_nn feeds
# them to TFN models in their native dimension (n=2), so 2D invariance is the
# actual deployment regime. Every n-capable model must also be invariant to 2D
# rotations.
CASES_2D = [
    ("TensorFieldNetwork",
     models.TensorFieldNetwork(num_classes=4, n=2, max_order=1, hidden_channels=8,
                               num_layers=2, k_neighbors=6)),
    ("GTTensorFieldNetwork",
     models.GTTensorFieldNetwork(n=2, num_classes=4, max_order=1, hidden_channels=8,
                                 num_layers=2, k_neighbors=6)),
    ("GTTensorFieldNetworkV2",
     models.GTTensorFieldNetworkV2(n=2, num_classes=4, max_order=1, hidden_channels=8,
                                   num_layers=2, k_neighbors=6)),
    ("GTTensorFieldNetworkWithAttention",
     models.GTTensorFieldNetworkWithAttention(n=2, num_classes=4, max_order=1,
                                              hidden_channels=8, num_layers=2,
                                              k_neighbors=6)),
    ("EndToEndTensorFieldNetwork",
     models.EndToEndTensorFieldNetwork(num_classes=4, n=2, max_order=1,
                                       hidden_channels=8, num_layers=2, k_neighbors=6)),
    ("OnEquivariantTensorFieldNetwork",
     models.OnEquivariantTensorFieldNetwork(n=2, num_classes=4, max_order=1,
                                            hidden_channels=8, num_layers=2, k_neighbors=6)),
    ("AttentionTensorFieldNetwork",
     models.AttentionTensorFieldNetwork(n=2, num_classes=4, max_order=1, hidden_channels=8,
                                        num_layers=2, num_heads=2, num_rbf=8, k_neighbors=6)),
    ("StochasticTensorFieldNetwork",
     models.StochasticTensorFieldNetwork(n=2, num_classes=4, num_mixtures=2, max_order=1,
                                          hidden_channels=8, num_layers=2)),
    ("HierarchicalGTTFN",
     models.HierarchicalGTTFN(n=2, num_classes=4, max_order=1, hidden_channels=8,
                              stage_sizes=[16, 8], stage_radii=[1.0, 1.0],
                              k_local=4, num_layers_per_stage=1)),
    ("CrossAttentionTensorFieldNetwork",
     models.CrossAttentionTensorFieldNetwork(num_classes=4, n=2, max_order=1,
                                             hidden_channels=8, num_layers=1, num_heads=2,
                                             transformer_layers=1, num_rbf=8, k_neighbors=6)),
    ("GraphMambaTensorFieldNetwork",
     models.GraphMambaTensorFieldNetwork(num_classes=4, n=2, max_order=1, hidden_channels=8,
                                         num_layers=2, num_rbf=8, k_neighbors=6)),
    ("RelaxedOnEquivariantTensorFieldNetwork",
     models.RelaxedOnEquivariantTensorFieldNetwork(n=2, num_classes=4, max_order=1,
                                                   hidden_channels=8, num_layers=2,
                                                   k_neighbors=6)),
    ("StochasticEquivariantTFN",
     models.StochasticEquivariantTFN(n=2, num_classes=4, num_mixtures=2, max_order=1,
                                     hidden_channels=8, num_layers=2)),
    ("TemporalCrossAttentionTFN",
     models.TemporalCrossAttentionTFN(n=2, num_classes=4, max_order=1, hidden_channels=8,
                                      num_layers=1, num_heads=2, transformer_layers=1,
                                      num_rbf=8, k_neighbors=6)),
]


def run():
    failures = []
    for name, model in CASES:
        try:
            d = check(name, model, dim=3)
            ok = d < TOL
            print(f"3D {name:40s} max_rot_diff = {d:.3e}  {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(name)
        except Exception as exc:
            failures.append(name)
            print(f"3D {name:40s} ERROR: {type(exc).__name__}: {str(exc)[:120]}")

    print()
    for name, model in CASES_2D:
        try:
            d = check(name, model, dim=2)
            ok = d < TOL
            print(f"2D {name:40s} max_rot_diff = {d:.3e}  {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{name} (2D)")
        except Exception as exc:
            failures.append(f"{name} (2D)")
            print(f"2D {name:40s} ERROR: {type(exc).__name__}: {str(exc)[:120]}")

    print()
    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {len(CASES) + len(CASES_2D)} MODELS ROTATION-INVARIANT (SO(3) + SO(2), tol={TOL})")


if __name__ == "__main__":
    run()
