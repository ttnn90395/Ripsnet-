"""
Rotation-invariance regression check for every TFN-style model.

Each model is built with small hyperparameters, run in eval mode on a random
batch of 3D point clouds and on the same clouds rotated by a random SO(3)
element; the maximum output difference must stay below a tolerance.

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


def check(name, model):
    model.eval()
    torch.manual_seed(0)
    batch = [torch.randn(sz, 3) for sz in [20, 24, 16]]
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


def run():
    failures = []
    for name, model in CASES:
        try:
            d = check(name, model)
            ok = d < TOL
            print(f"{name:42s} max_rot_diff = {d:.3e}  {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(name)
        except Exception as exc:
            failures.append(name)
            print(f"{name:42s} ERROR: {type(exc).__name__}: {str(exc)[:120]}")

    print()
    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {len(CASES)} MODELS ROTATION-INVARIANT (tol={TOL})")


if __name__ == "__main__":
    run()
