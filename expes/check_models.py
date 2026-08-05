"""
Construct-and-forward smoke test for every model exported by models.__all__.

Run from the repo root (or with the repo on PYTHONPATH):
    python expes/check_models.py

Prints one line per model; exits non-zero if any model fails to build
and forward a representative input.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

import models

torch.set_grad_enabled(False)

X  = torch.randn(24, 3) * 2.0
X2 = torch.randn(18, 3) * 2.0

CASES = [
    ("TensorFieldNetwork",
     lambda: models.TensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("GTTensorFieldNetwork",
     lambda: models.GTTensorFieldNetwork(n=3, num_classes=3),
     lambda m: m([X])),
    ("GTTensorFieldNetworkV2",
     lambda: models.GTTensorFieldNetworkV2(n=3, num_classes=3),
     lambda m: m([X])),
    ("GTTensorFieldNetworkWithAttention",
     lambda: models.GTTensorFieldNetworkWithAttention(n=3, num_classes=3),
     lambda m: m([X])),
    ("HierarchicalGTTFN",
     lambda: models.HierarchicalGTTFN(n=3, num_classes=3),
     lambda m: m([X])),
    ("HierarchicalTensorFieldNetwork",
     lambda: models.HierarchicalTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("OnEquivariantTensorFieldNetwork",
     lambda: models.OnEquivariantTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("AttentionTensorFieldNetwork",
     lambda: models.AttentionTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("CrossAttentionTensorFieldNetwork",
     lambda: models.CrossAttentionTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("StochasticTensorFieldNetwork",
     lambda: models.StochasticTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("RelaxedOnEquivariantTensorFieldNetwork",
     lambda: models.RelaxedOnEquivariantTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("HybridOnEquivariantTensorFieldNetwork",
     lambda: models.HybridOnEquivariantTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("EndToEndTensorFieldNetwork",
     lambda: models.EndToEndTensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("TemporalCrossAttentionTFN",
     lambda: models.TemporalCrossAttentionTFN(n=3, num_classes=3),
     lambda m: m([X])),
    ("StochasticEquivariantTFN",
     lambda: models.StochasticEquivariantTFN(n=3, num_classes=3),
     lambda m: m([X])),
    ("EquivariantGraphMambaNetwork",
     lambda: models.EquivariantGraphMambaNetwork(n=3, num_classes=3),
     lambda m: m([X])),
    ("GraphMambaTensorFieldNetwork",
     lambda: models.GraphMambaTensorFieldNetwork(num_classes=3),
     lambda m: m([X, X2])),
    ("GTTFNEncoder",
     lambda: models.GTTFNEncoder(n=3, embedding_dim=8),
     lambda m: m([X])),
    ("PointNet3D",
     lambda: models.PointNet3D(output_dim=3),
     lambda m: m([X])),
    ("PointNetTutorial",
     lambda: models.PointNetTutorial(output_dim=3),
     lambda m: m([X])),
    ("RipsPointNet",
     lambda: models.RipsPointNet(output_dim=3),
     lambda m: m([X])),
    ("AttnRipsPointNet",
     lambda: models.AttnRipsPointNet(output_dim=3, input_dim=3),
     lambda m: m([X])),
    ("ScalarDistanceDeepSet",
     lambda: models.ScalarDistanceDeepSet(output_dim=3),
     lambda m: m([X])),
    ("ScalarInputMLP",
     lambda: models.ScalarInputMLP(output_dim=3),
     lambda m: m(torch.randn(1, 1))),
    ("RaggedPersistenceModel",
     lambda: models.RaggedPersistenceModel(output_dim=3),
     lambda m: m([X])),
    ("PermopRagged",
     lambda: models.PermopRagged(),
     lambda m: m([X])),
    ("DistanceMatrixRaggedModel",
     lambda: models.DistanceMatrixRaggedModel(output_dim=3),
     lambda m: m([torch.cdist(X, X)])),
    ("DenseRagged",
     lambda: models.DenseRagged(),
     lambda m: m(torch.randn(2, 24, 3))),
    ("MultiInputModel",
     lambda: models.MultiInputModel(target_output_dim=3, scalar_input_dim=1),
     lambda m: m([X], torch.randn(1, 1))),
    ("OnEquivariantWrapper",
     lambda: models.OnEquivariantWrapper(models.GTTensorFieldNetwork(n=3, num_classes=10)),
     lambda m: m([X])),
    ("RelaxedEquivariantTFN",
     lambda: models.RelaxedEquivariantTFN(models.GTTensorFieldNetwork(n=3, num_classes=10)),
     lambda m: m([X])),
    ("EndToEndClassifier",
     lambda: models.EndToEndClassifier(
         tfn_backbone=models.GTTensorFieldNetwork(n=3, num_classes=10), num_classes=3),
     lambda m: m([X])),
    ("HybridTFNClassifier",
     lambda: models.HybridTFNClassifier(
         tfn_backbone=models.GTTensorFieldNetwork(n=3, num_classes=10), output_dim=3),
     lambda m: m([X])),
    ("EquivariantSetTransformer",
     lambda: models.EquivariantSetTransformer(n=3, num_classes=3),
     lambda m: m([X], [X])),
    ("SetTransformerTensorFieldNetwork",
     lambda: models.SetTransformerTensorFieldNetwork(num_classes=3),
     lambda m: m([X], [X])),
]


def run():
    failures = []
    for name, build, call in CASES:
        try:
            m = build()
            m.eval()
            out = call(m)
            if isinstance(out, (list, tuple)):
                out = out[0]
            print(f"OK   {name:45s} out={tuple(out.shape)}")
        except Exception as exc:
            failures.append(name)
            print(f"FAIL {name:45s} {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {len(CASES)} MODELS OK")


if __name__ == "__main__":
    run()
