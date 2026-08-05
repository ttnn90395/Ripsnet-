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


# ---------------------------------------------------------------------------
# Precomputed-geometry consistency
#
# The training/eval pipelines fast-path every TFN model through
# `inner._encode_single(pc, precomputed_geom=g)` + `inner.rho(stack)` (and the
# model-level `forward(..., precomputed_geom=...)` for CrossAttention).  If
# that path ever diverges from the direct forward, every experiment using the
# geometry cache silently computes the wrong thing, so both must agree on a
# mixed-size batch (which also catches torch.stack shape bugs).
# ---------------------------------------------------------------------------

GEOM_CASES = [
    ("GTTensorFieldNetwork",
     lambda: models.GTTensorFieldNetwork(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                         num_layers=2, k_neighbors=6), "fast"),
    ("GTTensorFieldNetworkV2",
     lambda: models.GTTensorFieldNetworkV2(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                           num_layers=2, k_neighbors=6), "fast"),
    ("GTTensorFieldNetworkWithAttention",
     lambda: models.GTTensorFieldNetworkWithAttention(n=3, num_classes=4, max_order=1,
                                                      hidden_channels=8, num_layers=2,
                                                      k_neighbors=6), "fast"),
    ("AttentionTensorFieldNetwork",
     lambda: models.AttentionTensorFieldNetwork(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                                num_layers=2, num_heads=2, num_rbf=8,
                                                k_neighbors=6), "fast"),
    ("StochasticTensorFieldNetwork",
     lambda: models.StochasticTensorFieldNetwork(n=3, num_classes=4, num_mixtures=2, max_order=1,
                                                 hidden_channels=8, num_layers=2), "fast"),
    ("GraphMambaTensorFieldNetwork",
     lambda: models.GraphMambaTensorFieldNetwork(num_classes=4, max_order=1, hidden_channels=8,
                                                 num_layers=2, num_rbf=8, k_neighbors=6), "fast"),
    ("TemporalCrossAttentionTFN",
     lambda: models.TemporalCrossAttentionTFN(n=3, num_classes=4, max_order=1, hidden_channels=8,
                                              num_layers=1, num_heads=2, transformer_layers=1,
                                              num_rbf=8, k_neighbors=6), "fast"),
    ("StochasticEquivariantTFN",
     lambda: models.StochasticEquivariantTFN(n=3, num_classes=4, num_mixtures=2, max_order=1,
                                             hidden_channels=8, num_layers=2), "fast"),
    ("CrossAttentionTensorFieldNetwork",
     lambda: models.CrossAttentionTensorFieldNetwork(num_classes=4, n=3, max_order=1,
                                                     hidden_channels=8, num_layers=1, num_heads=2,
                                                     transformer_layers=1, num_rbf=8,
                                                     k_neighbors=6), "model"),
]


def check_geom_paths():
    from collections import defaultdict

    from gt_tfn_layer import knn_geometry

    torch.manual_seed(0)
    failures = []
    for name, build, mode in GEOM_CASES:
        try:
            m = build()
            m.eval()
            inner = getattr(m, "_inner", m)
            # batch with a repeated size (20, 20, 17) to exercise the grouped
            # _encode_batch path used by train_nn.py forward_batch
            pcs = [torch.randn(20, 3) * 2.0, torch.randn(20, 3) * 2.0, torch.randn(17, 3) * 2.0]
            geoms = [knn_geometry(pc, inner.rbf, inner.gt_basis, inner.k_neighbors) for pc in pcs]

            with torch.no_grad():
                # reference: per-sample _encode_single + rho(stack)
                descs = [inner._encode_single(pc, precomputed_geom=g) for pc, g in zip(pcs, geoms)]
                out_ref = inner.rho(torch.stack(descs))

                # grouped path exactly as in train_nn.py forward_batch
                size_groups = defaultdict(list)
                for i, x in enumerate(pcs):
                    size_groups[x.shape[0]].append(i)
                descs = [None] * len(pcs)
                for sz, idxs in size_groups.items():
                    if len(idxs) == 1:
                        i = idxs[0]
                        descs[i] = inner._encode_single(pcs[i], precomputed_geom=geoms[i])
                    elif hasattr(inner, "_encode_batch"):
                        sub_batch = torch.stack([pcs[i] for i in idxs])
                        sub_rbf = torch.stack([geoms[i][0] for i in idxs])
                        sub_gt = torch.stack([geoms[i][1] for i in idxs])
                        sub_nbr = torch.stack([geoms[i][2] for i in idxs])
                        sub_out = inner._encode_batch(
                            sub_batch,
                            precomputed_geom=(sub_rbf, sub_gt, sub_nbr),
                            return_descriptors=True)
                        for off, i in enumerate(idxs):
                            descs[i] = sub_out[off]
                    else:
                        for i in idxs:
                            descs[i] = inner._encode_single(pcs[i], precomputed_geom=geoms[i])
                out_grouped = inner.rho(torch.stack(descs))

                if mode == "model":
                    out_direct = m(pcs, precomputed_geom=geoms)
                else:
                    out_direct = m(pcs)

            if isinstance(out_ref, (list, tuple)):
                out_ref = out_ref[0]
            if isinstance(out_grouped, (list, tuple)):
                out_grouped = out_grouped[0]
            if isinstance(out_direct, (list, tuple)):
                out_direct = out_direct[0]
            d1 = (out_ref - out_grouped).abs().max().item()
            d2 = (out_ref - out_direct).abs().max().item()
            if max(d1, d2) < 1e-5:
                print(f"GEOMOK {name:44s} grouped={d1:.2e} direct={d2:.2e}")
            else:
                failures.append(name)
                print(f"GEOMFAIL {name:44s} grouped={d1:.2e} direct={d2:.2e}")
        except Exception as exc:
            failures.append(name)
            print(f"GEOMFAIL {name:44s} {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} geometry failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {len(GEOM_CASES)} GEOMETRY PATHS CONSISTENT")


if __name__ == "__main__":
    run()
    check_geom_paths()
