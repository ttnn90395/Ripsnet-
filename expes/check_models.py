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
X2d = torch.randn(24, 2) * 2.0

CASES = [
    ("TensorFieldNetwork",
     lambda: models.TensorFieldNetwork(num_classes=3),
     lambda m: m([X])),
    ("TFNTensorFieldNetwork",
     lambda: models.TFNTensorFieldNetwork(num_classes=3, hidden_f0=8, hidden_f1=4,
                                          num_layers=2, num_rbf=8, classifier_dims=(16, 8)),
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
# 2D (n=2) forward gate
#
# The time-series datasets here are 2D and train_nn feeds every n-capable TFN
# model raw n-dim point clouds (n=2), so each wrapper must build and forward at
# n=2.  Several wrappers used to hardcode n=3 internally and crashed with a
# mat1/mat2 shape error on 2D data; this gate keeps the n=2 path working for
# every n-capable model, including GTTFNEncoder / EquivariantGraphMambaNetwork
# (not covered by the SO(2) equivariance gate).
# ---------------------------------------------------------------------------

D2_CASES = [
    ("TensorFieldNetwork",
     lambda: models.TensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                       num_layers=2, k_neighbors=6),
     lambda m: m([X2d])),
    ("GTTensorFieldNetwork",
     lambda: models.GTTensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                         num_layers=2, k_neighbors=6),
     lambda m: m([X2d])),
    ("GTTensorFieldNetworkV2",
     lambda: models.GTTensorFieldNetworkV2(n=2, num_classes=3, hidden_channels=8,
                                           num_layers=2, k_neighbors=6),
     lambda m: m([X2d])),
    ("GTTensorFieldNetworkWithAttention",
     lambda: models.GTTensorFieldNetworkWithAttention(n=2, num_classes=3, hidden_channels=8,
                                                      num_layers=2, k_neighbors=6),
     lambda m: m([X2d])),
    ("HierarchicalGTTFN",
     lambda: models.HierarchicalGTTFN(n=2, num_classes=3, hidden_channels=8,
                                      stage_sizes=[16, 8], stage_radii=[1.0, 1.0],
                                      k_local=4, num_layers_per_stage=1),
     lambda m: m([X2d])),
    ("HierarchicalTensorFieldNetwork",
     lambda: models.HierarchicalTensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                                   stage_sizes=[16, 8], stage_radii=[1.0, 1.0],
                                                   k_local=4, num_layers_per_stage=1),
     lambda m: m([X2d])),
    ("OnEquivariantTensorFieldNetwork",
     lambda: models.OnEquivariantTensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                                    num_layers=2, k_neighbors=6),
     lambda m: m([X2d])),
    ("AttentionTensorFieldNetwork",
     lambda: models.AttentionTensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                                num_layers=2, num_heads=2, num_rbf=8,
                                                k_neighbors=6),
     lambda m: m([X2d])),
    ("CrossAttentionTensorFieldNetwork",
     lambda: models.CrossAttentionTensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                                     num_layers=1, num_heads=2,
                                                     transformer_layers=1, num_rbf=8,
                                                     k_neighbors=6),
     lambda m: m([X2d])),
    ("StochasticTensorFieldNetwork",
     lambda: models.StochasticTensorFieldNetwork(n=2, num_classes=3, num_mixtures=2,
                                                 hidden_channels=8, num_layers=2),
     lambda m: m([X2d])),
    ("RelaxedOnEquivariantTensorFieldNetwork",
     lambda: models.RelaxedOnEquivariantTensorFieldNetwork(n=2, num_classes=3,
                                                           hidden_channels=8, num_layers=2,
                                                           k_neighbors=6),
     lambda m: m([X2d])),
    ("EndToEndTensorFieldNetwork",
     lambda: models.EndToEndTensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                               num_layers=2, k_neighbors=6),
     lambda m: m([X2d])),
    ("TemporalCrossAttentionTFN",
     lambda: models.TemporalCrossAttentionTFN(n=2, num_classes=3, hidden_channels=8,
                                              num_layers=1, num_heads=2, transformer_layers=1,
                                              num_rbf=8, k_neighbors=6),
     lambda m: m([X2d])),
    ("StochasticEquivariantTFN",
     lambda: models.StochasticEquivariantTFN(n=2, num_classes=3, num_mixtures=2,
                                             hidden_channels=8, num_layers=2),
     lambda m: m([X2d])),
    ("EquivariantGraphMambaNetwork",
     lambda: models.EquivariantGraphMambaNetwork(n=2, num_classes=3, hidden_channels=8,
                                                 num_layers=2, num_rbf=8, k_neighbors=6),
     lambda m: m([X2d])),
    ("GraphMambaTensorFieldNetwork",
     lambda: models.GraphMambaTensorFieldNetwork(n=2, num_classes=3, hidden_channels=8,
                                                 num_layers=2, num_rbf=8, k_neighbors=6),
     lambda m: m([X2d, X2d[:18]])),
    ("GTTFNEncoder",
     lambda: models.GTTFNEncoder(n=2, embedding_dim=8),
     lambda m: m([X2d])),
    ("SetTransformerTensorFieldNetwork",
     lambda: models.SetTransformerTensorFieldNetwork(n=2, num_classes=3, embedding_dim=8,
                                                     max_order=1, hidden_channels=8,
                                                     num_layers=2, num_heads=2, num_rbf=8,
                                                     k_neighbors=6),
     lambda m: m([X2d], [X2d[:18]])),
]


def check_2d_forward():
    failures = []
    for name, build, call in D2_CASES:
        try:
            m = build()
            m.eval()
            out = call(m)
            if isinstance(out, (list, tuple)):
                out = out[0]
            print(f"2DOK   {name:45s} n=2 fwd out={tuple(out.shape)}")
        except Exception as exc:
            failures.append(name)
            print(f"2DFAIL {name:45s} {type(exc).__name__}: {str(exc)[:90]}")

    print()
    if failures:
        print(f"{len(failures)} 2D failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {len(D2_CASES)} MODELS FORWARD AT n=2")


# ---------------------------------------------------------------------------
# Backward smoke
#
# Train-mode forward + loss.backward() must succeed and produce finite grads.
# Not every parameter is expected to receive a gradient in this toy setup
# (e.g. the stochastic models' logvar_net is only driven by their NLL loss,
# the hybrid models intentionally bypass the backbone's rho head, and
# PermopRagged has no trainable params), so we only fail on a backward
# exception, a non-finite loss, or zero finite grads — i.e. real autograd
# breakage.
# ---------------------------------------------------------------------------

def check_backward():
    torch.manual_seed(0)
    failures = []
    for name, build, call in CASES:
        try:
            m = build()
            m.train()
            with torch.enable_grad():
                out = call(m)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if not torch.is_floating_point(out) or not out.requires_grad:
                    print(f"BWSKIP {name:43s} non-differentiable output")
                    continue
                loss = out.square().mean()
                if not torch.isfinite(loss):
                    failures.append(name)
                    print(f"BWFAIL {name:43s} loss not finite")
                    continue
                loss.backward()
            grads = [p.grad for p in m.parameters() if p.requires_grad]
            finite = sum(1 for g in grads if g is not None and torch.isfinite(g).all())
            if finite > 0:
                print(f"BWOK   {name:43s} grads {finite}/{len(grads)}")
            else:
                failures.append(name)
                print(f"BWFAIL {name:43s} no finite grads")
        except Exception as exc:
            failures.append(name)
            print(f"BWFAIL {name:43s} {type(exc).__name__}: {str(exc)[:90]}")
    if failures:
        print(f"{len(failures)} backward failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {len(CASES)} BACKWARD PASSES OK")


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


# ---------------------------------------------------------------------------
# Uniform-geometry cache regression
#
# When every point cloud in a dataset has the same size, precompute_geometry
# stores a stacked dict {uniform, rbf, gt_edge, nbr_idx} with NO 'list' key.
# Models without inner._encode_batch must still be served per-sample geometry
# derived from those stacked tensors; reaching for geom['list'] used to raise
# KeyError.  This gate replays the exact branch logic of the training/eval
# pipelines and checks that the uniform-cache route equals the per-sample
# list route for every geometry-capable model.
# ---------------------------------------------------------------------------

def check_uniform_cache():
    from gt_tfn_layer import knn_geometry

    torch.manual_seed(0)
    failures = []
    for name, build, mode in GEOM_CASES:
        try:
            m = build()
            m.eval()
            inner = getattr(m, "_inner", m)
            n_pc = 16
            pcs = [torch.randn(20, 3) * 2.0 for _ in range(n_pc)]
            geoms = [knn_geometry(pc, inner.rbf, inner.gt_basis, inner.k_neighbors)
                     for pc in pcs]

            # assemble the stacked 'uniform' cache exactly as
            # precompute_geometry() does -- note: no 'list' key
            cache = {
                "uniform": True,
                "rbf":     torch.stack([g[0] for g in geoms]),
                "gt_edge": torch.stack([g[1] for g in geoms]),
                "nbr_idx": torch.stack([g[2] for g in geoms]),
            }

            with torch.no_grad():
                # per-sample list route (the non-uniform cache equivalent)
                if mode == "model":
                    out_list = m(pcs, precomputed_geom=geoms)
                else:
                    descs = [inner._encode_single(pc, precomputed_geom=g)
                             for pc, g in zip(pcs, geoms)]
                    out_list = inner.rho(torch.stack(descs))

                # uniform-cache route, sliced per mini-batch.  For models
                # without _encode_batch the per-sample geometry is zipped out
                # of the stacked tensors; accessing cache['list'] would be the
                # KeyError this gate guards against.
                outs = []
                bs = 7
                for s in range(0, n_pc, bs):
                    sl = slice(s, s + bs)
                    if mode == "model":
                        gl = list(zip(cache["rbf"][sl], cache["gt_edge"][sl],
                                      cache["nbr_idx"][sl]))
                        out = m(pcs[s:s + bs], precomputed_geom=gl)
                    elif hasattr(inner, "_encode_batch"):
                        out = inner._encode_batch(
                            torch.stack(pcs[s:s + bs]),
                            precomputed_geom=(cache["rbf"][sl], cache["gt_edge"][sl],
                                              cache["nbr_idx"][sl]))
                    else:
                        gl = list(zip(cache["rbf"][sl], cache["gt_edge"][sl],
                                      cache["nbr_idx"][sl]))
                        descs = [inner._encode_single(pc, precomputed_geom=(r, g, n))
                                 for pc, (r, g, n) in zip(pcs[s:s + bs], gl)]
                        out = inner.rho(torch.stack(descs))
                    outs.append(out)
                out_cache = torch.cat(outs, dim=0)

            if isinstance(out_list, (list, tuple)):
                out_list = out_list[0]
            if isinstance(out_cache, (list, tuple)):
                out_cache = out_cache[0]
            d = (out_list - out_cache).abs().max().item()
            if d < 1e-5:
                print(f"CACHEOK {name:44s} uniform={d:.2e}")
            else:
                failures.append(name)
                print(f"CACHEFAIL {name:44s} uniform={d:.2e}")
        except Exception as exc:
            failures.append(name)
            print(f"CACHEFAIL {name:44s} {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} uniform-cache failure(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {len(GEOM_CASES)} UNIFORM-CACHE PATHS CONSISTENT")


# ---------------------------------------------------------------------------
# Hierarchical stage-geometry cache
#
# HierarchicalTensorFieldNetwork has no _encode_batch, so every pipeline
# serves it per-sample geometry from the cache: the k-NN tensors plus the
# per-stage pooling geometry produced by precompute_hierarchical_geometry().
# This gate replays the forward_batch uniform branch (stacked rbf/gt/nbr +
# hier list) and asserts the cached-stage shortcut is bit-identical to the
# direct forward, which recomputes all stage geometry internally.
# ---------------------------------------------------------------------------

def check_hier_cache():
    from gt_tfn_layer import knn_geometry

    torch.manual_seed(0)
    failures = []
    try:
        m = models.HierarchicalTensorFieldNetwork(
            num_classes=4, n=3, max_order=1, hidden_channels=8,
            stage_sizes=[32, 16], stage_radii=[0.3, 0.6],
            k_local=8, k_global=8, num_layers_per_stage=1,
            num_rbf=8, cutoff=1.5)
        m.eval()
        inner = m._inner
        n_pc = 8
        pcs = [torch.randn(64, 3) * 2.0 for _ in range(n_pc)]
        geoms = [knn_geometry(pc, inner.rbf, inner.gt_basis, inner.k_neighbors)
                 for pc in pcs]
        stages = [inner.precompute_hierarchical_geometry(pc) for pc in pcs]
        cache = {
            "uniform": True,
            "rbf":     torch.stack([g[0] for g in geoms]),
            "gt_edge": torch.stack([g[1] for g in geoms]),
            "nbr_idx": torch.stack([g[2] for g in geoms]),
            "hier":    stages,
        }
        with torch.no_grad():
            out_direct = m(pcs)   # recomputes all stage geometry internally

            # uniform-cache route exactly as forward_batch()'s no-_encode_batch
            # branch: per-sample rbf/gt/nbr + stage geometry from the cache
            outs = []
            bs = 5
            for s in range(0, n_pc, bs):
                descs = []
                for i, pc in enumerate(pcs[s:s + bs]):
                    descs.append(inner._encode_single(
                        pc,
                        precomputed_geom=(cache["rbf"][s + i], cache["gt_edge"][s + i],
                                          cache["nbr_idx"][s + i]),
                        precomputed_stage_geom=cache["hier"][s + i]))
                outs.append(inner.rho(torch.stack(descs)))
            out_cache = torch.cat(outs, dim=0)

        if isinstance(out_direct, (list, tuple)):
            out_direct = out_direct[0]
        if isinstance(out_cache, (list, tuple)):
            out_cache = out_cache[0]
        d = (out_direct - out_cache).abs().max().item()
        if d < 1e-5:
            print(f"HIEROK  HierarchicalTensorFieldNetwork  cached-stage={d:.2e}")
        else:
            failures.append("HierarchicalTensorFieldNetwork")
            print(f"HIERFAIL HierarchicalTensorFieldNetwork  cached-stage={d:.2e}")
    except Exception as exc:
        failures.append("HierarchicalTensorFieldNetwork")
        print(f"HIERFAIL HierarchicalTensorFieldNetwork  {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} hierarchical-cache failure(s): {', '.join(failures)}")
        sys.exit(1)
    print("HIERARCHICAL STAGE-GEOMETRY CACHE CONSISTENT")


if __name__ == "__main__":
    run()
    check_geom_paths()
    check_uniform_cache()
    check_hier_cache()
    check_2d_forward()
    check_backward()
