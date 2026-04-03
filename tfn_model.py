"""
tfn_model.py  (v2 — all improvements integrated)
=================================================
Point cloud models: PointNet3D (baseline) and the full GT-TFN family.

Public API
----------
  PointNet3D               Deep-Sets baseline, 3D only (unchanged)
  TensorFieldNetwork       Original SO(3) TFN interface — now backed by GTTFNv2
  GTTensorFieldNetwork     n-dim SO(n) model (from gt_tfn_layer)
  GTTensorFieldNetworkV2   Recommended: all improvements, clean config
  HierarchicalGTTFN        Hierarchical pooling for large point clouds
  OnEquivariantWrapper     Lifts any SO(n) model to O(n)

All forward() calls accept  List[Tensor(N_i, n)]  as batch input.
GTTensorFieldNetworkV2 additionally accepts an optional node_attrs argument.

Improvement summary vs v1
--------------------------
  k-NN sparse neighborhoods   O(N·k) instead of O(N²)
  Gated nonlinearities        σ(linear(f0)) * f_type  per layer
  Residual streams            f_out[sig] += project(f_in[sig])
  Channel mixing              SiLU MLP over channels between layers
  type-2+ GT features         controlled by max_order
  Node attribute injection    extra scalar channels at input
  O(n) invariance             OnEquivariantWrapper averages over reflections
  Hierarchical pooling        HierarchicalGTTFN for large N
  Deeper radial MLP           3-layer instead of 2-layer
  SiLU + LayerNorm in rho     more stable classification head
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

# Core GT machinery
from gt_tfn_layer import (
    GTTensorFieldNetwork,
    RBFExpansion,
    ChannelMixer,
    EquivariantGate,
    ResidualProjection,
)
from gt_improvements import (
    HierarchicalGTTFN,
    OnEquivariantWrapper,
)
from gt_basis import GTSignature


# ============================================================================
# PointNet3D  (unchanged baseline)
# ============================================================================

class PointNet3D(nn.Module):
    """
    Deep-Sets / PointNet baseline.
    Permutation-invariant but NOT rotation-equivariant.
    forward(batch: List[Tensor(N_i, 3)]) → (B, output_dim)
    """

    def __init__(self, output_dim, phi_dims=[64, 128, 256], rho_dims=[256, 128]):
        super().__init__()
        phi_layers, in_f = [], 3
        for dim in phi_dims:
            phi_layers += [nn.Linear(in_f, dim), nn.ReLU()]
            in_f = dim
        self.phi_layers = nn.Sequential(*phi_layers)

        rho_layers, in_f = [], phi_dims[-1]
        for dim in rho_dims:
            rho_layers += [nn.Linear(in_f, dim), nn.ReLU()]
            in_f = dim
        rho_layers.append(nn.Linear(in_f, output_dim))
        self.rho_layers = nn.Sequential(*rho_layers)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.rho_layers[-1].out_features,
                               device=next(self.parameters()).device)
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
        mask   = torch.zeros(padded.shape[:2], dtype=torch.bool, device=padded.device)
        for i, pc in enumerate(batch):
            mask[i, :len(pc)] = True
        phi  = self.phi_layers(padded.reshape(-1, 3)).reshape(*padded.shape[:2], -1)
        phi  = phi.masked_fill(~mask.unsqueeze(-1), torch.finfo(phi.dtype).min)
        agg, _ = phi.max(dim=1)
        return self.rho_layers(agg)


# ============================================================================
# TensorFieldNetwork  (original interface — now backed by GTTFNv2 internals)
# ============================================================================

class TensorFieldNetwork(nn.Module):
    """
    SE(3)-equivariant point cloud classifier.
    Same interface as the original TFN; internally uses GTTensorFieldNetworkV2.
    """

    def __init__(
        self,
        num_classes:     int,
        max_order:       int        = 1,
        hidden_channels: int        = 32,
        num_layers:      int        = 4,
        num_rbf:         int        = 32,
        cutoff:          float      = 5.0,
        k_neighbors:     int        = 16,
        classifier_dims: List[int]  = [128, 64],
    ):
        super().__init__()
        self._inner = GTTensorFieldNetworkV2(
            n=3, num_classes=num_classes,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            classifier_dims=classifier_dims,
        )

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        return self._inner(batch)

    @property
    def rho(self):    return self._inner.rho
    @property
    def layers(self): return self._inner.mp_layers
    @property
    def rbf(self):    return self._inner.rbf


# ============================================================================
# GTTensorFieldNetworkV2  — the recommended model
# ============================================================================

class GTTensorFieldNetworkV2(GTTensorFieldNetwork):
    """
    Recommended GT-TFN with all improvements enabled by default.

    Inherits GTTensorFieldNetwork from gt_tfn_layer and exposes all
    improvement flags at the top level for easy configuration.

    New vs the base class
    ---------------------
    All improvements are ON by default (use_gate, use_residual,
    use_channel_mix, k_neighbors=16, radial_hidden=64).
    The constructor signature is otherwise identical.

    Example
    -------
    model = GTTensorFieldNetworkV2(
        n=3, num_classes=40, max_order=1,
        hidden_channels=64, num_layers=4,
        k_neighbors=16, node_attr_dim=0,
    )
    logits = model(batch)                   # List[Tensor(N_i, 3)]

    # With node attributes (e.g. atom type one-hot)
    logits = model(batch, node_attrs=attrs)  # attrs: List[Tensor(N_i, attr_dim)]
    """
    # GTTensorFieldNetworkV2 IS GTTensorFieldNetwork — the base class already
    # implements all improvements.  This subclass just:
    #   1. Changes the default hyper-parameters to the recommended values.
    #   2. Is re-exported from this file for discoverability.

    def __init__(
        self,
        n:               int,
        num_classes:     int,
        max_order:       int        = 1,
        hidden_channels: int        = 32,
        num_layers:      int        = 4,
        num_rbf:         int        = 32,
        cutoff:          float      = 5.0,
        k_neighbors:     int        = 16,
        use_gate:        bool       = True,
        use_residual:    bool       = True,
        use_channel_mix: bool       = True,
        node_attr_dim:   int        = 0,
        classifier_dims: List[int]  = [128, 64],
        radial_hidden:   int        = 64,
    ):
        super().__init__(
            n=n, num_classes=num_classes,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            use_gate=use_gate,
            use_residual=use_residual,
            use_channel_mix=use_channel_mix,
            node_attr_dim=node_attr_dim,
            classifier_dims=classifier_dims,
            radial_hidden=radial_hidden,
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "PointNet3D",
    "TensorFieldNetwork",
    "GTTensorFieldNetwork",
    "GTTensorFieldNetworkV2",
    "HierarchicalGTTFN",
    "OnEquivariantWrapper",
]


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── PointNet3D ────────────────────────────────────────────────────────
    print("=== PointNet3D ===")
    pnet  = PointNet3D(output_dim=10)
    batch3 = [torch.randn(n, 3) for n in [32, 64, 48]]
    print(f"  {pnet(batch3).shape}")

    # ── TensorFieldNetwork  (original interface) ───────────────────────────
    print("\n=== TensorFieldNetwork (via GTTFNv2) ===")
    tfn   = TensorFieldNetwork(num_classes=10, k_neighbors=8,
                               hidden_channels=16, num_layers=2)
    out   = tfn(batch3)
    print(f"  {out.shape}")
    R3, _ = torch.linalg.qr(torch.randn(3,3))
    if torch.det(R3)<0: R3[:,0]*=-1
    diff3 = (out - tfn([pc@R3.T for pc in batch3])).abs().max().item()
    print(f"  SO(3) invariance: {diff3:.2e}")

    # ── GTTensorFieldNetworkV2  (n=4) ─────────────────────────────────────
    print("\n=== GTTensorFieldNetworkV2 (n=4, with node attrs) ===")
    batch4 = [torch.randn(sz, 4) for sz in [16, 24, 20]]
    attrs4 = [torch.randn(sz, 3) for sz in [16, 24, 20]]
    m4     = GTTensorFieldNetworkV2(n=4, num_classes=8, hidden_channels=16,
                                    num_layers=2, k_neighbors=8, node_attr_dim=3)
    out4   = m4(batch4, node_attrs=attrs4)
    print(f"  {out4.shape}")
    R4, _ = torch.linalg.qr(torch.randn(4,4))
    if torch.det(R4)<0: R4[:,0]*=-1
    diff4  = (out4 - m4([pc@R4.T for pc in batch4], node_attrs=attrs4)).abs().max().item()
    print(f"  SO(4) invariance: {diff4:.2e}")

    # ── O(3) via wrapper ──────────────────────────────────────────────────
    print("\n=== OnEquivariantWrapper (O(3)) ===")
    base    = GTTensorFieldNetworkV2(n=3, num_classes=5, hidden_channels=8,
                                     num_layers=2, k_neighbors=8)
    o3_model = OnEquivariantWrapper(base)
    out_o3   = o3_model(batch3)
    F3       = torch.eye(3); F3[0,0] = -1
    diff_ref = (out_o3 - o3_model([pc@F3 for pc in batch3])).abs().max().item()
    print(f"  O(3) reflection invariance: {diff_ref:.2e}  {'OK' if diff_ref<1e-4 else 'FAIL'}")

    # ── HierarchicalGTTFN ─────────────────────────────────────────────────
    print("\n=== HierarchicalGTTFN ===")
    hier = HierarchicalGTTFN(n=3, num_classes=10, max_order=1,
                             hidden_channels=16, stage_sizes=[16, 4],
                             stage_radii=[0.3, 0.6], k_local=8, k_global=8,
                             num_layers_per_stage=1, node_attr_dim=0)
    print(f"  Params: {sum(p.numel() for p in hier.parameters()):,}")
    big_batch = [torch.randn(64, 3) for _ in range(3)]
    out_h = hier(big_batch)
    print(f"  {out_h.shape}")

    # ── Parameter comparison table ────────────────────────────────────────
    print("\n=== Parameter counts ===")
    models = [
        ("PointNet3D",            pnet),
        ("TensorFieldNetwork",    tfn),
        ("GTTFNv2 (n=3)",         base),
        ("GTTFNv2 (n=4)",         m4),
        ("HierarchicalGTTFN",     hier),
    ]
    for name, m in models:
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:30s}  {n_params:>10,}")
