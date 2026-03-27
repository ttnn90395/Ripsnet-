"""
tfn_model.py  (updated)
=======================
Point cloud models with SE(n)-equivariance via the GT basis.

Public API (unchanged from the original file)
---------------------------------------------
  PointNet3D              — baseline DeepSets model, 3D only
  TensorFieldNetwork      — original interface, now backed by GTTensorFieldNetwork(n=3)
  GTTensorFieldNetwork    — new: works for any ambient dimension n

Backward compatibility
----------------------
All existing call sites of PointNet3D and TensorFieldNetwork continue to work
without modification.  The new GTTensorFieldNetwork is available for n != 3.

Example
-------
    # 3D classification — same as before
    model = TensorFieldNetwork(num_classes=40)
    logits = model(batch)   # batch: List[Tensor(N_i, 3)]

    # 4D classification — new
    model4 = GTTensorFieldNetwork(n=4, num_classes=10)
    logits4 = model4(batch4)  # batch4: List[Tensor(N_i, 4)]
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List

# GT machinery (new)
from gt_tfn_layer import GTTensorFieldNetwork   # re-exported below


# ===========================================================================
# PointNet3D  (unchanged)
# ===========================================================================

class PointNet3D(nn.Module):
    """
    Deep-Sets / PointNet baseline.
    Permutation-invariant but NOT rotation-equivariant.
    Interface: forward(batch: List[Tensor(N_i, 3)]) → (B, output_dim)
    """

    def __init__(self, output_dim, phi_dims=[64, 128, 256], rho_dims=[256, 128]):
        super().__init__()

        # Phi: shared MLP applied to each point
        phi_layers = []
        in_features = 3
        for dim in phi_dims:
            phi_layers.append(nn.Linear(in_features, dim))
            phi_layers.append(nn.ReLU())
            in_features = dim
        self.phi_layers = nn.Sequential(*phi_layers)

        # Rho: global MLP applied to max-pooled features
        rho_layers = []
        in_features = phi_dims[-1]
        for dim in rho_dims:
            rho_layers.append(nn.Linear(in_features, dim))
            rho_layers.append(nn.ReLU())
            in_features = dim
        rho_layers.append(nn.Linear(in_features, output_dim))
        self.rho_layers = nn.Sequential(*rho_layers)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(
                0, self.rho_layers[-1].out_features,
                device=next(self.parameters()).device,
            )

        padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)

        mask = torch.zeros(
            padded_batch.shape[0], padded_batch.shape[1],
            dtype=torch.bool, device=padded_batch.device,
        )
        for i, pc in enumerate(batch):
            mask[i, :len(pc)] = True

        phi_out = self.phi_layers(padded_batch.view(-1, padded_batch.shape[-1]))
        phi_out = phi_out.view(padded_batch.shape[0], padded_batch.shape[1], -1)

        min_val = torch.finfo(phi_out.dtype).min
        masked_phi_out = phi_out.masked_fill(~mask.unsqueeze(-1), min_val)

        aggregated_features, _ = torch.max(masked_phi_out, dim=1)
        return self.rho_layers(aggregated_features)


# ===========================================================================
# TensorFieldNetwork  (interface unchanged, now backed by GTTensorFieldNetwork)
# ===========================================================================

class TensorFieldNetwork(nn.Module):
    """
    SE(3)-equivariant point cloud classifier.

    Interface is identical to the original TensorFieldNetwork.
    Internally delegates to GTTensorFieldNetwork(n=3, ...) so the GT basis
    and CG recursion are used even for the 3D case.

    Parameters
    ----------
    num_classes      : int    number of output classes
    max_order        : int    maximum GT feature order (1 = scalars + vectors,
                              2 = adds quadrupole-type features)
    hidden_channels  : int    feature channels per irrep type per layer
    num_layers       : int    number of equivariant layers
    num_rbf          : int    RBF basis size
    cutoff           : float  RBF distance cutoff
    classifier_dims  : list   hidden dims of the final classification MLP
    """

    def __init__(
        self,
        num_classes:     int,
        max_order:       int        = 1,
        hidden_channels: int        = 16,
        num_layers:      int        = 3,
        num_rbf:         int        = 16,
        cutoff:          float      = 5.0,
        classifier_dims: List[int]  = [64, 32],
    ):
        super().__init__()
        self._inner = GTTensorFieldNetwork(
            n               = 3,
            num_classes     = num_classes,
            max_order       = max_order,
            hidden_channels = hidden_channels,
            num_layers      = num_layers,
            num_rbf         = num_rbf,
            cutoff          = cutoff,
            classifier_dims = classifier_dims,
        )

    # Expose inner parameters transparently
    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        batch  : List of (N_i, 3) tensors
        returns: (B, num_classes) logits
        """
        return self._inner(batch)

    # Keep the old attribute interface for anyone who inspected .rho etc.
    @property
    def rho(self):
        return self._inner.rho

    @property
    def layers(self):
        return self._inner.layers

    @property
    def rbf(self):
        return self._inner.rbf


# ===========================================================================
# Re-export GTTensorFieldNetwork for direct n-dim use
# ===========================================================================

__all__ = [
    "PointNet3D",
    "TensorFieldNetwork",
    "GTTensorFieldNetwork",
]


# ===========================================================================
# __main__ — comparison / regression test
# ===========================================================================

if __name__ == "__main__":
    import math
    torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # 1.  PointNet3D  (unchanged behaviour)
    # -----------------------------------------------------------------------
    print("=== PointNet3D ===")
    pnet = PointNet3D(output_dim=10)
    batch3 = [torch.randn(n, 3) for n in [32, 64, 48]]
    out_pnet = pnet(batch3)
    print(f"  output shape: {out_pnet.shape}")

    # -----------------------------------------------------------------------
    # 2.  TensorFieldNetwork  (3D, original interface)
    # -----------------------------------------------------------------------
    print("\n=== TensorFieldNetwork (n=3, via GT) ===")
    tfn = TensorFieldNetwork(num_classes=10, max_order=1,
                             hidden_channels=8, num_layers=2)
    out_tfn = tfn(batch3)
    print(f"  output shape: {out_tfn.shape}")

    # SE(3) invariance
    R3, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(R3) < 0:
        R3[:, 0] *= -1
    out_rot = tfn([pc @ R3.T for pc in batch3])
    diff3 = (out_tfn - out_rot).abs().max().item()
    print(f"  SE(3) invariance (max diff): {diff3:.2e}")

    # -----------------------------------------------------------------------
    # 3.  GTTensorFieldNetwork  (4D)
    # -----------------------------------------------------------------------
    print("\n=== GTTensorFieldNetwork (n=4) ===")
    batch4 = [torch.randn(n, 4) for n in [16, 24, 20]]
    model4 = GTTensorFieldNetwork(n=4, num_classes=8, max_order=1,
                                  hidden_channels=8, num_layers=2)
    out4 = model4(batch4)
    print(f"  output shape: {out4.shape}")

    R4, _ = torch.linalg.qr(torch.randn(4, 4))
    if torch.det(R4) < 0:
        R4[:, 0] *= -1
    out4_rot = model4([pc @ R4.T for pc in batch4])
    diff4 = (out4 - out4_rot).abs().max().item()
    print(f"  SE(4) invariance (max diff): {diff4:.2e}")

    # -----------------------------------------------------------------------
    # 4.  Parameter counts
    # -----------------------------------------------------------------------
    print("\n=== Parameter counts ===")
    print(f"  PointNet3D:                   {sum(p.numel() for p in pnet.parameters()):>10,}")
    print(f"  TensorFieldNetwork (n=3):     {sum(p.numel() for p in tfn.parameters()):>10,}")
    print(f"  GTTensorFieldNetwork (n=4):   {sum(p.numel() for p in model4.parameters()):>10,}")
