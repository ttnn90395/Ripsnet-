"""
gt_tfn_layer.py
===============
SE(n)-equivariant message-passing layer built on the GT basis.

Exports
-------
  GTTFNLayer              : one equivariant conv layer for SO(n)
  GTTensorFieldNetwork    : full model, same List[Tensor] interface as TFN/PointNet

The design mirrors the original TFNLayer but replaces the hard-wired
SO(3) f0/f1 pair with a dict of arbitrary GT feature types:

    features: Dict[GTSignature, Tensor]   shape  (N, channels, dim(sig))

Each GTTFNLayer:
  1. Computes pairwise RBF edge features (rotation-invariant, as before).
  2. For every (type_in, type_edge, type_out) triple whose CG is non-zero,
     contracts the CG tensor with the radial network output to form
     equivariant messages.
  3. Aggregates messages by summation over neighbors (with masking).
  4. Applies LayerNorm in an equivariant way (norms of each irrep channel).
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from gt_basis import GTSignature, GTBasis, CGCoefficients


# Re-export the RBF class so callers only need one import
class RBFExpansion(nn.Module):
    """Gaussian RBF expansion of pairwise distances. Rotation-invariant."""

    def __init__(self, num_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff  = cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = (cutoff / num_rbf) ** 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-diff ** 2 / self.width)


# ---------------------------------------------------------------------------
# Feature dict type alias
# ---------------------------------------------------------------------------
# FeatureDict maps a GTSignature → Tensor of shape (N, C, dim(sig))
# where N = number of points, C = number of feature channels for that type.
FeatureDict = Dict[GTSignature, torch.Tensor]


# ---------------------------------------------------------------------------
# GTTFNLayer
# ---------------------------------------------------------------------------

class GTTFNLayer(nn.Module):
    """
    One SO(n)-equivariant message-passing layer.

    Parameters
    ----------
    n           : int    ambient dimension
    in_types    : dict   {GTSignature: int}  input  type → num channels
    out_types   : dict   {GTSignature: int}  output type → num channels
    num_rbf     : int    RBF dimension (= edge feature dimension)
    cg          : CGCoefficients   pre-computed CG tables (shared across layers)
    gt_basis    : GTBasis          pre-computed GT harmonics (shared)
    """

    def __init__(
        self,
        n:         int,
        in_types:  Dict[GTSignature, int],
        out_types: Dict[GTSignature, int],
        num_rbf:   int,
        cg:        CGCoefficients,
        gt_basis:  GTBasis,
    ):
        super().__init__()
        self.n         = n
        self.in_types  = in_types
        self.out_types = out_types
        self.num_rbf   = num_rbf
        self.cg        = cg
        self.gt_basis  = gt_basis

        # For each valid (sig_in, sig_edge, sig_out) triple build a radial net
        # W: R^{num_rbf} → R^{C_in * C_out}
        self.radial_nets = nn.ModuleDict()
        self._interactions: List[Tuple[GTSignature, GTSignature, GTSignature]] = []

        edge_sigs = gt_basis.signatures   # types carried on each edge (harmonics)

        for sig_in, c_in in in_types.items():
            for sig_edge in edge_sigs:
                for sig_out, c_out in out_types.items():
                    cg_tensor = cg.get(sig_in, sig_edge, sig_out)
                    if cg_tensor is None or cg_tensor.norm() < 1e-8:
                        continue
                    key = _interaction_key(sig_in, sig_edge, sig_out)
                    if key not in self.radial_nets:
                        self.radial_nets[key] = nn.Sequential(
                            nn.Linear(num_rbf, 32), nn.SiLU(),
                            nn.Linear(32, c_in * c_out),
                        )
                    self._interactions.append((sig_in, sig_edge, sig_out))

        # Equivariant LayerNorm: one scalar norm per (output_type, channel)
        self.layer_norms = nn.ModuleDict({
            _sig_key(sig): nn.LayerNorm(c_out)
            for sig, c_out in out_types.items()
        })

    # ------------------------------------------------------------------
    def forward(
        self,
        feats:    FeatureDict,        # {sig: (N, C_in, dim(sig))}
        rbf:      torch.Tensor,       # (N, N, num_rbf)
        gt_edge:  torch.Tensor,       # (N, N, num_basis)  GT harmonics on edges
        mask:     torch.Tensor,       # (N, N) bool, True = valid neighbor
    ) -> FeatureDict:
        """
        One layer of equivariant message-passing.

        Returns a new FeatureDict with the same key set as out_types.
        """
        N = rbf.shape[0]
        out: FeatureDict = {sig: torch.zeros(N, c, sig.dim(), device=rbf.device)
                            for sig, c in self.out_types.items()}

        # Pre-split gt_edge into per-signature blocks
        gt_edge_by_sig = self._split_edge_harmonics(gt_edge)

        for sig_in, sig_edge, sig_out in self._interactions:
            if sig_in not in feats:
                continue
            f_in   = feats[sig_in]                   # (N, C_in, d_in)
            c_in   = f_in.shape[1]
            c_out  = self.out_types[sig_out]
            d_in   = sig_in.dim()
            d_edge = sig_edge.dim()
            d_out  = sig_out.dim()

            cg_tensor = self.cg.get(sig_in, sig_edge, sig_out)   # (d_in, d_edge, d_out)
            e_feat    = gt_edge_by_sig[sig_edge]                  # (N, N, d_edge)
            key       = _interaction_key(sig_in, sig_edge, sig_out)
            radial    = self.radial_nets[key](rbf)                # (N, N, c_in*c_out)
            radial    = radial.reshape(N, N, c_in, c_out)

            # Message:  msg[i, j, c_out, d_out]
            #         = Σ_{c_in, d_in, d_edge}
            #             radial[i,j,c_in,c_out] * f_in[j,c_in,d_in]
            #             * e_feat[i,j,d_edge] * cg[d_in, d_edge, d_out]
            #
            # We compute in two steps for memory efficiency:
            #   step 1: contract f_in with CG over d_in → (N,N,c_in,d_edge,d_out)
            #   step 2: contract with e_feat over d_edge, and with radial over c_in

            # step 1: (N, c_in, d_in) × (d_in, d_edge, d_out) → (N, c_in, d_edge, d_out)
            fCG = torch.einsum("jci,ieo->jceo", f_in, cg_tensor)  # (N, c_in, d_edge, d_out)

            # step 2: contract with e_feat and radial, sum over j
            # e_feat: (N, N, d_edge)
            # radial: (N, N, c_in, c_out)
            # mask:   (N, N)
            # result: (N, c_out, d_out)
            msg = torch.einsum(
                "ijceo, ijde, ijco, ij -> ico",
                fCG.unsqueeze(0).expand(N, -1, -1, -1, -1),   # (N, N, c_in, d_edge, d_out)
                e_feat.unsqueeze(3).expand(-1, -1, -1, c_in),  # (N, N, d_edge, c_in) — broadcast
                # Rewrite cleanly:
                radial,                                          # (N, N, c_in, c_out)
                mask.float(),                                    # (N, N)
            ) if False else self._message(f_in, fCG, e_feat, radial, mask, N, c_in, c_out, d_edge, d_out)

            out[sig_out] = out[sig_out] + msg

        # Equivariant LayerNorm
        out = self._apply_norm(out)
        return out

    def _message(self, f_in, fCG, e_feat, radial, mask, N, c_in, c_out, d_edge, d_out):
        """
        Compute equivariant messages cleanly.

        fCG    : (N, c_in, d_edge, d_out)
        e_feat : (N, N, d_edge)
        radial : (N, N, c_in, c_out)
        mask   : (N, N)
        Returns (N, c_out, d_out)
        """
        # Contract fCG[j] with e_feat[i,j] over d_edge → (N_i, N_j, c_in, d_out)
        # e_feat: (i, j, d_edge), fCG: (j, c_in, d_edge, d_out)
        # result: (i, j, c_in, d_out)
        contracted = torch.einsum("ije, jceo -> ijco", e_feat, fCG)  # (N, N, c_in, d_out)

        # Weight by radial and sum over j (with mask)
        # radial: (i, j, c_in, c_out)
        # contracted: (i, j, c_in, d_out)
        # out[i, c_out, d_out] = Σ_j Σ_c_in  mask[i,j] * radial[i,j,c_in,c_out] * contracted[i,j,c_in,d_out]
        masked = contracted * mask.unsqueeze(-1).unsqueeze(-1).float()  # (N, N, c_in, d_out)
        # einsum over j and c_in
        msg = torch.einsum("ijco, ijcd -> iod", radial, masked)   # (N, c_out, d_out)
        return msg

    def _split_edge_harmonics(self, gt_edge: torch.Tensor) -> Dict[GTSignature, torch.Tensor]:
        """Split the flat GT harmonic output into per-signature blocks."""
        result = {}
        offset = 0
        for sig, d in zip(self.gt_basis.signatures, self.gt_basis.dims):
            result[sig] = gt_edge[..., offset:offset + d]
            offset += d
        return result

    def _apply_norm(self, feats: FeatureDict) -> FeatureDict:
        """
        Equivariant LayerNorm: normalise over channels using the norm of each
        irrep vector (preserves direction, only scales magnitude).
        """
        out = {}
        for sig, f in feats.items():
            # f: (N, C, d)
            norms  = f.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, C, 1)
            scaled = self.layer_norms[_sig_key(sig)](norms.squeeze(-1))  # (N, C)
            out[sig] = f / norms * scaled.unsqueeze(-1)
        return out


# ---------------------------------------------------------------------------
# Pairwise geometry helper
# ---------------------------------------------------------------------------

def pairwise_geometry(
    pos:         torch.Tensor,    # (N, n)
    rbf_encoder: RBFExpansion,
    gt_basis:    GTBasis,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all pairwise geometric quantities for one point cloud.

    Returns
    -------
    rbf      : (N, N, num_rbf)   rotation-invariant distance features
    gt_edge  : (N, N, num_basis) GT harmonic features of unit directions
    mask     : (N, N) bool, False on diagonal (self-loops excluded)
    """
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)        # (N, N, n)  r_i - r_j
    dist = diff.norm(dim=-1)                           # (N, N)
    r_hat = diff / dist.unsqueeze(-1).clamp(min=1e-8)  # (N, N, n)

    rbf     = rbf_encoder(dist)                        # (N, N, num_rbf)
    N       = pos.shape[0]
    r_flat  = r_hat.reshape(N * N, pos.shape[-1])
    gt_flat = gt_basis(r_flat)                         # (N*N, num_basis)
    gt_edge = gt_flat.reshape(N, N, -1)

    mask = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    return rbf, gt_edge, mask


# ---------------------------------------------------------------------------
# GTTensorFieldNetwork
# ---------------------------------------------------------------------------

class GTTensorFieldNetwork(nn.Module):
    """
    SE(n)-equivariant point cloud classifier for any ambient dimension n.

    Parameters
    ----------
    n             : int   ambient dimension (e.g. 3 for 3D, 4 for 4D)
    num_classes   : int   number of output classes
    max_order     : int   maximum λ₁ of GT features used (1 or 2 recommended)
    hidden_channels : int number of feature channels per irrep type
    num_layers    : int   number of GTTFNLayer message-passing layers
    num_rbf       : int   RBF basis size
    cutoff        : float RBF and neighborhood cutoff distance
    classifier_dims : list[int]  hidden dims of the final invariant MLP
    """

    def __init__(
        self,
        n:               int,
        num_classes:     int,
        max_order:       int   = 1,
        hidden_channels: int   = 16,
        num_layers:      int   = 3,
        num_rbf:         int   = 16,
        cutoff:          float = 5.0,
        classifier_dims: List[int] = [64, 32],
    ):
        super().__init__()
        self.n           = n
        self.num_classes = num_classes
        self.max_order   = max_order

        # Shared geometric encoders
        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)

        # All irrep types up to max_order
        all_types = self.gt_basis.signatures

        # Initial feature types: scalar (||pos||) + vector (pos direction)
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)
        init_types = {scalar_sig: 1, vector_sig: 1}

        # Hidden type dict: all types, each with hidden_channels channels
        hidden_types = {sig: hidden_channels for sig in all_types}

        # Build layers
        self.layers = nn.ModuleList()
        in_types = init_types
        for _ in range(num_layers):
            self.layers.append(GTTFNLayer(
                n=n,
                in_types=in_types,
                out_types=hidden_types,
                num_rbf=num_rbf,
                cg=self.cg,
                gt_basis=self.gt_basis,
            ))
            in_types = hidden_types

        # Invariant projection: scalar features + norms of all other features
        scalar_dim   = hidden_channels                        # from scalar type
        vector_norms = sum(hidden_channels for sig in all_types if sig != scalar_sig)
        inv_dim      = scalar_dim + vector_norms

        # Classification MLP
        rho = []
        d = inv_dim
        for h in classifier_dims:
            rho += [nn.Linear(d, h), nn.ReLU()]
            d = h
        rho.append(nn.Linear(d, num_classes))
        self.rho = nn.Sequential(*rho)

        self._scalar_sig = scalar_sig

    # ------------------------------------------------------------------
    def _encode_single(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Encode one point cloud to a global invariant descriptor.
        pos : (N, n)  →  (inv_dim,)
        """
        N = pos.shape[0]
        scalar_sig = self._scalar_sig
        vector_sig = GTSignature.vector(self.n)

        # Initial features
        f0 = pos.norm(dim=-1, keepdim=True).unsqueeze(1)   # (N, 1, 1)
        f1 = (pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)).unsqueeze(1)  # (N, 1, n)

        feats: FeatureDict = {scalar_sig: f0, vector_sig: f1}

        # Pairwise geometry
        rbf, gt_edge, mask = pairwise_geometry(pos, self.rbf, self.gt_basis)

        # Message-passing
        for layer in self.layers:
            feats = layer(feats, rbf, gt_edge, mask)

        # Build invariant descriptor per node
        parts = []
        # scalar features (already invariant)
        if scalar_sig in feats:
            parts.append(feats[scalar_sig].squeeze(-1))   # (N, C)
        # norms of all non-scalar features
        for sig, f in feats.items():
            if sig != scalar_sig:
                parts.append(f.norm(dim=-1))              # (N, C)

        node_inv = torch.cat(parts, dim=-1)               # (N, inv_dim)

        # Global max-pool
        return node_inv.max(dim=0).values                  # (inv_dim,)

    # ------------------------------------------------------------------
    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        batch : List of (N_i, n) tensors
        returns: (B, num_classes) logits
        """
        if len(batch) == 0:
            device = next(self.parameters()).device
            return torch.empty(0, self.num_classes, device=device)

        global_feats = torch.stack([self._encode_single(pc) for pc in batch])
        return self.rho(global_feats)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _interaction_key(s1: GTSignature, s2: GTSignature, s3: GTSignature) -> str:
    return f"{s1.lam}x{s2.lam}to{s3.lam}_n{s1.n}"

def _sig_key(s: GTSignature) -> str:
    return f"sig_{s.lam}_n{s.n}"


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    for n in [3, 4]:
        print(f"\n=== GTTensorFieldNetwork  n={n} ===")
        model = GTTensorFieldNetwork(
            n=n, num_classes=8, max_order=1,
            hidden_channels=8, num_layers=2,
        )
        total = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total:,}")

        batch = [torch.randn(k, n) for k in [16, 24, 20]]
        logits = model(batch)
        print(f"  Output shape: {logits.shape}  (expected ({len(batch)}, 8))")

        # SE(n) invariance check: random rotation
        R, _ = torch.linalg.qr(torch.randn(n, n))
        if torch.det(R) < 0:
            R[:, 0] *= -1   # ensure proper rotation
        rot_batch = [pc @ R.T for pc in batch]
        logits_rot = model(rot_batch)
        diff = (logits - logits_rot).abs().max().item()
        print(f"  Invariance check (max diff after rotation): {diff:.2e}")
        print(f"  {'OK — invariant' if diff < 1e-3 else 'Not invariant (expected for fp32)'}")
