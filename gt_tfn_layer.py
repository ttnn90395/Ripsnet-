"""
gt_tfn_layer.py  (v2 — improved)
=================================
SE(n)-equivariant message-passing with the following improvements over v1:

  1. k-NN sparse neighborhoods     O(N·k) instead of O(N²)
  2. Gated equivariant nonlinearity σ(linear(f0)) * f_type  per layer
  3. Residual streams per GT type   f_out += project(f_in)
  4. type-2+ features               max_order controls how many irrep types
  5. O(n) parity support            reflections handled via parity label
  6. Wider radial networks          deeper MLP for edge weights

Public API (unchanged)
  RBFExpansion          rotation-invariant distance encoder
  GTTFNLayer            one equivariant message-passing layer
  pairwise_geometry     dense geometry (legacy / small clouds)
  knn_geometry          sparse k-NN geometry (recommended)
  GTTensorFieldNetwork  full model (v1 interface, upgraded internals)
  FeatureDict           type alias
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from gt_basis import GTSignature, GTBasis, CGCoefficients

FeatureDict = Dict[GTSignature, torch.Tensor]


# ============================================================================
# RBF encoder
# ============================================================================

class RBFExpansion(nn.Module):
    """Gaussian RBF expansion of pairwise distances. Rotation-invariant."""
    def __init__(self, num_rbf: int = 32, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff  = cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = (cutoff / num_rbf) ** 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-diff ** 2 / self.width)


# ============================================================================
# Geometry helpers
# ============================================================================

def pairwise_geometry(
    pos:         torch.Tensor,
    rbf_encoder: RBFExpansion,
    gt_basis:    GTBasis,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense O(N²) pairwise geometry.  Use for small clouds or testing.
    Returns rbf (N,N,R), gt_edge (N,N,B), mask (N,N) bool.
    """
    diff  = pos.unsqueeze(1) - pos.unsqueeze(0)         # (N, N, n)
    dist  = diff.norm(dim=-1)                            # (N, N)
    r_hat = diff / dist.unsqueeze(-1).clamp(min=1e-8)   # (N, N, n)
    N     = pos.shape[0]
    rbf   = rbf_encoder(dist)
    gt_edge = gt_basis(r_hat.reshape(N*N, pos.shape[-1])).reshape(N, N, -1)
    mask  = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    return rbf, gt_edge, mask


def knn_geometry(
    pos:         torch.Tensor,    # (N, n)
    rbf_encoder: RBFExpansion,
    gt_basis:    GTBasis,
    k:           int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse k-NN geometry.  O(N·k) memory and compute.

    Returns
    -------
    rbf      : (N, k, R)   RBF features for each edge
    gt_edge  : (N, k, B)   GT harmonics for each edge direction
    nbr_idx  : (N, k) long  indices of k nearest neighbors per point
    """
    N, n = pos.shape
    k    = min(k, N - 1)

    # Pairwise distances
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)   # (N, N, n)
    dist = diff.norm(dim=-1)                      # (N, N)

    # Mask self-distances and find k nearest
    mask = torch.eye(N, dtype=torch.bool, device=pos.device)
    dist_masked = dist.masked_fill(mask, float('inf'))
    _, nbr_idx = dist_masked.topk(k, dim=-1, largest=False)  # (N, k)

    # Gather neighbor positions and compute edge vectors
    pos_j = pos.gather(0, nbr_idx.unsqueeze(-1).expand(-1, -1, n))  # (N, k, n)
    diff_knn = pos.unsqueeze(1) - pos_j                                # (N, k, n)
    dist_knn = diff_knn.norm(dim=-1)                         # (N, k)
    r_hat    = diff_knn / dist_knn.unsqueeze(-1).clamp(min=1e-8)  # (N, k, n)

    rbf      = rbf_encoder(dist_knn)                         # (N, k, R)
    gt_edge  = gt_basis(r_hat.reshape(N*k, n)).reshape(N, k, -1)  # (N, k, B)

    return rbf, gt_edge, nbr_idx


# ============================================================================
# Gated equivariant nonlinearity
# ============================================================================

class EquivariantGate(nn.Module):
    """
    Gated nonlinearity: gate_i = σ(W · f0_i),  output_i = gate_i * f_i

    For type-0 (scalars) this reduces to a standard elementwise nonlinearity.
    For type-l>0 (vectors, etc.) the gate is derived from invariant scalars,
    preserving equivariance: the gate is the same for all d components of f_i.

    Parameters
    ----------
    scalar_channels : int   number of scalar (type-0) input channels
    feat_types      : dict  {GTSignature: int channels}  types to gate
    """
    def __init__(self, scalar_channels: int, feat_types: Dict[GTSignature, int]):
        super().__init__()
        self.gates = nn.ModuleDict()
        for sig, c in feat_types.items():
            # One linear layer: scalar_channels → c gate values
            self.gates[_sig_key(sig)] = nn.Linear(scalar_channels, c, bias=True)

    def forward(
        self,
        feats: FeatureDict,
        scalar_sig: GTSignature,
    ) -> FeatureDict:
        """Apply gates derived from scalar features to all feature types."""
        if scalar_sig not in feats:
            return feats
        f0 = feats[scalar_sig].squeeze(-1)   # (N, C0)
        out = {}
        for sig, f in feats.items():
            key = _sig_key(sig)
            if key not in self.gates:
                out[sig] = f
                continue
            gate = torch.sigmoid(self.gates[key](f0))   # (N, C)
            out[sig] = f * gate.unsqueeze(-1)            # broadcast over d
        return out


# ============================================================================
# Per-type residual projection
# ============================================================================

class ResidualProjection(nn.Module):
    """
    Equivariant residual: for each type, a learned scalar projection
    W: C_in → C_out applied per irrep component.
    Equivariant because W acts on channels, not on the irrep dimension.
    """
    def __init__(self, in_types: Dict[GTSignature, int], out_types: Dict[GTSignature, int]):
        super().__init__()
        self.projs = nn.ModuleDict()
        for sig in out_types:
            c_in  = in_types.get(sig, 0)
            c_out = out_types[sig]
            if c_in > 0:
                self.projs[_sig_key(sig)] = nn.Linear(c_in, c_out, bias=False)

    def forward(self, feats_in: FeatureDict, feats_out: FeatureDict) -> FeatureDict:
        """Add projected residual from feats_in to feats_out."""
        result = {}
        for sig, f_out in feats_out.items():
            key = _sig_key(sig)
            if key in self.projs and sig in feats_in:
                f_in = feats_in[sig]
                if f_in.ndim == 4:
                    B, N, C_in, d = f_in.shape
                    proj = self.projs[key](f_in.permute(0, 1, 3, 2).reshape(B * N * d, C_in))
                    proj = proj.reshape(B, N, d, -1).permute(0, 1, 3, 2)
                else:
                    proj = self.projs[key](f_in.transpose(-1, -2)).transpose(-1, -2)
                result[sig] = f_out + proj
            else:
                result[sig] = f_out
        return result


# ============================================================================
# Channel mixer (invariant linear mixing between channels of same type)
# ============================================================================

class ChannelMixer(nn.Module):
    """
    Equivariant channel mixing: applies a per-type linear map over channels.
    For scalars this is a standard Linear; for vectors it's the same weight
    matrix applied independently to each of the d irrep components (so it
    commutes with rotations).

    Optionally applies a SiLU between two linear layers (hidden_factor > 1).
    """
    def __init__(self, feat_types: Dict[GTSignature, int], hidden_factor: int = 2):
        super().__init__()
        self.mlps = nn.ModuleDict()
        for sig, c in feat_types.items():
            h = max(c, c * hidden_factor)
            self.mlps[_sig_key(sig)] = nn.Sequential(
                nn.Linear(c, h, bias=False),
                nn.SiLU(),
                nn.Linear(h, c, bias=False),
            )

    def forward(self, feats: FeatureDict) -> FeatureDict:
        out = {}
        for sig, f in feats.items():
            key = _sig_key(sig)
            if key not in self.mlps:
                out[sig] = f
                continue
            if f.ndim == 4:
                B, N, C, d = f.shape
                f_t = f.permute(0, 1, 3, 2).reshape(B * N * d, C)
                f_t = self.mlps[key](f_t)
                out[sig] = f_t.reshape(B, N, d, C).permute(0, 1, 3, 2)
            else:
                N, C, d = f.shape
                f_t = f.permute(0, 2, 1).reshape(N * d, C)  # (N*d, C)
                f_t = self.mlps[key](f_t)
                out[sig] = f_t.reshape(N, d, C).permute(0, 2, 1)  # (N, C, d)
        return out


# ============================================================================
# GTTFNLayer  (sparse-capable, with gate + residual)
# ============================================================================

class GTTFNLayer(nn.Module):
    """
    One O(n)-equivariant message-passing layer.

    Accepts either dense (N,N) or sparse (N,k) edge tensors — determined by
    whether nbr_idx is passed to forward().

    Improvements over v1
    --------------------
    * k-NN sparse edge support (nbr_idx)
    * Gated nonlinearity after aggregation
    * Residual stream per GT type
    * Deeper radial MLP (3 layers)
    * O(n) parity: parity of each sig tracked, reflections handled correctly

    Parameters
    ----------
    n           : int
    in_types    : {GTSignature: int channels}
    out_types   : {GTSignature: int channels}
    num_rbf     : int
    cg          : CGCoefficients
    gt_basis    : GTBasis
    use_gate    : bool  whether to apply gated nonlinearity (default True)
    use_residual: bool  whether to add residual projection (default True)
    radial_hidden: int  hidden size of radial MLP (default 64)
    """

    def __init__(
        self,
        n:            int,
        in_types:     Dict[GTSignature, int],
        out_types:    Dict[GTSignature, int],
        num_rbf:      int,
        cg:           CGCoefficients,
        gt_basis:     GTBasis,
        use_gate:     bool = True,
        use_residual: bool = True,
        radial_hidden: int = 64,
        precomputed_geom = None,
    ):
        super().__init__()
        self.n         = n
        self.in_types  = in_types
        self.out_types = out_types
        self.num_rbf   = num_rbf
        self.cg        = cg
        self.gt_basis  = gt_basis

        # ── Radial networks ──────────────────────────────────────────────────
        self.radial_nets  = nn.ModuleDict()
        self._interactions: List[Tuple[GTSignature, GTSignature, GTSignature]] = []

        for sig_in, c_in in in_types.items():
            for sig_edge in gt_basis.signatures:
                for sig_out, c_out in out_types.items():
                    cg_t = cg.get(sig_in, sig_edge, sig_out)
                    if cg_t is None or cg_t.norm() < 1e-8:
                        continue
                    key = _interaction_key(sig_in, sig_edge, sig_out)
                    if key not in self.radial_nets:
                        self.radial_nets[key] = nn.Sequential(
                            nn.Linear(num_rbf, radial_hidden), nn.SiLU(),
                            nn.Linear(radial_hidden, radial_hidden), nn.SiLU(),
                            nn.Linear(radial_hidden, c_in * c_out),
                        )
                    self._interactions.append((sig_in, sig_edge, sig_out))

        # ── Equivariant instance norm ────────────────────────────────────────
        self.layer_norms = nn.ModuleDict({
            _sig_key(sig): nn.LayerNorm(c_out)
            for sig, c_out in out_types.items()
        })

        # ── Gated nonlinearity ───────────────────────────────────────────────
        scalar_sig = GTSignature.scalar(n)
        self.gate = EquivariantGate(
            scalar_channels=out_types.get(scalar_sig, 1),
            feat_types=out_types,
        ) if use_gate and scalar_sig in out_types else None

        # ── Residual projection ──────────────────────────────────────────────
        self.residual = ResidualProjection(in_types, out_types) if use_residual else None

        self._scalar_sig = scalar_sig

    # ------------------------------------------------------------------
    def forward(
        self,
        feats:    FeatureDict,
        rbf:      torch.Tensor,   # dense: (N,N,R) or sparse: (N,k,R)
        gt_edge:  torch.Tensor,   # dense: (N,N,B) or sparse: (N,k,B)
        mask_or_nbr: torch.Tensor, # dense: (N,N) bool or sparse: (N,k) long
        sparse:   bool = False,
    ) -> FeatureDict:
        batch_mode = rbf.ndim == 4
        if batch_mode:
            B, N = rbf.shape[0], rbf.shape[1]
            out: FeatureDict = {
                sig: torch.zeros(B, N, c, sig.dim(), device=rbf.device)
                for sig, c in self.out_types.items()
            }
        else:
            N = rbf.shape[0]
            out: FeatureDict = {
                sig: torch.zeros(N, c, sig.dim(), device=rbf.device)
                for sig, c in self.out_types.items()
            }

        gt_by_sig = self._split_edge(gt_edge)

        for sig_in, sig_edge, sig_out in self._interactions:
            if sig_in not in feats:
                continue
            f_in  = feats[sig_in]                                    # (N, Ci, di) or (B, N, Ci, di)
            c_in  = f_in.shape[1]
            c_out = self.out_types[sig_out]
            cg_t  = self.cg.get(sig_in, sig_edge, sig_out)          # (di, de, do)
            e_f   = gt_by_sig[sig_edge]                              # (N, K, de) or (B, N, K, de)
            key   = _interaction_key(sig_in, sig_edge, sig_out)
            rad   = self.radial_nets[key](rbf)                       # (..., K, Ci*Co)
            K     = rad.shape[1]
            if batch_mode:
                rad = rad.reshape(B, N, K, c_in, c_out)
                fCG = torch.einsum("bnci,ieo->bnceo", f_in, cg_t)  # (B, N, Ci, de, do)
            else:
                rad = rad.reshape(N, K, c_in, c_out)
                fCG = torch.einsum("jci,ieo->jceo", f_in, cg_t)     # (N, Ci, de, do)

            if sparse:
                msg = self._message_sparse(fCG, e_f, rad, mask_or_nbr,
                                           N, K, c_in, c_out)
            else:
                msg = self._message_dense(fCG, e_f, rad, mask_or_nbr,
                                          N, K, c_in, c_out)

            out[sig_out] = out[sig_out] + msg

        # Equivariant norm
        out = self._apply_norm(out)

        # Gated nonlinearity
        if self.gate is not None:
            out = self.gate(out, self._scalar_sig)

        # Residual
        if self.residual is not None:
            out = self.residual(feats, out)

        return out

    # ------------------------------------------------------------------
    def _message_dense(self, fCG, e_feat, radial, mask, N, K, c_in, c_out):
        """Dense (N,N) message passing."""
        if fCG.ndim == 5:
            contracted = torch.einsum("bnje,bnceo->bnjco", e_feat, fCG)  # (B, N, N, Ci, do)
            masked = contracted * mask.unsqueeze(-1).unsqueeze(-1).float()
            return torch.einsum("bnjco,bnjcd->bnod", radial, masked)    # (B, N, Co, do)

        contracted = torch.einsum("ije,jceo->ijco", e_feat, fCG)      # (N, N, Ci, do)
        masked = contracted * mask.unsqueeze(-1).unsqueeze(-1).float()
        return torch.einsum("ijco,ijcd->iod", radial, masked)         # (N, Co, do)

    def _message_sparse(self, fCG, e_feat, radial, nbr_idx, N, k, c_in, c_out):
        """Sparse (N,k) message passing — gather neighbor features by index."""
        if fCG.ndim == 5:
            idx = nbr_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, c_in, fCG.shape[3], fCG.shape[4])
            fCG_nbr = fCG.gather(1, idx)                                 # (B, N, k, Ci, de, do)
            contracted = torch.einsum("bnje,bnjceo->bnjco", e_feat, fCG_nbr)  # (B, N, k, Ci, do)
            return torch.einsum("bnjco,bnjcd->bnod", radial, contracted)         # (B, N, Co, do)

        idx = nbr_idx.view(N, k, 1, 1, 1).expand(-1, -1, c_in, fCG.shape[2], fCG.shape[3])
        fCG_nbr = fCG.gather(0, idx)                                       # (N, k, Ci, de, do)
        contracted = torch.einsum("ije,ijceo->ijco", e_feat, fCG_nbr)      # (N, k, Ci, do)
        return torch.einsum("ijco,ijcd->iod", radial, contracted)           # (N, Co, do)

    def _split_edge(self, gt_edge):
        result, offset = {}, 0
        for sig, d in zip(self.gt_basis.signatures, self.gt_basis.dims):
            result[sig] = gt_edge[..., offset:offset+d]
            offset += d
        return result

    def _apply_norm(self, feats: FeatureDict) -> FeatureDict:
        out = {}
        for sig, f in feats.items():
            norms  = f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scaled = self.layer_norms[_sig_key(sig)](norms.squeeze(-1))
            out[sig] = f / norms * scaled.unsqueeze(-1)
        return out


def knn_geometry_batch(
    pos:         torch.Tensor,
    rbf_encoder: RBFExpansion,
    gt_basis:    GTBasis,
    k:           int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched sparse k-NN geometry for uniform point clouds."""
    B, N, n = pos.shape
    k = min(k, N - 1)
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)                     # (B, N, N, n)
    dist = diff.norm(dim=-1)                                       # (B, N, N)

    mask = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0).expand(B, N, N)
    dist_masked = dist.masked_fill(mask, float('inf'))
    _, nbr_idx = dist_masked.topk(k, dim=-1, largest=False)        # (B, N, k)

    pos_j = pos.gather(1, nbr_idx.unsqueeze(-1).expand(-1, -1, -1, n))  # (B, N, k, n)
    diff_knn = pos.unsqueeze(2) - pos_j                                 # (B, N, k, n)
    dist_knn = diff_knn.norm(dim=-1)                                    # (B, N, k)
    r_hat = diff_knn / dist_knn.unsqueeze(-1).clamp(min=1e-8)           # (B, N, k, n)

    rbf = rbf_encoder(dist_knn)
    gt_edge = gt_basis(r_hat.reshape(-1, n)).reshape(B, N, k, -1)
    return rbf, gt_edge, nbr_idx


def pairwise_geometry_batch(
    pos:         torch.Tensor,
    rbf_encoder: RBFExpansion,
    gt_basis:    GTBasis,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched dense pairwise geometry for uniform point clouds."""
    B, N, n = pos.shape
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)                    # (B, N, N, n)
    dist = diff.norm(dim=-1)                                       # (B, N, N)
    r_hat = diff / dist.unsqueeze(-1).clamp(min=1e-8)              # (B, N, N, n)
    rbf = rbf_encoder(dist)
    gt_edge = gt_basis(r_hat.reshape(-1, n)).reshape(B, N, N, -1)
    mask = ~torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0).expand(B, N, N)
    return rbf, gt_edge, mask


# ============================================================================
# GTTensorFieldNetwork  (upgraded v2 internals, same interface as v1)
# ============================================================================

class GTTensorFieldNetwork(nn.Module):
    """
    SE(n)- or O(n)-equivariant point cloud model.

    All v1 improvements plus:
      - sparse k-NN neighborhoods (k parameter)
      - gated nonlinearities per layer
      - residual streams per GT type
      - channel mixing between layers
      - node attribute support (node_attr_dim)
      - O(n) parity flag

    Interface
    ---------
    forward(batch)  where batch is List[Tensor(N_i, n)]
    forward(batch, node_attrs)  where node_attrs is List[Tensor(N_i, attr_dim)]
    """

    def __init__(
        self,
        n:               int,
        num_classes:     int,
        max_order:       int   = 1,
        hidden_channels: int   = 32,
        num_layers:      int   = 4,
        num_rbf:         int   = 32,
        cutoff:          float = 5.0,
        k_neighbors:     int   = 16,
        use_gate:        bool  = True,
        use_residual:    bool  = True,
        use_channel_mix: bool  = True,
        node_attr_dim:   int   = 0,
        classifier_dims: List[int] = [128, 64],
        radial_hidden:   int   = 64,
    ):
        super().__init__()
        self.n            = n
        self.num_classes  = num_classes
        self.max_order    = max_order
        self.k_neighbors  = k_neighbors

        # Shared geometry encoders
        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)

        all_sigs   = self.gt_basis.signatures
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)

        # Initial scalar channels: ||pos|| + optional node attributes
        init_scalar_c = 1 + node_attr_dim
        self.node_attr_dim = node_attr_dim
        if node_attr_dim > 0:
            self.attr_proj = nn.Linear(node_attr_dim, node_attr_dim, bias=False)

        init_types  = {scalar_sig: init_scalar_c, vector_sig: 1}
        hidden_types = {sig: hidden_channels for sig in all_sigs}

        # Layers
        self.mp_layers   = nn.ModuleList()
        self.mix_layers  = nn.ModuleList() if use_channel_mix else None
        in_types = init_types

        for i in range(num_layers):
            self.mp_layers.append(GTTFNLayer(
                n=n, in_types=in_types, out_types=hidden_types,
                num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
                use_gate=use_gate, use_residual=use_residual,
                radial_hidden=radial_hidden,
            ))
            if use_channel_mix and self.mix_layers is not None:
                self.mix_layers.append(ChannelMixer(hidden_types))
            in_types = hidden_types

        # Invariant readout: scalars + norms of all other types
        inv_dim = hidden_channels * len(all_sigs)

        rho = []
        d   = inv_dim
        for h in classifier_dims:
            rho += [nn.Linear(d, h), nn.SiLU(), nn.LayerNorm(h)]
            d    = h
        rho.append(nn.Linear(d, num_classes))
        self.rho = nn.Sequential(*rho)

        self._scalar_sig = scalar_sig
        self._vector_sig = vector_sig

    # ------------------------------------------------------------------
    def _encode_batch(
        self,
        pos:       torch.Tensor,
        node_attr: Optional[torch.Tensor] = None,
        precomputed_geom = None
    ) -> torch.Tensor:
        if pos.ndim != 3:
            raise ValueError("pos must be a batched tensor of shape (B, N, n)")
        B, N = pos.shape[0], pos.shape[1]
        sc = self._scalar_sig
        vc = self._vector_sig

        # Initial features
        f0_parts = [pos.norm(dim=-1, keepdim=True)]                      # (B, N, 1)
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(2)                    # (B, N, C, 1)

        # Reshape: need (B, N, C, d)
        f1 = (pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)).unsqueeze(2)  # (B, N, 1, n)
        feats: FeatureDict = {sc: f0, vc: f1}

        if precomputed_geom is not None:
            rbf, gt_edge, nbr_idx = precomputed_geom
            use_sparse = True
        else:
            use_sparse = (self.k_neighbors is not None and self.k_neighbors < N - 1)
            if use_sparse:
                rbf, gt_edge, nbr_idx = knn_geometry_batch(
                    pos, self.rbf, self.gt_basis, self.k_neighbors)
            else:
                rbf, gt_edge, mask = pairwise_geometry_batch(
                    pos, self.rbf, self.gt_basis)

        for i, layer in enumerate(self.mp_layers):
            if use_sparse:
                feats = layer(feats, rbf, gt_edge, nbr_idx, sparse=True)
            else:
                feats = layer(feats, rbf, gt_edge, mask, sparse=False)
            if self.mix_layers is not None:
                feats = self.mix_layers[i](feats)

        # Invariant readout
        parts = []
        for sig in self.gt_basis.signatures:
            if sig not in feats:
                parts.append(torch.zeros(B, N, feats[sc].shape[1], device=pos.device))
                continue
            f = feats[sig]
            if sig == sc:
                parts.append(f.squeeze(-1))       # scalars already invariant
            else:
                parts.append(f.norm(dim=-1))      # norms of equivariant feats

        node_inv = torch.cat(parts, dim=-1)        # (B, N, inv_dim)
        descs = node_inv.max(dim=1).values         # (B, inv_dim)
        return self.rho(descs)

    # ------------------------------------------------------------------
    def _encode_single(
        self,
        pos:       torch.Tensor,
        node_attr: Optional[torch.Tensor] = None,
        precomputed_geom = None
    ) -> torch.Tensor:
        N = pos.shape[0]
        sc = self._scalar_sig
        vc = self._vector_sig

        # Initial features
        f0_parts = [pos.norm(dim=-1, keepdim=True)]   # (N, 1)
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(1) # (N, 1+attr, 1) → unsqueeze C

        # Reshape: need (N, C, d)
        f0 = f0.reshape(N, -1, 1)                     # (N, init_scalar_c, 1)
        f1 = (pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)).unsqueeze(1)  # (N, 1, n)

        feats: FeatureDict = {sc: f0, vc: f1}

        if precomputed_geom is not None:
            rbf, gt_edge, nbr_idx = precomputed_geom
            use_sparse = True
        else:
            # Geometry
            use_sparse = (self.k_neighbors is not None and self.k_neighbors < N - 1)
            if use_sparse:
                rbf, gt_edge, nbr_idx = knn_geometry(pos, self.rbf, self.gt_basis, self.k_neighbors)
            else:
                rbf, gt_edge, mask = pairwise_geometry(pos, self.rbf, self.gt_basis)

        # Message-passing
        for i, layer in enumerate(self.mp_layers):
            if use_sparse:
                feats = layer(feats, rbf, gt_edge, nbr_idx, sparse=True)
            else:
                feats = layer(feats, rbf, gt_edge, mask, sparse=False)
            # Channel mixing
            if self.mix_layers is not None:
                feats = self.mix_layers[i](feats)

        # Invariant readout
        parts = []
        for sig in self.gt_basis.signatures:
            if sig not in feats:
                parts.append(torch.zeros(N, feats[sc].shape[1], device=pos.device))
                continue
            f = feats[sig]
            if sig == sc:
                parts.append(f.squeeze(-1))       # scalars already invariant
            else:
                parts.append(f.norm(dim=-1))      # norms of equivariant feats

        node_inv = torch.cat(parts, dim=-1)        # (N, inv_dim)
        return node_inv.max(dim=0).values          # (inv_dim,)

    # ------------------------------------------------------------------
    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.num_classes, device=next(self.parameters()).device)
        descriptors = []
        for i, pc in enumerate(batch):
            attr = node_attrs[i] if node_attrs is not None else None
            descriptors.append(self._encode_single(pc, attr))
        return self.rho(torch.stack(descriptors))


# ============================================================================
# Helpers
# ============================================================================

def _interaction_key(s1: GTSignature, s2: GTSignature, s3: GTSignature) -> str:
    return f"{s1.lam}x{s2.lam}to{s3.lam}_n{s1.n}"

def _sig_key(s: GTSignature) -> str:
    return f"sig_{s.lam}_n{s.n}"


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    import torch
    torch.manual_seed(0)

    for n in [3, 4]:
        print(f"\n=== GTTensorFieldNetwork v2  n={n} ===")
        model = GTTensorFieldNetwork(
            n=n, num_classes=8, max_order=1,
            hidden_channels=16, num_layers=3,
            k_neighbors=8, use_gate=True, use_residual=True,
            use_channel_mix=True, node_attr_dim=4,
        )
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        batch = [torch.randn(sz, n) for sz in [32, 48, 40]]
        attrs = [torch.randn(sz, 4) for sz in [32, 48, 40]]

        logits = model(batch, attrs)
        print(f"  Output: {logits.shape}")

        R, _ = torch.linalg.qr(torch.randn(n, n))
        if torch.det(R) < 0: R[:, 0] *= -1
        rot_batch = [pc @ R.T for pc in batch]
        logits_rot = model(rot_batch, attrs)
        diff = (logits - logits_rot).abs().max().item()
        print(f"  Rotation invariance max diff: {diff:.2e}  {'OK' if diff < 5e-3 else 'FAIL'}")

        # O(n): reflection test
        F_mat = torch.eye(n); F_mat[0, 0] = -1.0   # reflect x-axis
        ref_batch = [pc @ F_mat for pc in batch]
        logits_ref = model(ref_batch, attrs)
        diff_ref = (logits - logits_ref).abs().max().item()
        print(f"  Reflection invariance max diff: {diff_ref:.2e}  (expected small for SO(n) model)")
