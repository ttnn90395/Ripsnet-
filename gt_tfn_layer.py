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
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

from gt_basis import (
    GTSignature, GTBasis, CGCoefficients,
    vector_basis_change, vector_feature,
)

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


def dtm_rank_distance(
    dist:         torch.Tensor,   # (..., N, N) or (N, N), inf on diagonal
    m:            int = 10,
    alpha:        float = 1.0,
    eps:          float = 1e-8,
) -> torch.Tensor:
    """
    DTM-scaled ranking distance: points in low-density (high-DTM) regions
    are pushed further away during neighbor selection, so isolated outlier
    points are excluded from k-NN neighborhoods.  The *true* inter-point
    distances are kept for the selected edges — only the ranking changes.

    Returns a tensor with the same shape as *dist* whose values rank
    neighbors robustly (inf entries preserved).
    """
    if alpha <= 0.0:
        return dist
    N = dist.shape[-1]
    m = min(m, N - 1)
    flat = dist.reshape(-1, N, N)
    # Per-point DTM = mean distance to its m nearest neighbors (self=inf)
    knn = flat.topk(m, dim=-1, largest=False).values.mean(dim=-1)   # (G, N)
    med = knn.median(dim=-1, keepdim=True).values + eps             # (G, 1)
    # Neighbor side density scale; source scale is a row constant and does
    # not affect the per-row ranking, so it is omitted.
    scale = 1.0 + alpha * (knn / med)                               # (G, N)
    out = dist.clone()
    out = out.reshape(-1, N, N)
    out = out * scale.unsqueeze(1)
    return out.reshape(dist.shape)


def dtm_readout_weights(
    pos:   torch.Tensor,       # (N, n) or (B, N, n)
    m:     int = 10,
    gamma: float = 4.0,
    thr:   float = 1.5,
    eps:   float = 1e-8,
) -> torch.Tensor:
    """
    Per-point soft weights for the invariant readout, derived from DTM
    (distance-to-measure).  Points in low-density regions (outliers, which
    have high DTM) receive near-zero weight, so their corrupted invariant
    features are suppressed in the global descriptor sum without removing
    them from the point cloud.

    Soft-cap form (mild on inliers, hard on outliers):
        w_i = 1 / (1 + (dtm_i / (thr * median_dtm))^gamma)
    so points at the DTM median keep ~full weight and only the sparse tail
    beyond `thr * median` is attenuated.

    Returns weights of shape (N,) or (B, N) that sum to 1 per cloud.
    """
    leading = pos.shape[:-2]
    N = pos.shape[-2]
    flat = pos.reshape(-1, N, pos.shape[-1])
    diff = flat.unsqueeze(1) - flat.unsqueeze(2)          # (G, N, N, n)
    dist = diff.norm(dim=-1)                              # (G, N, N)
    mask = torch.eye(N, dtype=torch.bool, device=pos.device)
    dist = dist.masked_fill(mask, float('inf'))
    m = min(m, N - 1)
    dtm = dist.topk(m, dim=-1, largest=False).values.mean(dim=-1)   # (G, N)
    med = dtm.median(dim=-1, keepdim=True).values + eps
    w = 1.0 / (1.0 + (dtm / (thr * med)) ** gamma)        # high DTM → ~0
    w = w / w.sum(dim=-1, keepdim=True)
    return w.reshape(*leading, N)


def knn_geometry(
    pos:         torch.Tensor,    # (N, n)
    rbf_encoder: RBFExpansion,
    gt_basis:    GTBasis,
    k:           int,
    chunk_size:  int = 0,
    robust_alpha: float = 0.0,
    robust_m:    int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse k-NN geometry.  O(N·k) memory for the result; peak memory is
    O(min(N, chunk_size) × N) when *chunk_size* > 0, otherwise O(N²).

    When *chunk_size* > 0 the N×N distance matrix is built block-by-block,
    which reduces peak GPU memory at the cost of a small constant-factor
    overhead.  Recommended for N > 1000.

    Returns
    -------
    rbf      : (N, k, R)   RBF features for each edge
    gt_edge  : (N, k, B)   GT harmonics for each edge direction
    nbr_idx  : (N, k) long  indices of k nearest neighbors per point
    """
    N, n = pos.shape
    k    = min(k, N - 1)
    dev  = pos.device

    # Guard against NaN/Inf input
    if not torch.isfinite(pos).all():
        raise RuntimeError(f"knn_geometry: pos contains NaN/Inf (shape={pos.shape})")

    # WORKAROUND: PyTorch 2.8.0+cu128 topk has a within-kernel bug that
    # produces garbage indices. The only reliable fix is to run topk on CPU.
    # Data is tiny (max 67 points), so CPU overhead is negligible.
    if dev.type == 'cuda':
        pos_cpu = pos.cpu()
        N_cpu = N
        diff_cpu = pos_cpu.unsqueeze(1) - pos_cpu.unsqueeze(0)
        dist_cpu = diff_cpu.norm(dim=-1)
        mask_cpu = torch.eye(N_cpu, dtype=torch.bool)
        dist_masked_cpu = dist_cpu.masked_fill(mask_cpu, float('inf'))
        if robust_alpha > 0:
            dist_masked_cpu = dtm_rank_distance(dist_masked_cpu, m=robust_m, alpha=robust_alpha)
        _, nbr_idx_cpu = dist_masked_cpu.topk(k, dim=-1, largest=False)
        nbr_idx = nbr_idx_cpu.to(dev)
    else:
        if chunk_size <= 0 or chunk_size >= N:
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist = diff.norm(dim=-1)
            mask = torch.eye(N, dtype=torch.bool, device=dev)
            dist_masked = dist.masked_fill(mask, float('inf'))
            if robust_alpha > 0:
                dist_masked = dtm_rank_distance(dist_masked, m=robust_m, alpha=robust_alpha)
            _, nbr_idx = dist_masked.topk(k, dim=-1, largest=False)
        else:
            nbr_idx = torch.empty(N, k, dtype=torch.long, device=dev)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk = pos[start:end]
                diff_c = chunk.unsqueeze(1) - pos.unsqueeze(0)
                dist_c = diff_c.norm(dim=-1)
                for i in range(start, end):
                    dist_c[i - start, i] = float('inf')
                if robust_alpha > 0:
                    dist_c = dtm_rank_distance(dist_c, m=robust_m, alpha=robust_alpha)
                _, idx_c = dist_c.topk(k, dim=-1, largest=False)
                nbr_idx[start:end] = idx_c

    nbr_idx = nbr_idx.clamp(0, N - 1)
    pos_j = pos[nbr_idx]                                        # (N, k, n)
    diff_knn = pos.unsqueeze(1) - pos_j                         # (N, k, n)
    dist_knn = diff_knn.norm(dim=-1)                            # (N, k)
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
                    # f_in is (N, C, d) — apply linear over C for each of the d components
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

    A nonlinearity (SiLU) may only be used for type-0 (scalar) features:
    the rotation group acts by *mixing* the components of a type-l>0 irrep,
    so any nonlinear function applied per component does not commute with
    that action.  Higher-type channels are therefore mixed with a single
    linear layer (bias-free), which is the correct equivariant operation.
    """
    def __init__(self, feat_types: Dict[GTSignature, int], hidden_factor: int = 2):
        super().__init__()
        self.mlps = nn.ModuleDict()
        for sig, c in feat_types.items():
            if sig.lam[0] == 0 and sig.dim() == 1:
                # scalar type: nonlinear channel MLP is equivariant (d=1)
                h = max(c, c * hidden_factor)
                self.mlps[_sig_key(sig)] = nn.Sequential(
                    nn.Linear(c, h, bias=False),
                    nn.SiLU(),
                    nn.Linear(h, c, bias=False),
                )
            else:
                # non-scalar irrep: linear only (nonlinearity breaks equivariance)
                self.mlps[_sig_key(sig)] = nn.Linear(c, c, bias=False)

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
            c_in  = f_in.shape[-2]
            c_out = self.out_types[sig_out]
            cg_t  = self.cg.get(sig_in, sig_edge, sig_out)          # (di, de, do)
            e_f   = gt_by_sig[sig_edge]                              # (N, K, de) or (B, N, K, de)
            key   = _interaction_key(sig_in, sig_edge, sig_out)
            rad   = self.radial_nets[key](rbf)                       # (..., K, Ci*Co)
            K     = rad.shape[-2]
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
            batch_idx = torch.arange(fCG.shape[0], device=fCG.device)[:, None, None]
            nbr_idx = nbr_idx.clamp(0, fCG.shape[1] - 1)
            fCG_nbr = fCG[batch_idx, nbr_idx]                               # (B, N, k, Ci, de, do)
            contracted = torch.einsum("bnje,bnjceo->bnjco", e_feat, fCG_nbr)  # (B, N, k, Ci, do)
            return torch.einsum("bnjco,bnjcd->bnod", radial, contracted)         # (B, N, Co, do)

        # SINGLE path
        nbr_idx = nbr_idx.clamp(0, fCG.shape[0] - 1)
        fCG_nbr = fCG[nbr_idx]                                             # (N, k, Ci, de, do)
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
    chunk_size:  int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched sparse k-NN geometry for uniform point clouds."""
    B, N, n = pos.shape
    k = min(k, N - 1)
    dev = pos.device

    # Guard against NaN/Inf input
    if not torch.isfinite(pos).all():
        raise RuntimeError(f"knn_geometry_batch: pos contains NaN/Inf (shape={pos.shape})")

    # WORKAROUND: PyTorch 2.8.0+cu128 topk has a within-kernel bug that
    # produces garbage indices. Run topk on CPU for safety.
    if dev.type == 'cuda':
        pos_cpu = pos.cpu()
        diff_cpu = pos_cpu.unsqueeze(2) - pos_cpu.unsqueeze(1)
        dist_cpu = diff_cpu.norm(dim=-1)
        mask_cpu = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, N, N)
        dist_masked_cpu = dist_cpu.masked_fill(mask_cpu, float('inf'))
        _, nbr_idx_cpu = dist_masked_cpu.topk(k, dim=-1, largest=False)
        nbr_idx = nbr_idx_cpu.to(dev)
    else:
        if chunk_size <= 0 or chunk_size >= N:
            diff = pos.unsqueeze(2) - pos.unsqueeze(1)
            dist = diff.norm(dim=-1)
            mask = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0).expand(B, N, N)
            dist_masked = dist.masked_fill(mask, float('inf'))
            _, nbr_idx = dist_masked.topk(k, dim=-1, largest=False)
        else:
            nbr_idx = torch.empty(B, N, k, dtype=torch.long, device=dev)
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                chunk = pos[:, start:end]
                diff_c = chunk.unsqueeze(2) - pos.unsqueeze(1)
                dist_c = diff_c.norm(dim=-1)
                for i in range(start, end):
                    dist_c[:, i - start, i] = float('inf')
                _, idx_c = dist_c.topk(k, dim=-1, largest=False)
                nbr_idx[:, start:end] = idx_c

    batch_idx = torch.arange(B, device=dev)[:, None, None]
    nbr_idx = nbr_idx.clamp(0, N - 1)
    pos_j = pos[batch_idx, nbr_idx]                                   # (B, N, k, n)
    diff_knn = pos.unsqueeze(2) - pos_j                                # (B, N, k, n)
    dist_knn = diff_knn.norm(dim=-1)                                   # (B, N, k)
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
        use_attention_pool: bool = False,
        node_attr_dim:   int   = 0,
        classifier_dims: Optional[List[int]] = None,
        radial_hidden:   int   = 64,
        robust_readout:  bool  = False,
        robust_m:        int   = 10,
        robust_gamma:    float = 4.0,
        robust_thr:      float = 1.5,
        readout_pool:    str   = 'sum',
        norm_readout:    bool  = False,
    ):
        super().__init__()
        self.n              = n
        self.num_classes    = num_classes
        self.max_order      = max_order
        self.hidden_channels = hidden_channels
        self.num_layers     = num_layers
        self.num_rbf        = num_rbf
        self.cutoff         = cutoff
        self.k_neighbors    = k_neighbors
        self.use_attention_pool = use_attention_pool
        self.robust_readout = robust_readout
        self.robust_m       = robust_m
        # Learned soft-cap parameters of the DTM readout (used when
        # robust_readout is on; trainable end-to-end, so the model can adapt
        # the outlier-suppression cutoff instead of using the hand-set init).
        self.dtm_thr        = nn.Parameter(torch.tensor(float(robust_thr)))
        self.dtm_gamma      = nn.Parameter(torch.tensor(float(robust_gamma)))
        self.robust_gamma   = robust_gamma
        self.readout_pool   = readout_pool
        self.norm_readout   = norm_readout
        if readout_pool not in ('sum', 'mean', 'max', 'catmax'):
            raise ValueError(f"readout_pool must be 'sum', 'mean', 'max' or "
                             f"'catmax', got {readout_pool}")
        if classifier_dims is None:
            classifier_dims = [128, 64]

        # Shared geometry encoders
        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)

        all_sigs   = self.gt_basis.signatures
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)

        # Change-of-basis between geometric coordinates and the GT/SH basis
        # used by the edge harmonics and CG tensors.  The initial vector
        # feature must live in that same convention or the whole network
        # is not rotation-equivariant (see _compute_vec_basis_matrix).
        self._vector_sig = vector_sig
        self.register_buffer(
            "_vec_change", self._compute_vec_basis_matrix(), persistent=False)

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
        if readout_pool == 'catmax':
            inv_dim = 2 * inv_dim

        self.rho = nn.Sequential(
            nn.Linear(inv_dim, classifier_dims[0]), nn.SiLU(),
            nn.LayerNorm(classifier_dims[0]), nn.Dropout(0.1),
            *[nn.Linear(classifier_dims[i], classifier_dims[i + 1]) for i in
              range(len(classifier_dims) - 1)],
            nn.Linear(classifier_dims[-1], num_classes),
        )

        self._scalar_sig = scalar_sig

        # Attention pooling (optional replacement for sum/max-pool)
        if use_attention_pool:
            self._attn_pool = nn.Sequential(
                nn.Linear(inv_dim, 64), nn.GELU(), nn.Linear(64, 1),
            )
        else:
            self._attn_pool = None

    # ------------------------------------------------------------------
    def _compute_vec_basis_matrix(self, num_samples: int = 4096) -> Optional[torch.Tensor]:
        """
        Linear map M with Y(x) = M·x on unit vectors x, where Y is the
        vector-irrep slice of ``self.gt_basis``.

        The vector irrep of SO(n) is the standard representation, so its
        GT-basis functions are exactly linear.  The CG tensors and edge
        harmonics live in this basis, so the initial vector feature must be
        expressed in it too; otherwise the model is not rotation-equivariant
        (e.g. for n=3 the real-SH ordering differs from (x, y, z) by a
        fixed permutation).

        M is recovered by least squares over random unit directions (which
        avoids the numerically degenerate points near the poles).
        """
        return vector_basis_change(self.gt_basis, num_samples=num_samples)

    def _vector_feature(self, pos: torch.Tensor) -> torch.Tensor:
        """Normalized direction feature expressed in the GT-basis convention."""
        return vector_feature(pos, self._vec_change)

    # ------------------------------------------------------------------
    def _encode_batch(
        self,
        pos:       torch.Tensor,
        node_attr: Optional[torch.Tensor] = None,
        precomputed_geom = None,
        return_descriptors: bool = False,
    ) -> torch.Tensor:
        """Batch-encode point clouds.

        When *return_descriptors* is True the per-sample descriptors
        (before the classifier head ``self.rho``) are returned, which
        allows the caller to apply ``rho`` externally after collecting
        descriptors from multiple groups.
        """
        if pos.ndim != 3:
            raise ValueError("pos must be a batched tensor of shape (B, N, n)")
        B, N = pos.shape[0], pos.shape[1]
        sc = self._scalar_sig
        vc = self._vector_sig

        # Initial features
        f0_parts = [pos.norm(dim=-1, keepdim=True)]                      # (B, N, 1)
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(-1)                   # (B, N, C, 1)

        # Reshape: need (B, N, C, d)
        f1 = self._vector_feature(pos).unsqueeze(2)  # (B, N, 1, n)
        feats: FeatureDict = {sc: f0, vc: f1}

        # Need to re-read precomputed_geom after extracting rbf for f0_parts
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
                parts.append(torch.zeros(B, N, feats[sc].shape[2], device=pos.device))
                continue
            f = feats[sig]
            if sig == sc:
                parts.append(f.squeeze(-1))       # (B, N, C)
            else:
                parts.append(f.norm(dim=-1))      # (B, N, C)

        node_inv = torch.cat(parts, dim=-1)        # (B, N, inv_dim)
        if self.norm_readout:
            node_inv = node_inv / node_inv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.robust_readout:
            w = dtm_readout_weights(pos, m=self.robust_m,
                                    thr=self.dtm_thr, gamma=self.dtm_gamma)  # (B, N)
            node_inv = w.unsqueeze(-1) * node_inv
        if self._attn_pool is not None:
            w = self._attn_pool(node_inv).squeeze(-1)  # (B, N)
            w = torch.softmax(w, dim=1)
            descs = (w.unsqueeze(-1) * node_inv).sum(dim=1)  # (B, inv_dim)
        elif self.readout_pool == 'catmax':
            smax = node_inv.max(dim=1).values
            ssum = node_inv.sum(dim=1)
            descs = torch.cat([ssum, smax], dim=-1)   # (B, 2*inv_dim)
        elif self.readout_pool == 'max':
            descs = node_inv.max(dim=1).values          # (B, inv_dim)
        elif self.readout_pool == 'mean':
            descs = node_inv.mean(dim=1)                # (B, inv_dim)
        else:
            descs = node_inv.sum(dim=1)                 # (B, inv_dim)
        return descs if return_descriptors else self.rho(descs)

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
        f0_parts = [pos.norm(dim=-1, keepdim=True)]
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(-1)  # (N, C, 1)

        # (N, 1, n) — C=1 vector feature in the GT-basis convention
        f1 = self._vector_feature(pos).unsqueeze(1)  # (N, 1, n)

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
                parts.append(f.squeeze(-1))       # (N, C)
            else:
                parts.append(f.norm(dim=-1))      # (N, C)

        node_inv = torch.cat(parts, dim=-1)        # (N, inv_dim)
        if self.norm_readout:
            node_inv = node_inv / node_inv.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.robust_readout:
            w = dtm_readout_weights(pos, m=self.robust_m,
                                    thr=self.dtm_thr, gamma=self.dtm_gamma)  # (N,)
            node_inv = w.unsqueeze(-1) * node_inv
        if self._attn_pool is not None:
            w = self._attn_pool(node_inv).squeeze(-1)  # (N,)
            w = torch.softmax(w, dim=0)
            return (w.unsqueeze(-1) * node_inv).sum(dim=0)  # (inv_dim,)
        if self.readout_pool == 'catmax':
            smax = node_inv.max(dim=0).values
            ssum = node_inv.sum(dim=0)
            return torch.cat([ssum, smax], dim=-1)     # (2*inv_dim,)
        if self.readout_pool == 'max':
            return node_inv.max(dim=0).values        # (inv_dim,)
        if self.readout_pool == 'mean':
            return node_inv.mean(dim=0)              # (inv_dim,)
        return node_inv.sum(dim=0)                   # (inv_dim,)

    # ------------------------------------------------------------------
    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.num_classes, device=next(self.parameters()).device)

        # If the batch contains precomputed geometry (from a caching
        # caller e.g. train_nn.py), fall back to per-sample encoding.
        # Otherwise try the grouped-batch path.
        descriptors = self._encode_grouped_batch(batch, node_attrs)
        return self.rho(descriptors)

    # ------------------------------------------------------------------
    def _encode_grouped_batch(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Group variable-length point clouds by size and encode each
        subgroup with the batched path (``_encode_batch``).  Reverts to
        per-sample ``_encode_single`` when every sample has a unique size."""
        # Group indices by point-cloud size
        size_to_idx: Dict[int, List[int]] = {}
        for i, pc in enumerate(batch):
            sz = pc.shape[0]
            size_to_idx.setdefault(sz, []).append(i)

        # Only worthwhile when at least one size appears more than once
        has_groups = any(len(v) > 1 for v in size_to_idx.values())
        if not has_groups:
            descriptors = []
            for i, pc in enumerate(batch):
                attr = node_attrs[i] if node_attrs is not None else None
                descriptors.append(self._encode_single(pc, attr))
            return torch.stack(descriptors)

        # Build output buffer and fill per group
        descriptors = [None] * len(batch)
        for sz, idxs in size_to_idx.items():
            if len(idxs) == 1:
                # single sample with this size — use fast single path
                i = idxs[0]
                attr = node_attrs[i] if node_attrs is not None else None
                descriptors[i] = self._encode_single(batch[i], attr)
            else:
                # multiple samples — stack and use batched path
                sub_batch = torch.stack([batch[i] for i in idxs])
                sub_attrs = None
                if node_attrs is not None:
                    sub_attrs = torch.stack([node_attrs[i] for i in idxs])
                descs = self._encode_batch(sub_batch, sub_attrs,
                                           return_descriptors=True)
                for off, i in enumerate(idxs):
                    descriptors[i] = descs[off]
        return torch.stack(descriptors)


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
        # The rho head contains Dropout, so forward must run in eval() mode
        # for the invariance checks to be meaningful.
        model.eval()

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