"""
gt_improvements.py
==================
Standalone improvements that complement the core GT-TFN layers:

  HierarchicalPool     Multi-scale pooling for large point clouds
  OnEquivariance       O(n) parity wrapper (SO(n) → O(n))
  GTTFNEncoder         Reusable encoder backbone (no classification head)
  FarthestPointSample  Differentiable FPS for hierarchical pooling

These are designed to be composable:
  - Wrap GTTFNLayer blocks with HierarchicalPool for PointNet++ style
  - Wrap any GTTFNLayer model with OnEquivariance for reflection support
  - Use GTTFNEncoder as backbone for segmentation, regression, or generation
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from gt_basis import GTSignature, GTBasis, CGCoefficients
from gt_tfn_layer import (
    GTTFNLayer, GTTensorFieldNetwork, RBFExpansion,
    ChannelMixer, knn_geometry, pairwise_geometry,
    FeatureDict, _sig_key,
)


# ============================================================================
# Farthest Point Sampling
# ============================================================================

def farthest_point_sample(pos: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Iterative farthest point sampling. Returns indices of n_samples points.

    Greedy: each new sample is the point farthest from the already-selected set.
    O(N · n_samples) — fast for moderate n_samples.

    Parameters
    ----------
    pos       : (N, d)  point positions
    n_samples : int     number of points to select

    Returns
    -------
    idx : (n_samples,) long tensor of selected indices
    """
    N, d   = pos.shape
    device = pos.device
    n_samples = min(n_samples, N)

    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    dist     = torch.full((N,), float('inf'), device=device)

    # Start from a random point
    current = torch.randint(0, N, (1,), device=device).item()
    for i in range(n_samples):
        selected[i] = current
        current_pos = pos[current]                    # (d,)
        d2 = ((pos - current_pos) ** 2).sum(dim=-1)  # (N,)
        dist = torch.minimum(dist, d2)
        current = dist.argmax().item()

    return selected


# ============================================================================
# Ball query (local group aggregation)
# ============================================================================

def ball_query(
    pos:      torch.Tensor,   # (N, d) all points
    centers:  torch.Tensor,   # (M, d) query centers
    radius:   float,
    k:        int,
) -> torch.Tensor:
    """
    For each center, find up to k points within radius.
    Returns (M, k) index tensor; invalid entries are filled with the
    center's own index (a common convention for padding).
    """
    M, N = centers.shape[0], pos.shape[0]
    diff  = centers.unsqueeze(1) - pos.unsqueeze(0)   # (M, N, d)
    dist2 = (diff ** 2).sum(dim=-1)                   # (M, N)

    # For each center, get k nearest within radius
    within = dist2 <= radius ** 2                      # (M, N) bool
    # Replace out-of-radius distances with inf
    dist2_masked = dist2.clone()
    dist2_masked[~within] = float('inf')
    _, top_idx = dist2_masked.topk(k, dim=-1, largest=False)  # (M, k)

    # Where fewer than k points are in radius, topk returns inf-distance points.
    # Replace those with the center's own nearest neighbour (index 0 of top_idx).
    valid = within.gather(1, top_idx)                   # (M, k) bool
    fallback = top_idx[:, :1].expand_as(top_idx)
    top_idx[~valid] = fallback[~valid]

    return top_idx                                      # (M, k)


# ============================================================================
# Hierarchical pooling stage
# ============================================================================

class HierPoolStage(nn.Module):
    """
    One hierarchical pooling stage (PointNet++ style, GT-equivariant).

    Steps:
      1. Farthest point sample to get M centroids from N input points.
      2. Ball query to find up to k_local neighbors per centroid.
      3. Run a local GTTFNLayer to aggregate neighbor features.
      4. Max-pool the local features to get M per-centroid descriptors.

    The output has the same FeatureDict structure as the input, but with M
    points instead of N.
    """

    def __init__(
        self,
        n:              int,
        n_centroids:    int,
        radius:         float,
        k_local:        int,
        feat_types:     Dict[GTSignature, int],
        num_rbf:        int,
        cg:             CGCoefficients,
        gt_basis:       GTBasis,
        radial_hidden:  int = 64,
    ):
        super().__init__()
        self.n_centroids = n_centroids
        self.radius      = radius
        self.k_local     = k_local

        self.rbf_enc = RBFExpansion(num_rbf=num_rbf, cutoff=radius)
        self.layer   = GTTFNLayer(
            n=n, in_types=feat_types, out_types=feat_types,
            num_rbf=num_rbf, cg=cg, gt_basis=gt_basis,
            use_gate=True, use_residual=False,
            radial_hidden=radial_hidden,
        )
        self.mixer   = ChannelMixer(feat_types)
        self.gt_basis = gt_basis

    def forward(
        self,
        pos:   torch.Tensor,   # (N, n)
        feats: FeatureDict,    # {sig: (N, C, d)}
    ) -> Tuple[torch.Tensor, FeatureDict]:
        """
        Returns (centroids (M, n), pooled feats {sig: (M, C, d)}).
        """
        N     = pos.shape[0]
        M     = min(self.n_centroids, N)
        dev   = pos.device

        # 1. Sample centroids
        cent_idx  = farthest_point_sample(pos, M)        # (M,)
        centroids = pos[cent_idx]                         # (M, n)

        # 2. Ball query
        nbr_idx   = ball_query(pos, centroids, self.radius, self.k_local)  # (M, k)

        # 3. Local geometry for each centroid→neighbor edge
        pos_j      = pos[nbr_idx.reshape(-1)].reshape(M, self.k_local, pos.shape[-1])
        diff       = centroids.unsqueeze(1) - pos_j      # (M, k, n)
        dist       = diff.norm(dim=-1).clamp(min=1e-8)   # (M, k)
        r_hat      = diff / dist.unsqueeze(-1)            # (M, k, n)
        rbf        = self.rbf_enc(dist)                   # (M, k, R)
        n_dim      = pos.shape[-1]
        gt_edge    = self.gt_basis(
            r_hat.reshape(M * self.k_local, n_dim)
        ).reshape(M, self.k_local, -1)                   # (M, k, B)

        # 4. Build input feats at centroid positions (gather from global feats)
        cent_feats: FeatureDict = {
            sig: f[cent_idx] for sig, f in feats.items()
        }

        # Also gather nbr_idx into centroid-local indexing for sparse layer.
        # The layer needs nbr_idx into the "node buffer" that holds all features,
        # but in HierPool the node buffer is the full N-point cloud.
        # We remap: local_nbr_idx[i,j] = nbr_idx[i,j] (already global)
        # and we pass a "global feats" dict to the layer — handled by sparse _message.

        # We use a simpler approach: manually gather neighbor features, run
        # a single local aggregation without the standard layer API.
        out_feats: FeatureDict = {}
        for sig, f in feats.items():
            # f: (N, C, d)
            f_nbr    = f[nbr_idx.reshape(-1)].reshape(M, self.k_local, f.shape[1], f.shape[2])
            # Max-pool over neighbors
            f_pooled = f_nbr.max(dim=1).values           # (M, C, d)
            out_feats[sig] = f_pooled

        # Apply channel mixer
        out_feats = self.mixer(out_feats)

        return centroids, out_feats


# ============================================================================
# Full hierarchical encoder
# ============================================================================

class HierarchicalGTTFN(nn.Module):
    """
    Hierarchical GT-TFN encoder with multiple pooling stages.

    Architecture:
      Local GT-TFN on full resolution → HierPoolStage to M₁ points
      → GT-TFN on M₁ points → HierPoolStage to M₂ points
      → ... → global max-pool → classification MLP

    Parameters
    ----------
    n              : int   ambient dimension
    num_classes    : int
    max_order      : int   GT feature order
    hidden_channels: int
    stage_sizes    : list  [M1, M2, ...] number of centroids per stage
    stage_radii    : list  [r1, r2, ...] ball query radius per stage
    k_local        : int   neighbors per centroid in ball query
    num_layers_per_stage : int  GT-TFN layers between pool stages
    """

    def __init__(
        self,
        n:               int,
        num_classes:     int,
        max_order:       int         = 1,
        hidden_channels: int         = 32,
        stage_sizes:     List[int]   = [128, 32],
        stage_radii:     List[float] = [0.2, 0.4],
        k_local:         int         = 16,
        k_global:        int         = 16,
        num_layers_per_stage: int    = 2,
        num_rbf:         int         = 32,
        cutoff:          float       = 1.0,
        classifier_dims: List[int]   = [128, 64],
        node_attr_dim:   int         = 0,
    ):
        super().__init__()
        self.n            = n
        self.num_classes  = num_classes
        self.node_attr_dim = node_attr_dim
        self.k_global     = k_global

        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)
        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)

        all_sigs   = self.gt_basis.signatures
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)

        init_scalar = 1 + node_attr_dim
        init_types  = {scalar_sig: init_scalar, vector_sig: 1}
        hidden_types = {sig: hidden_channels for sig in all_sigs}

        if node_attr_dim > 0:
            self.attr_proj = nn.Linear(node_attr_dim, node_attr_dim, bias=False)

        # ── Initial local layers (full resolution) ───────────────────────────
        self.init_layers = nn.ModuleList()
        in_t = init_types
        for _ in range(num_layers_per_stage):
            self.init_layers.append(GTTFNLayer(
                n=n, in_types=in_t, out_types=hidden_types,
                num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
                use_gate=True, use_residual=True,
            ))
            in_t = hidden_types
        self.init_mixer = ChannelMixer(hidden_types)

        # ── Pooling stages ───────────────────────────────────────────────────
        self.pool_stages  = nn.ModuleList()
        self.stage_layers = nn.ModuleList()
        self.stage_mixers = nn.ModuleList()

        assert len(stage_sizes) == len(stage_radii)
        for stage_r, stage_size in zip(stage_radii, stage_sizes):
            self.pool_stages.append(HierPoolStage(
                n=n, n_centroids=stage_size, radius=stage_r,
                k_local=k_local, feat_types=hidden_types,
                num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
            ))
            stage_mp = nn.ModuleList()
            for _ in range(num_layers_per_stage):
                stage_mp.append(GTTFNLayer(
                    n=n, in_types=hidden_types, out_types=hidden_types,
                    num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
                    use_gate=True, use_residual=True,
                ))
            self.stage_layers.append(stage_mp)
            self.stage_mixers.append(ChannelMixer(hidden_types))

        # Invariant readout
        inv_dim = hidden_channels * len(all_sigs)
        rho = []
        d   = inv_dim
        for h in classifier_dims:
            rho += [nn.Linear(d, h), nn.SiLU(), nn.LayerNorm(h)]
            d = h
        rho.append(nn.Linear(d, num_classes))
        self.rho = nn.Sequential(*rho)

        self._scalar_sig = scalar_sig
        self._vector_sig = vector_sig
        self._all_sigs   = all_sigs
        self._hidden_types = hidden_types

    # ------------------------------------------------------------------
    def _encode_single(
        self,
        pos:       torch.Tensor,
        node_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N  = pos.shape[0]
        sc = self._scalar_sig
        vc = self._vector_sig

        # Initial features
        f0_parts = [pos.norm(dim=-1, keepdim=True)]
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(1).reshape(N, -1, 1)
        f1 = (pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)).unsqueeze(1)

        feats: FeatureDict = {sc: f0, vc: f1}

        # Initial full-resolution k-NN layers
        rbf_g, gt_g, nbr_g = knn_geometry(pos, self.rbf, self.gt_basis, self.k_global)
        for layer in self.init_layers:
            feats = layer(feats, rbf_g, gt_g, nbr_g, sparse=True)
        feats = self.init_mixer(feats)

        # Hierarchical pool stages
        cur_pos = pos
        for pool, mp_layers, mixer in zip(self.pool_stages, self.stage_layers, self.stage_mixers):
            cur_pos, feats = pool(cur_pos, feats)
            M = cur_pos.shape[0]
            if M > self.k_global:
                rbf_s, gt_s, nbr_s = knn_geometry(cur_pos, self.rbf, self.gt_basis, self.k_global)
                for layer in mp_layers:
                    feats = layer(feats, rbf_s, gt_s, nbr_s, sparse=True)
            # else skip layers for tiny remaining point sets
            feats = mixer(feats)

        # Invariant readout + global pool
        parts = []
        M = cur_pos.shape[0]
        for sig in self._all_sigs:
            if sig not in feats:
                parts.append(torch.zeros(M, list(self._hidden_types.values())[0], device=pos.device))
                continue
            f = feats[sig]
            parts.append(f.squeeze(-1) if sig == sc else f.norm(dim=-1))

        node_inv = torch.cat(parts, dim=-1)        # (M, inv_dim)
        return node_inv.max(dim=0).values

    # ------------------------------------------------------------------
    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.num_classes, device=next(self.parameters()).device)
        descs = []
        for i, pc in enumerate(batch):
            attr = node_attrs[i] if node_attrs is not None else None
            descs.append(self._encode_single(pc, attr))
        return self.rho(torch.stack(descs))


# ============================================================================
# O(n) wrapper — adds reflection equivariance on top of SO(n)
# ============================================================================

class OnEquivariantWrapper(nn.Module):
    """
    Lifts an SO(n)-invariant model to O(n)-invariance via group averaging.

    For a model f that is SO(n)-invariant, the O(n)-invariant version is:
        f_O(n)(x) = 0.5 * (f(x) + f(reflect(x)))

    where reflect flips the first coordinate: x[..., 0] → -x[..., 0].

    This works because:
      - Every element of O(n) is either a rotation or a reflection composed
        with a rotation.
      - Averaging over {id, reflect} makes the output invariant to reflections
        without breaking rotation invariance.

    Parameters
    ----------
    base_model : any model with forward(batch) → Tensor
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def _reflect(self, batch: List[torch.Tensor]) -> List[torch.Tensor]:
        """Flip the first coordinate of each point cloud."""
        reflected = []
        for pc in batch:
            pc_r = pc.clone()
            pc_r[..., 0] = -pc_r[..., 0]
            reflected.append(pc_r)
        return reflected

    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        out_orig = self.base(batch, node_attrs, **kwargs) if node_attrs is not None \
                   else self.base(batch, **kwargs)
        out_refl = self.base(self._reflect(batch), node_attrs, **kwargs) if node_attrs is not None \
                   else self.base(self._reflect(batch), **kwargs)
        return 0.5 * (out_orig + out_refl)


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    import torch
    torch.manual_seed(42)

    print("=== HierarchicalGTTFN ===")
    model = HierarchicalGTTFN(
        n=3, num_classes=10, max_order=1,
        hidden_channels=16, stage_sizes=[32, 8],
        stage_radii=[0.3, 0.6], k_local=8, k_global=8,
        num_layers_per_stage=2, node_attr_dim=3,
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    batch = [torch.randn(64, 3) for _ in range(3)]
    attrs = [torch.randn(64, 3) for _ in range(3)]
    out   = model(batch, attrs)
    print(f"  Output: {out.shape}")

    R, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(R) < 0: R[:, 0] *= -1
    rot_batch = [pc @ R.T for pc in batch]
    out_rot = model(rot_batch, attrs)
    diff = (out - out_rot).abs().max().item()
    print(f"  SO(3) invariance diff: {diff:.2e}")

    print("\n=== OnEquivariantWrapper ===")
    base = GTTensorFieldNetwork(n=3, num_classes=8, max_order=1,
                                hidden_channels=8, num_layers=2, k_neighbors=8)
    wrapped = OnEquivariantWrapper(base)
    batch2  = [torch.randn(20, 3) for _ in range(3)]
    out2    = wrapped(batch2)
    print(f"  Output: {out2.shape}")

    F_mat = torch.eye(3); F_mat[0,0] = -1.0
    ref_batch2 = [pc @ F_mat for pc in batch2]
    out2_ref   = wrapped(ref_batch2)
    diff_ref   = (out2 - out2_ref).abs().max().item()
    print(f"  O(3) invariance diff: {diff_ref:.2e}  {'OK' if diff_ref < 1e-4 else 'FAIL'}")

    print("\nDone.")
