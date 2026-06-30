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
    GTTFNLayer, GTTensorFieldNetwork, 
    RBFExpansion,
    ChannelMixer, EquivariantGate, ResidualProjection,
    knn_geometry, knn_geometry_batch,
    pairwise_geometry, pairwise_geometry_batch,
    FeatureDict, _sig_key, _interaction_key,
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
    current = torch.randint(0, N, (), device=device, dtype=torch.long)
    for i in range(n_samples):
        selected[i] = current
        current_pos = pos[current]                    # (d,)
        d2 = ((pos - current_pos) ** 2).sum(dim=-1)  # (N,)
        dist = torch.minimum(dist, d2)
        current = dist.argmax(dim=0)

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
    k = min(k, N)
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

    def precompute_geometry(self, pos: torch.Tensor) -> dict:
        """
        Precompute all geometry-dependent tensors for this pool stage.

        Returns a dict with keys: cent_idx, centroids, nbr_idx, rbf, gt_edge.
        These depend only on *pos* and can be cached across training epochs.
        """
        N = pos.shape[0]
        M = min(self.n_centroids, N)
        dev = pos.device

        cent_idx  = farthest_point_sample(pos, M)
        centroids = pos[cent_idx]

        nbr_idx   = ball_query(pos, centroids, self.radius, self.k_local)

        K = nbr_idx.shape[1]
        pos_j      = pos[nbr_idx.reshape(-1)].reshape(M, K, pos.shape[-1])
        diff       = centroids.unsqueeze(1) - pos_j
        dist       = diff.norm(dim=-1).clamp(min=1e-8)
        r_hat      = diff / dist.unsqueeze(-1)
        rbf        = self.rbf_enc(dist)
        n_dim      = pos.shape[-1]
        gt_edge    = self.gt_basis(
            r_hat.reshape(M * K, n_dim)
        ).reshape(M, K, -1)

        return {
            'cent_idx':  cent_idx,
            'centroids': centroids,
            'nbr_idx':   nbr_idx,
            'rbf':       rbf,
            'gt_edge':   gt_edge,
        }

    def forward(
        self,
        pos:   torch.Tensor,   # (N, n)
        feats: FeatureDict,    # {sig: (N, C, d)}
        precomputed: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, FeatureDict]:
        """
        Returns (centroids (M, n), pooled feats {sig: (M, C, d)}).

        When *precomputed* is provided (from ``precompute_geometry``), the
        expensive FPS / ball-query / edge-tensor computation is skipped.
        """
        if precomputed is not None:
            cent_idx  = precomputed['cent_idx']
            centroids = precomputed['centroids']
            nbr_idx   = precomputed['nbr_idx']
            rbf       = precomputed['rbf']
            gt_edge   = precomputed['gt_edge']
        else:
            N     = pos.shape[0]
            M     = min(self.n_centroids, N)
            dev   = pos.device

            cent_idx  = farthest_point_sample(pos, M)
            centroids = pos[cent_idx]

            nbr_idx   = ball_query(pos, centroids, self.radius, self.k_local)

            K = nbr_idx.shape[1]
            pos_j      = pos[nbr_idx.reshape(-1)].reshape(M, K, pos.shape[-1])
            diff       = centroids.unsqueeze(1) - pos_j
            dist       = diff.norm(dim=-1).clamp(min=1e-8)
            r_hat      = diff / dist.unsqueeze(-1)
            rbf        = self.rbf_enc(dist)
            n_dim      = pos.shape[-1]
            gt_edge    = self.gt_basis(
                r_hat.reshape(M * K, n_dim)
            ).reshape(M, K, -1)

        K = nbr_idx.shape[1]

        # Feature aggregation: max-pool over neighbors
        out_feats: FeatureDict = {}
        for sig, f in feats.items():
            f_nbr    = f[nbr_idx.reshape(-1)].reshape(
                centroids.shape[0], K, f.shape[1], f.shape[2])
            f_pooled = f_nbr.max(dim=1).values
            out_feats[sig] = f_pooled

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
        self.k_neighbors = k_global  # alias for precompute_geometry caching

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
        self.rho = nn.Sequential(
            nn.Linear(inv_dim, 64), nn.SiLU(), nn.LayerNorm(64),
            nn.Linear(64, num_classes),
        )

        self._scalar_sig = scalar_sig
        self._vector_sig = vector_sig
        self._all_sigs   = all_sigs
        self._hidden_types = hidden_types

    def precompute_hierarchical_geometry(self, pos: torch.Tensor) -> List[dict]:
        """
        Precompute all geometry for every hierarchical stage.

        Returns a list of dicts (one per pool stage) with keys for both the
        pool aggregation (``cent_idx``, ``centroids``, ``nbr_idx``, ``rbf``,
        ``gt_edge``) and the post-pool message-passing (``post_rbf``,
        ``post_gt``, ``post_nbr``).
        """
        stage_geom = []
        cur_pos = pos
        for pool in self.pool_stages:
            sg = pool.precompute_geometry(cur_pos)

            centroids = sg['centroids']
            if centroids.shape[0] > self.k_global:
                pr, pg, pn = knn_geometry(
                    centroids, self.rbf, self.gt_basis, self.k_global)
                sg['post_rbf'] = pr
                sg['post_gt']  = pg
                sg['post_nbr'] = pn

            stage_geom.append(sg)
            cur_pos = centroids
        return stage_geom

    # ------------------------------------------------------------------
    def _encode_single(
        self,
        pos:       torch.Tensor,
        node_attr: Optional[torch.Tensor] = None,
        precomputed_geom: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        precomputed_stage_geom: Optional[List[dict]] = None,
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

        # Initial full-resolution k-NN layers (use cached geometry if available)
        if precomputed_geom is not None:
            rbf_g, gt_g, nbr_g = precomputed_geom
        else:
            rbf_g, gt_g, nbr_g = knn_geometry(pos, self.rbf, self.gt_basis, self.k_global)
        for layer in self.init_layers:
            feats = layer(feats, rbf_g, gt_g, nbr_g, sparse=True)
        feats = self.init_mixer(feats)

        # Hierarchical pool stages
        cur_pos = pos
        for s, (pool, mp_layers, mixer) in enumerate(
            zip(self.pool_stages, self.stage_layers, self.stage_mixers)
        ):
            stage_geom = precomputed_stage_geom[s] if precomputed_stage_geom is not None else None
            cur_pos, feats = pool(cur_pos, feats, precomputed=stage_geom)
            M = cur_pos.shape[0]
            if M > self.k_global:
                if stage_geom is not None and 'post_rbf' in stage_geom:
                    rbf_s, gt_s, nbr_s = (
                        stage_geom['post_rbf'], stage_geom['post_gt'], stage_geom['post_nbr'])
                else:
                    rbf_s, gt_s, nbr_s = knn_geometry(
                        cur_pos, self.rbf, self.gt_basis, self.k_global)
                for layer in mp_layers:
                    feats = layer(feats, rbf_s, gt_s, nbr_s, sparse=True)
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
        return node_inv.sum(dim=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
        precomputed_geom: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        precomputed_stage_geom: Optional[List[List[dict]]] = None,
    ) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.num_classes, device=next(self.parameters()).device)
        descs = []
        for i, pc in enumerate(batch):
            attr = node_attrs[i] if node_attrs is not None else None
            geom_i    = precomputed_geom[i] if precomputed_geom is not None else None
            stage_i   = precomputed_stage_geom[i] if precomputed_stage_geom is not None else None
            descs.append(self._encode_single(
                pc, attr, precomputed_geom=geom_i, precomputed_stage_geom=stage_i))
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


class GTTFNEncoder(nn.Module):
    """Reusable TFN encoder backbone for embeddings or downstream heads."""

    def __init__(
        self,
        n: int,
        embedding_dim: int = 128,
        max_order: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        classifier_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if classifier_dims is None:
            classifier_dims = [128, 64]
        self.encoder = GTTensorFieldNetwork(
            n=n,
            num_classes=embedding_dim,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            classifier_dims=classifier_dims,
        )

    def forward(self, batch: List[torch.Tensor], node_attrs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return self.encoder(batch, node_attrs) if node_attrs is not None else self.encoder(batch)


# ============================================================================
# GTTFNAttentionLayer — multi-head cross-attention message passing
# ============================================================================

class GTTFNAttentionLayer(nn.Module):
    """
    Equivariant message-passing layer with multi-head cross-attention.

    Instead of summing neighbor messages uniformly (as in GTTFNLayer), this
    layer computes learned attention weights per neighbor from radial features
    and aggregates with a weighted sum.  Multiple heads let different
    interaction patterns be learned.

    Equivariance is preserved because attention weights are scalars computed
    from rotation-invariant distances (via RBF features).
    """

    def __init__(
        self,
        n:             int,
        in_types:      Dict[GTSignature, int],
        out_types:     Dict[GTSignature, int],
        num_rbf:       int,
        cg:            CGCoefficients,
        gt_basis:      GTBasis,
        num_heads:     int = 4,
        use_gate:      bool = True,
        use_residual:  bool = True,
        radial_hidden: int = 64,
    ):
        super().__init__()
        self.n          = n
        self.in_types   = in_types
        self.out_types  = out_types
        self.num_rbf    = num_rbf
        self.cg         = cg
        self.gt_basis   = gt_basis
        self.num_heads  = num_heads

        # Radial + attention networks per interaction
        self.radial_nets   = nn.ModuleDict()
        self.attn_nets     = nn.ModuleDict()
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
                        # Single scalar attention weight per neighbour
                        self.attn_nets[key] = nn.Sequential(
                            nn.Linear(num_rbf, radial_hidden), nn.SiLU(),
                            nn.Linear(radial_hidden, 1),
                        )
                    self._interactions.append((sig_in, sig_edge, sig_out))

        # Equivariant instance norm
        self.layer_norms = nn.ModuleDict({
            _sig_key(sig): nn.LayerNorm(c_out)
            for sig, c_out in out_types.items()
        })

        # Gated nonlinearity
        scalar_sig = GTSignature.scalar(n)
        self.gate = EquivariantGate(
            scalar_channels=out_types.get(scalar_sig, 1),
            feat_types=out_types,
        ) if use_gate and scalar_sig in out_types else None

        # Residual projection
        self.residual = ResidualProjection(in_types, out_types) if use_residual else None

        self._scalar_sig = scalar_sig

    # ------------------------------------------------------------------
    def forward(
        self,
        feats:    FeatureDict,
        rbf:      torch.Tensor,
        gt_edge:  torch.Tensor,
        mask_or_nbr: torch.Tensor,
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
            f_in  = feats[sig_in]
            c_in  = f_in.shape[-2]
            c_out = self.out_types[sig_out]
            cg_t  = self.cg.get(sig_in, sig_edge, sig_out)
            e_f   = gt_by_sig[sig_edge]
            key   = _interaction_key(sig_in, sig_edge, sig_out)
            rad   = self.radial_nets[key](rbf)
            attn  = self.attn_nets[key](rbf)
            K     = rad.shape[-2]

            # Softmax attention over neighbours (scalar weight per neighbour)
            if sparse:
                attn = F.softmax(attn, dim=-2)
            else:
                attn = F.softmax(
                    attn.masked_fill(~mask_or_nbr.unsqueeze(-1), float('-inf')),
                    dim=-2)

            if batch_mode:
                rad = rad.reshape(B, N, K, c_in, c_out)
                fCG = torch.einsum("bnci,ieo->bnceo", f_in, cg_t)
            else:
                rad = rad.reshape(N, K, c_in, c_out)
                fCG = torch.einsum("jci,ieo->jceo", f_in, cg_t)

            if sparse:
                msg = self._message_sparse_attn(fCG, e_f, rad, attn,
                                                mask_or_nbr, N, K, c_in, c_out)
            else:
                msg = self._message_dense_attn(fCG, e_f, rad, attn,
                                               mask_or_nbr, N, K, c_in, c_out)

            out[sig_out] = out[sig_out] + msg

        out = self._apply_norm(out)
        if self.gate is not None:
            out = self.gate(out, self._scalar_sig)
        if self.residual is not None:
            out = self.residual(feats, out)
        return out

    # ------------------------------------------------------------------
    def _message_sparse_attn(self, fCG, e_feat, radial, attn,
                             nbr_idx, N, k, c_in, c_out):
        """Sparse (N,k) message passing with scalar attention weighting."""
        if fCG.ndim == 5:  # batched
            batch_idx = torch.arange(fCG.shape[0], device=fCG.device)[:, None, None]
            if nbr_idx.max() >= fCG.shape[1] or nbr_idx.min() < 0:
                nbr_idx = nbr_idx.clamp(0, fCG.shape[1] - 1)
            fCG_nbr = fCG[batch_idx, nbr_idx]                              # (B,N,k,Ci,de,do)
            contracted = torch.einsum("bnje,bnjceo->bnjco", e_feat, fCG_nbr) # (B,N,k,Ci,do)
            contracted = contracted * attn.unsqueeze(-2)                     # (B,N,k,Ci,do) * (B,N,k,1,1)
            return torch.einsum("bnjco,bnjcd->bnod", radial, contracted)

        # single
        if nbr_idx.max() >= fCG.shape[0] or nbr_idx.min() < 0:
            nbr_idx = nbr_idx.clamp(0, fCG.shape[0] - 1)
        fCG_nbr = fCG[nbr_idx]                                              # (N,k,Ci,de,do)
        contracted = torch.einsum("ije,ijceo->ijco", e_feat, fCG_nbr)       # (N,k,Ci,do)
        contracted = contracted * attn.unsqueeze(-1)                        # (N,k,Ci,do) * (N,k,1,1)
        return torch.einsum("ijco,ijcd->iod", radial, contracted)

    # ------------------------------------------------------------------
    def _message_dense_attn(self, fCG, e_feat, radial, attn,
                            mask, N, K, c_in, c_out):
        """Dense (N,N) message passing with scalar attention weighting."""
        if fCG.ndim == 5:  # batched
            contracted = torch.einsum("bnje,bnceo->bnjco", e_feat, fCG)     # (B,N,N,Ci,do)
            contracted = contracted * mask.unsqueeze(-1).unsqueeze(-1).float()
            contracted = contracted * attn.unsqueeze(-2)                     # (B,N,N,1,1)
            return torch.einsum("bnjco,bnjcd->bnod", radial, contracted)

        contracted = torch.einsum("ije,jceo->ijco", e_feat, fCG)           # (N,N,Ci,do)
        contracted = contracted * mask.unsqueeze(-1).unsqueeze(-1).float()
        contracted = contracted * attn.unsqueeze(-1)                        # (N,N,1,1)
        return torch.einsum("ijco,ijcd->iod", radial, contracted)

    # ------------------------------------------------------------------
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


# ============================================================================
# GTTensorFieldNetworkWithAttention — full model with attention layers
# ============================================================================

class GTTensorFieldNetworkWithAttention(nn.Module):
    """
    GTTensorFieldNetwork with multi-head cross-attention message passing.

    Same interface as GTTensorFieldNetwork but replaces each GTTFNLayer with
    a GTTFNAttentionLayer that weights neighbour contributions via learned
    attention over radial features.

    Parameters
    ----------
    n               : ambient dimension (2,3,…)
    num_classes     : output classes
    max_order       : max GT irrep order  (default 1)
    hidden_channels : channels per irrep  (default 32)
    num_layers      : number of layers    (default 4)
    num_heads       : attention heads     (default 4)
    num_rbf         : RBF centres         (default 32)
    cutoff          : RBF cutoff distance (default 5.0)
    k_neighbors     : k-NN neighbourhood  (default 16)
    use_gate        : gated nonlinearity  (default True)
    use_residual    : residual streams    (default True)
    use_channel_mix : channel mixing MLP  (default True)
    node_attr_dim   : per-point attributes (default 0)
    classifier_dims : MLP head           (default [128, 64])
    radial_hidden   : radial MLP hidden   (default 64)
    """

    def __init__(
        self,
        n:               int,
        num_classes:     int,
        max_order:       int   = 1,
        hidden_channels: int   = 32,
        num_layers:      int   = 4,
        num_heads:       int   = 4,
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

        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)

        all_sigs   = self.gt_basis.signatures
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)

        init_scalar_c = 1 + node_attr_dim
        self.node_attr_dim = node_attr_dim
        if node_attr_dim > 0:
            self.attr_proj = nn.Linear(node_attr_dim, node_attr_dim, bias=False)

        init_types   = {scalar_sig: init_scalar_c, vector_sig: 1}
        hidden_types = {sig: hidden_channels for sig in all_sigs}

        self.mp_layers   = nn.ModuleList()
        self.mix_layers  = nn.ModuleList() if use_channel_mix else None
        in_types = init_types

        for i in range(num_layers):
            self.mp_layers.append(GTTFNAttentionLayer(
                n=n, in_types=in_types, out_types=hidden_types,
                num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
                num_heads=num_heads, use_gate=use_gate,
                use_residual=use_residual, radial_hidden=radial_hidden,
            ))
            if use_channel_mix and self.mix_layers is not None:
                self.mix_layers.append(ChannelMixer(hidden_types))
            in_types = hidden_types

        inv_dim = hidden_channels * len(all_sigs)
        self.rho = nn.Sequential(
            nn.Linear(inv_dim, 64), nn.SiLU(), nn.LayerNorm(64),
            nn.Linear(64, num_classes),
        )

        self._scalar_sig = scalar_sig
        self._vector_sig = vector_sig

    # ------------------------------------------------------------------
    def _encode_batch(
        self,
        pos:       torch.Tensor,
        node_attr: Optional[torch.Tensor] = None,
        precomputed_geom = None,
        return_descriptors: bool = False,
    ) -> torch.Tensor:
        """Batch-encode point clouds with attention-based message passing."""
        if pos.ndim != 3:
            raise ValueError("pos must be a batched tensor of shape (B, N, n)")
        B, N = pos.shape[0], pos.shape[1]
        sc = self._scalar_sig
        vc = self._vector_sig

        f0_parts = [pos.norm(dim=-1, keepdim=True)]
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(-1)

        pos_norm = pos.norm(dim=-1, keepdim=True)
        pos_safe = pos / pos_norm.where(pos_norm > 0, torch.ones_like(pos_norm))
        f1 = pos_safe.unsqueeze(2)
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

        parts = []
        for sig in self.gt_basis.signatures:
            if sig not in feats:
                parts.append(torch.zeros(B, N, feats[sc].shape[2], device=pos.device))
                continue
            f = feats[sig]
            if sig == sc:
                parts.append(f.squeeze(-1))
            else:
                parts.append(f.norm(dim=-1))

        node_inv = torch.cat(parts, dim=-1)
        descs = node_inv.sum(dim=1)
        return descs if return_descriptors else self.rho(descs)

    # ------------------------------------------------------------------
    def _encode_single(
        self,
        pos:       torch.Tensor,
        node_attr: Optional[torch.Tensor] = None,
        precomputed_geom = None,
    ) -> torch.Tensor:
        N = pos.shape[0]
        sc = self._scalar_sig
        vc = self._vector_sig

        f0_parts = [pos.norm(dim=-1, keepdim=True)]
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(-1)

        pos_norm = pos.norm(dim=-1, keepdim=True)
        pos_safe = pos / pos_norm.where(pos_norm > 0, torch.ones_like(pos_norm))
        f1 = pos_safe.unsqueeze(1)
        feats: FeatureDict = {sc: f0, vc: f1}

        if precomputed_geom is not None:
            rbf, gt_edge, nbr_idx = precomputed_geom
            use_sparse = True
        else:
            use_sparse = (self.k_neighbors is not None and self.k_neighbors < N - 1)
            if use_sparse:
                rbf, gt_edge, nbr_idx = knn_geometry(pos, self.rbf, self.gt_basis, self.k_neighbors)
            else:
                rbf, gt_edge, mask = pairwise_geometry(pos, self.rbf, self.gt_basis)

        for i, layer in enumerate(self.mp_layers):
            if use_sparse:
                feats = layer(feats, rbf, gt_edge, nbr_idx, sparse=True)
            else:
                feats = layer(feats, rbf, gt_edge, mask, sparse=False)
            if self.mix_layers is not None:
                feats = self.mix_layers[i](feats)

        parts = []
        for sig in self.gt_basis.signatures:
            if sig not in feats:
                parts.append(torch.zeros(N, feats[sc].shape[1], device=pos.device))
                continue
            f = feats[sig]
            if sig == sc:
                parts.append(f.squeeze(-1))
            else:
                parts.append(f.norm(dim=-1))

        node_inv = torch.cat(parts, dim=-1)
        return node_inv.sum(dim=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass: encode batch and apply classifier head."""
        if len(batch) == 0:
            return torch.empty(0, self.num_classes, device=next(self.parameters()).device)
        descriptors = self._encode_grouped_batch(batch, node_attrs)
        return self.rho(descriptors)

    # ------------------------------------------------------------------
    def _encode_grouped_batch(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Group variable-length point clouds by size and encode each subgroup."""
        size_to_idx: Dict[int, List[int]] = {}
        for i, pc in enumerate(batch):
            sz = pc.shape[0]
            size_to_idx.setdefault(sz, []).append(i)

        has_groups = any(len(v) > 1 for v in size_to_idx.values())
        if not has_groups:
            descriptors = []
            for i, pc in enumerate(batch):
                attr = node_attrs[i] if node_attrs is not None else None
                descriptors.append(self._encode_single(pc, attr))
            return torch.stack(descriptors)

        device = batch[0].device
        descriptors = [None] * len(batch)
        for sz, idxs in size_to_idx.items():
            if len(idxs) == 1:
                i = idxs[0]
                attr = node_attrs[i] if node_attrs is not None else None
                descriptors[i] = self._encode_single(batch[i], attr)
            else:
                sub_batch = torch.stack([batch[i] for i in idxs])
                sub_attrs = None
                if node_attrs is not None:
                    sub_attrs = torch.stack([node_attrs[i] for i in idxs])
                descs = self._encode_batch(sub_batch, sub_attrs, return_descriptors=True)
                for off, i in enumerate(idxs):
                    descriptors[i] = descs[off]
        return torch.stack(descriptors)


# ============================================================================
# TemporalCrossAttentionTFN  —  GT-TFN + temporal transformer over points
# ============================================================================

class TemporalCrossAttentionTFN(nn.Module):
    """
    GT-TFN backbone with a transformer encoder that attends over the
    point (time) dimension, giving the model explicit access to temporal
    structure beyond what k-NN spatial attention provides.

    Architecture:
      1. GTTensorFieldNetwork backbone produces per-point descriptors
         (equivariant message passing in 2D time-value space).
      2. Per-point descriptors are summed → global descriptor (standard).
      3. Additionally, per-point descriptors are fed through a
         TransformerEncoder with sinusoidal positional encoding,
         then pooled → temporal descriptor.
      4. Global + temporal descriptors are concatenated → classifier head.

    Parameters
    ----------
    n               : ambient dimension (2 or 3)
    num_classes     : output classes
    max_order       : max GT irrep order  (default 1)
    hidden_channels : channels per irrep  (default 32)
    num_layers      : number of GT-TFN layers  (default 4)
    num_rbf         : RBF centres         (default 32)
    cutoff          : RBF cutoff distance (default 5.0)
    k_neighbors     : k-NN neighbourhood  (default 16)
    num_heads       : transformer heads   (default 4)
    transformer_layers : transformer layers (default 2)
    classifier_dims : MLP head           (default [128, 64])
    radial_hidden   : radial MLP hidden   (default 64)
    dropout         : transformer dropout (default 0.1)
    """

    def __init__(
        self,
        n:                int,
        num_classes:      int,
        max_order:        int   = 1,
        hidden_channels:  int   = 32,
        num_layers:       int   = 4,
        num_rbf:          int   = 32,
        cutoff:           float = 5.0,
        k_neighbors:      int   = 16,
        num_heads:        int   = 4,
        transformer_layers: int = 2,
        classifier_dims:  Optional[List[int]] = None,
        radial_hidden:    int   = 64,
        dropout:          float = 0.1,
    ):
        super().__init__()
        if classifier_dims is None:
            classifier_dims = [128, 64]
        self.n                = n
        self.num_classes      = num_classes
        self.max_order        = max_order
        self.k_neighbors      = k_neighbors

        # GT-TFN backbone (produces per-point descriptors)
        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)

        all_sigs   = self.gt_basis.signatures
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)

        init_types   = {scalar_sig: 2, vector_sig: 1}  # norm + 1 for scalar
        hidden_types = {sig: hidden_channels for sig in all_sigs}

        self.mp_layers  = nn.ModuleList()
        self.mix_layers = nn.ModuleList()
        in_types = init_types
        for i in range(num_layers):
            self.mp_layers.append(GTTFNLayer(
                n=n, in_types=in_types, out_types=hidden_types,
                num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
                use_gate=True, use_residual=True,
            ))
            self.mix_layers.append(ChannelMixer(hidden_types))
            in_types = hidden_types

        self.inv_dim = hidden_channels * len(all_sigs)

        # Temporal transformer encoder over point descriptors
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, self.inv_dim))
        nn.init.normal_(self.pos_encoder, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.inv_dim, nhead=num_heads,
            dim_feedforward=self.inv_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers)

        # Combined classifier: global pool + temporal pool
        combined_dim = self.inv_dim * 2
        self.rho = nn.Sequential(
            nn.Linear(combined_dim, classifier_dims[0]), nn.SiLU(),
            nn.LayerNorm(classifier_dims[0]) if n > 0 else nn.Identity(),
            *sum(([
                nn.Linear(d1, d2), nn.SiLU(),
                nn.LayerNorm(d2) if n > 0 else nn.Identity(),
            ] for d1, d2 in zip(classifier_dims, classifier_dims[1:])), []),
            nn.Linear(classifier_dims[-1], num_classes),
        )

        self._scalar_sig = scalar_sig
        self._vector_sig = vector_sig

    # ------------------------------------------------------------------
    def _encode_point_descriptors(
        self,
        pos: torch.Tensor,
        precomputed_geom=None,
    ) -> torch.Tensor:
        """Encode point cloud → per-point invariant descriptors (N, D)."""
        N = pos.shape[0]
        sc = self._scalar_sig
        vc = self._vector_sig

        f0_parts = [pos.norm(dim=-1, keepdim=True),
                    torch.ones(N, 1, device=pos.device)]
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(-1)

        pos_norm = pos.norm(dim=-1, keepdim=True)
        pos_safe = pos / pos_norm.where(pos_norm > 0, torch.ones_like(pos_norm))
        f1 = pos_safe.unsqueeze(1)
        feats: FeatureDict = {sc: f0, vc: f1}

        if precomputed_geom is not None:
            rbf, gt_edge, nbr_idx = precomputed_geom
            use_sparse = True
        else:
            use_sparse = (self.k_neighbors is not None and self.k_neighbors < N - 1)
            if use_sparse:
                rbf, gt_edge, nbr_idx = knn_geometry(
                    pos, self.rbf, self.gt_basis, self.k_neighbors)
            else:
                rbf, gt_edge, mask = pairwise_geometry(
                    pos, self.rbf, self.gt_basis)

        for i, layer in enumerate(self.mp_layers):
            if use_sparse:
                feats = layer(feats, rbf, gt_edge, nbr_idx, sparse=True)
            else:
                feats = layer(feats, rbf, gt_edge, mask, sparse=False)
            feats = self.mix_layers[i](feats)

        parts = []
        for sig in self.gt_basis.signatures:
            if sig not in feats:
                parts.append(torch.zeros(N, feats[sc].shape[1], device=pos.device))
                continue
            f = feats[sig]
            if sig == sc:
                parts.append(f.squeeze(-1))
            else:
                parts.append(f.norm(dim=-1))
        return torch.cat(parts, dim=-1)  # (N, D)

    # ------------------------------------------------------------------
    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass: encode + temporal transformer + classifier."""
        descs_list = []
        for pc in batch:
            point_desc = self._encode_point_descriptors(pc)
            global_desc = point_desc.sum(dim=0, keepdim=True)  # (1, D)

            # Temporal transformer over point dimension
            N = point_desc.shape[0]
            pe = self.pos_encoder[:, :N, :]
            temporal_input = point_desc.unsqueeze(0) + pe  # (1, N, D)
            temporal_out = self.transformer_encoder(temporal_input)
            temporal_desc = temporal_out.mean(dim=1)       # (1, D)

            combined = torch.cat([global_desc, temporal_desc], dim=-1)
            descs_list.append(self.rho(combined))
        return torch.cat(descs_list, dim=0)

    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        return self

    def cpu(self):
        super().cpu()
        return self


# ============================================================================
# EquivariantSetTransformer  —  encoder-decoder with cross-attention
# ============================================================================

class EquivariantSetTransformer(nn.Module):
    """
    Equivariant set transformer for PL/PI prediction from point clouds.

    Encodes a context point cloud with a GTTFNEncoder, then decodes query
    points via equivariant cross-attention.  Useful for:
      - Super-resolution / densification
      - Few-shot learning on point sets
      - Conditional generation of topological descriptors

    Parameters
    ----------
    n               : ambient dimension
    embedding_dim   : encoder output dimension  (default 128)
    num_classes     : output classes            (default 1 for PI regression)
    max_order       : max GT irrep order        (default 1)
    hidden_channels : encoder channels          (default 32)
    num_layers      : encoder layers            (default 4)
    num_heads       : cross-attention heads     (default 4)
    num_rbf         : RBF centres               (default 32)
    cutoff          : RBF cutoff               (default 5.0)
    k_neighbors     : k-NN neighbourhood       (default 16)
    decoder_dims    : decoder MLP hidden dims  (default [128, 64])
    """

    def __init__(
        self,
        n:               int,
        embedding_dim:   int   = 128,
        num_classes:     int   = 1,
        max_order:       int   = 1,
        hidden_channels: int   = 32,
        num_layers:      int   = 4,
        num_heads:       int   = 4,
        num_rbf:         int   = 32,
        cutoff:          float = 5.0,
        k_neighbors:     int   = 16,
        decoder_dims:    Optional[List[int]] = None,
    ):
        super().__init__()
        if decoder_dims is None:
            decoder_dims = [128, 64]
        self.n             = n
        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes
        self.k_neighbors   = k_neighbors

        # Shared geometry
        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)

        # Encoder: produces one global descriptor per context cloud
        self.encoder = GTTensorFieldNetwork(
            n=n, num_classes=embedding_dim,
            max_order=max_order, hidden_channels=hidden_channels,
            num_layers=num_layers, num_rbf=num_rbf, cutoff=cutoff,
            k_neighbors=k_neighbors, classifier_dims=[hidden_channels * 4, hidden_channels * 2],
        )

        # Cross-attention decoder: attends from query positions to encoded context
        self.query_proj = nn.Linear(n, embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(embedding_dim, decoder_dims[0]), nn.SiLU(),
            nn.LayerNorm(decoder_dims[0]),
            nn.Linear(decoder_dims[0], decoder_dims[-1]), nn.SiLU(),
            nn.LayerNorm(decoder_dims[-1]),
            nn.Linear(decoder_dims[-1], num_classes),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        context_batch: List[torch.Tensor],
        query_batch:   List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context_batch : List[Tensor(N_i, n)]  — context point clouds
        query_batch   : List[Tensor(M_i, n)]  — query points (same batch size)

        Returns
        -------
        out : Tensor(B, num_classes)  — per-cloud predictions
        """
        B = len(context_batch)
        # Encode each context cloud → one descriptor per sample
        ctx_descs = []
        for i, pc in enumerate(context_batch):
            d = self.encoder([pc])
            ctx_descs.append(d)  # (1, embedding_dim)
        ctx = torch.cat(ctx_descs, dim=0).unsqueeze(1)  # (B, 1, embedding_dim)

        # Encode query positions
        query_feats = []
        for i, q in enumerate(query_batch):
            qf = self.query_proj(q).unsqueeze(0)  # (1, M_i, embedding_dim)
            # Add position encoding
            query_feats.append(qf)

        # Cross-attention per sample (different M_i per sample)
        outputs = []
        for i in range(B):
            q = query_feats[i]                          # (1, M_i, D)
            out_i, _ = self.cross_attn(q, ctx[i:i+1], ctx[i:i+1])  # (1, M_i, D)
            out_i = out_i.mean(dim=1)                     # (1, D)
            outputs.append(self.decoder_mlp(out_i))       # (1, num_classes)

        return torch.cat(outputs, dim=0)                 # (B, num_classes)


# ============================================================================
# StochasticEquivariantTFN  —  uncertainty-aware equivariant model
# ============================================================================

class _StochasticMixtureHead(nn.Module):
    """Head used as ``.rho`` by the training/analysis harness.

    Takes invariant descriptors ``(B, D)`` and returns the weighted mixture
    mean prediction ``(B, num_classes)``.
    """
    def __init__(
        self,
        encoder_mlp:   nn.Module,
        mu_net:        nn.Module,
        logit_net:     nn.Module,
        num_mixtures:  int,
        num_classes:   int,
    ):
        super().__init__()
        self.encoder_mlp  = encoder_mlp
        self.mu_net       = mu_net
        self.logit_net    = logit_net
        self.num_mixtures = num_mixtures
        self.num_classes  = num_classes

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        h = self.encoder_mlp(desc)
        mu     = self.mu_net(h).reshape(-1, self.num_mixtures, self.num_classes)
        logits = self.logit_net(h)
        weights = F.softmax(logits, dim=-1)
        return (weights.unsqueeze(-1) * mu).sum(dim=1)


class StochasticEquivariantTFN(nn.Module):
    """
    Uncertainty-aware equivariant model that outputs a Gaussian mixture
    distribution over PL/PI descriptors instead of a point estimate.

    Built on top of GTTensorFieldNetwork.  The classifier head is replaced
    with a head that outputs mixture means, log-variances, and (optionally)
    mixture logits, enabling:
      - Epistemic uncertainty quantification
      - Multi-modal output distributions
      - Robust PL/PI prediction with confidence intervals

    Parameters
    ----------
    n               : ambient dimension
    num_classes     : output classes (per-mixture-component)
    num_mixtures    : number of Gaussian components (default 3)
    max_order       : max GT irrep order  (default 1)
    hidden_channels : channels per irrep  (default 32)
    num_layers      : number of layers    (default 4)
    num_rbf         : RBF centres         (default 32)
    cutoff          : RBF cutoff distance (default 5.0)
    k_neighbors     : k-NN neighbourhood  (default 16)
    use_gate        : gated nonlinearity  (default True)
    use_residual    : residual streams    (default True)
    use_channel_mix : channel mixing MLP  (default True)
    node_attr_dim   : per-point attributes (default 0)
    encoder_dims    : MLP head before mixture (default [256, 128])
    """

    def __init__(
        self,
        n:               int,
        num_classes:     int,
        num_mixtures:    int   = 3,
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
        encoder_dims:    Optional[List[int]] = None,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [256, 128]
        self.num_classes   = num_classes
        self.num_mixtures  = num_mixtures

        # Shared geometry & core
        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)
        self.k_neighbors = k_neighbors

        all_sigs   = self.gt_basis.signatures
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)

        init_scalar_c = 1 + node_attr_dim
        self.node_attr_dim = node_attr_dim
        if node_attr_dim > 0:
            self.attr_proj = nn.Linear(node_attr_dim, node_attr_dim, bias=False)

        init_types   = {scalar_sig: init_scalar_c, vector_sig: 1}
        hidden_types = {sig: hidden_channels for sig in all_sigs}

        self.mp_layers   = nn.ModuleList()
        self.mix_layers  = nn.ModuleList() if use_channel_mix else None
        in_types = init_types

        for _ in range(num_layers):
            self.mp_layers.append(GTTFNLayer(
                n=n, in_types=in_types, out_types=hidden_types,
                num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
                use_gate=use_gate, use_residual=use_residual,
            ))
            if use_channel_mix and self.mix_layers is not None:
                self.mix_layers.append(ChannelMixer(hidden_types))
            in_types = hidden_types

        inv_dim = hidden_channels * len(all_sigs)

        # Shared encoder MLP
        d = inv_dim
        enc_layers = []
        for h in encoder_dims:
            enc_layers += [nn.Linear(d, h), nn.SiLU(), nn.LayerNorm(h)]
            d = h
        self.encoder_mlp = nn.Sequential(*enc_layers)

        # Mixture heads
        self.mu_net    = nn.Linear(d, num_mixtures * num_classes)
        self.logvar_net = nn.Linear(d, num_mixtures * num_classes)
        self.logit_net  = nn.Linear(d, num_mixtures)

        # rho head: used by the training/analysis harness (forward_single/batch).
        # Encodes invariant descriptors → weighted mixture mean prediction.
        self.rho = _StochasticMixtureHead(
            self.encoder_mlp, self.mu_net, self.logit_net,
            num_mixtures, num_classes,
        )

        self._scalar_sig = scalar_sig
        self._vector_sig = vector_sig

    # ------------------------------------------------------------------
    def _encode_features(self, pos, node_attr, precomputed_geom):
        """Shared feature encoding, returns invariant descriptors."""
        sc, vc = self._scalar_sig, self._vector_sig
        N = pos.shape[0]

        f0_parts = [pos.norm(dim=-1, keepdim=True)]
        if node_attr is not None and self.node_attr_dim > 0:
            f0_parts.append(self.attr_proj(node_attr))
        f0 = torch.cat(f0_parts, dim=-1).unsqueeze(-1)

        pos_norm = pos.norm(dim=-1, keepdim=True)
        pos_safe = pos / pos_norm.where(pos_norm > 0, torch.ones_like(pos_norm))
        f1 = pos_safe.unsqueeze(1)
        feats: FeatureDict = {sc: f0, vc: f1}

        if precomputed_geom is not None:
            rbf, gt_edge, nbr_idx = precomputed_geom
            use_sparse = True
        else:
            use_sparse = (self.k_neighbors is not None and self.k_neighbors < N - 1)
            if use_sparse:
                rbf, gt_edge, nbr_idx = knn_geometry(pos, self.rbf, self.gt_basis, self.k_neighbors)
            else:
                rbf, gt_edge, mask = pairwise_geometry(pos, self.rbf, self.gt_basis)

        for i, layer in enumerate(self.mp_layers):
            if use_sparse:
                feats = layer(feats, rbf, gt_edge, nbr_idx, sparse=True)
            else:
                feats = layer(feats, rbf, gt_edge, mask, sparse=False)
            if self.mix_layers is not None:
                feats = self.mix_layers[i](feats)

        parts = []
        for sig in self.gt_basis.signatures:
            if sig not in feats:
                parts.append(torch.zeros(N, feats[sc].shape[1], device=pos.device))
                continue
            f = feats[sig]
            parts.append(f.squeeze(-1) if sig == sc else f.norm(dim=-1))

        node_inv = torch.cat(parts, dim=-1)
        return node_inv.sum(dim=0)

    # ------------------------------------------------------------------
    def _encode_single(
        self,
        pos:                    torch.Tensor,
        node_attr:              Optional[torch.Tensor] = None,
        precomputed_geom:       Optional[Tuple] = None,
        precomputed_stage_geom: Optional[List] = None,
    ) -> torch.Tensor:
        """Encode a single point cloud into an invariant descriptor.

        Delegates to ``_encode_features`` for the actual computation;
        ``precomputed_stage_geom`` is accepted for API compatibility with
        hierarchical TFN models but is ignored (StochasticEquivariantTFN
        has no hierarchical pooling stages).
        """
        return self._encode_features(pos, node_attr, precomputed_geom)

    # ------------------------------------------------------------------
    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
        return_dist: bool = False,
    ) -> torch.Tensor:
        """
        If *return_dist=False* (default), returns the mean prediction.
        If *return_dist=True*, returns a dict with keys:
            'mu'       : (B, num_mixtures, num_classes)
            'logvar'   : (B, num_mixtures, num_classes)
            'logits'   : (B, num_mixtures)
        """
        N = max(len(batch), 1)
        descs = torch.zeros(N, self.encoder_mlp[0].in_features, device=next(self.parameters()).device)
        for i, pc in enumerate(batch):
            attr = node_attrs[i] if node_attrs is not None else None
            descs[i] = self._encode_features(pc, attr, None)

        h = self.encoder_mlp(descs)
        mu     = self.mu_net(h).reshape(N, self.num_mixtures, self.num_classes)
        logvar = self.logvar_net(h).reshape(N, self.num_mixtures, self.num_classes)
        logits = self.logit_net(h)

        if return_dist:
            return {'mu': mu, 'logvar': logvar, 'logits': logits}

        # Weighted average for point prediction
        weights = F.softmax(logits, dim=-1)                     # (B, K)
        mean_pred = (weights.unsqueeze(-1) * mu).sum(dim=1)      # (B, C)
        return mean_pred

    def nll_loss(self, target: torch.Tensor, dist: dict) -> torch.Tensor:
        """Negative log-likelihood loss under the predicted mixture."""
        mu     = dist['mu']       # (B, K, C)
        logvar = dist['logvar']   # (B, K, C)
        logits = dist['logits']   # (B, K)

        var = logvar.exp().clamp(min=1e-6)
        diff = target.unsqueeze(1) - mu                         # (B, K, C)
        log_prob = -0.5 * (diff.pow(2) / var + logvar + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=-1)                          # (B, K)

        log_weights = F.log_softmax(logits, dim=-1)
        log_mix = torch.logsumexp(log_weights + log_prob, dim=-1)  # (B,)
        return -log_mix.mean()


# ============================================================================
# EquivariantGraphMamba  —  state-space equivariant layers
# ============================================================================

class GTMambaLayer(nn.Module):
    """
    Equivariant state-space message-passing layer inspired by Mamba (S6).

    For each point, the hidden state evolves through a selective scan over
    its k-nearest neighbours.  The recurrence is parameterised by learned
    scalar gates computed from the radial features, preserving equivariance.

    Conceptually:
        h_0  = linear(feats)
        for j in neighbours (sorted by distance):
            gate = sigmoid(MLP(rbf[d_{i,j}]))
            h_j  = (1 - gate) * h_{j-1} + gate * message(neighbour_j, h_{j-1})
        out = h_k

    This gives a depth-proportional receptive field with O(k) cost,
    enabling long-range dependencies without stacking many layers.
    """

    def __init__(
        self,
        n:             int,
        in_types:      Dict[GTSignature, int],
        out_types:     Dict[GTSignature, int],
        num_rbf:       int,
        cg:            CGCoefficients,
        gt_basis:      GTBasis,
        use_gate:      bool = True,
        use_residual:  bool = True,
        radial_hidden: int = 64,
    ):
        super().__init__()
        self.n          = n
        self.in_types   = in_types
        self.out_types  = out_types
        self.cg         = cg
        self.gt_basis   = gt_basis

        self.radial_nets    = nn.ModuleDict()
        self.state_nets     = nn.ModuleDict()
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
                        # Selective gate: scalar per channel per neighbour
                        self.state_nets[key] = nn.Sequential(
                            nn.Linear(num_rbf, radial_hidden), nn.SiLU(),
                            nn.Linear(radial_hidden, c_out),
                        )
                    self._interactions.append((sig_in, sig_edge, sig_out))

        self.layer_norms = nn.ModuleDict({
            _sig_key(sig): nn.LayerNorm(c_out)
            for sig, c_out in out_types.items()
        })

        scalar_sig = GTSignature.scalar(n)
        self.gate = EquivariantGate(
            scalar_channels=out_types.get(scalar_sig, 1),
            feat_types=out_types,
        ) if use_gate and scalar_sig in out_types else None
        self.residual = ResidualProjection(in_types, out_types) if use_residual else None
        self._scalar_sig = scalar_sig

    # ------------------------------------------------------------------
    def forward(
        self,
        feats:    FeatureDict,
        rbf:      torch.Tensor,
        gt_edge:  torch.Tensor,
        nbr_idx:  torch.Tensor,
        sparse:   bool = True,
    ) -> FeatureDict:
        """State-space message passing: selective scan over neighbours."""
        N = rbf.shape[0]
        out: FeatureDict = {
            sig: torch.zeros(N, c, sig.dim(), device=rbf.device)
            for sig, c in self.out_types.items()
        }
        gt_by_sig = self._split_edge(gt_edge)

        for sig_in, sig_edge, sig_out in self._interactions:
            if sig_in not in feats:
                continue
            f_in  = feats[sig_in]                                    # (N, Ci, di)
            c_in  = f_in.shape[-2]
            c_out = self.out_types[sig_out]
            cg_t  = self.cg.get(sig_in, sig_edge, sig_out)           # (di, de, do)
            e_f   = gt_by_sig[sig_edge]                              # (N, k, de)
            key   = _interaction_key(sig_in, sig_edge, sig_out)
            rad   = self.radial_nets[key](rbf)                       # (N, k, Ci*Co)
            gate  = torch.sigmoid(self.state_nets[key](rbf))          # (N, k, Co)
            k     = rad.shape[-2]

            rad = rad.reshape(N, k, c_in, c_out)
            fCG = torch.einsum("jci,ieo->jceo", f_in, cg_t)          # (N, Ci, de, do)

            # Selective scan over neighbours
            fCG_nbr = fCG[nbr_idx]                                   # (N, k, Ci, de, do)
            contracted = torch.einsum("ije,ijceo->ijco", e_f, fCG_nbr)  # (N, k, Ci, do)

            # State-space recurrence: h_j = (1-g) * h_{j-1} + g * msg_j
            # where g is the selective gate and msg_j = rad * contracted
            msg = torch.einsum("ijco,ijcd->iod", rad, contracted)    # (N, Co, do)

            # Cumulative scan with selective gate
            h = msg.new_zeros(N, 1, *msg.shape[1:])
            h_l = h.clone()
            for j in range(k):
                g_j = gate[:, j:j+1, None, :].transpose(-1, -2)      # (N, 1, Co, 1)
                h_l = (1 - g_j) * h_l + g_j * msg[:, j:j+1]          # (N, 1, Co, do)
                h = h + h_l
            h = h / k                                                 # average over steps

            out[sig_out] = out[sig_out] + h.squeeze(1)

        out = self._apply_norm(out)
        if self.gate is not None:
            out = self.gate(out, self._scalar_sig)
        if self.residual is not None:
            out = self.residual(feats, out)
        return out

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


class EquivariantGraphMambaNetwork(nn.Module):
    """
    Full equivariant model using state-space (Mamba-style) layers.

    Replaces the GTTFNLayer stack with GTMambaLayers that perform selective
    scans over k-NN neighbourhoods, enabling long-range dependencies with
    O(k) cost per layer.

    Parameters (same as GTTensorFieldNetwork for drop-in compatibility)
    ----------
    n               : ambient dimension
    num_classes     : output classes
    max_order       : max GT irrep order  (default 1)
    hidden_channels : channels per irrep  (default 32)
    num_layers      : number of layers    (default 4)
    num_rbf         : RBF centres         (default 32)
    cutoff          : RBF cutoff          (default 5.0)
    k_neighbors     : k-NN                (default 16)
    use_gate        : gated nonlinearity  (default True)
    use_residual    : residual streams    (default True)
    use_channel_mix : channel mixing MLP  (default True)
    classifier_dims : MLP head           (default [128, 64])
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
        classifier_dims: List[int] = [128, 64],
    ):
        super().__init__()
        self.n            = n
        self.num_classes  = num_classes
        self.k_neighbors  = k_neighbors

        self.rbf      = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)
        self.gt_basis = GTBasis(n=n, max_order=max_order)
        self.cg       = CGCoefficients(n=n, max_order=max_order)

        all_sigs   = self.gt_basis.signatures
        scalar_sig = GTSignature.scalar(n)
        vector_sig = GTSignature.vector(n)

        init_types   = {scalar_sig: 1, vector_sig: 1}
        hidden_types = {sig: hidden_channels for sig in all_sigs}

        self.mp_layers  = nn.ModuleList()
        in_types = init_types
        for _ in range(num_layers):
            self.mp_layers.append(GTMambaLayer(
                n=n, in_types=in_types, out_types=hidden_types,
                num_rbf=num_rbf, cg=self.cg, gt_basis=self.gt_basis,
                use_gate=use_gate, use_residual=use_residual,
            ))
            in_types = hidden_types

        self.channel_mixer = ChannelMixer(hidden_types) if use_channel_mix else None

        inv_dim = hidden_channels * len(all_sigs)
        self.rho = nn.Sequential(
            nn.Linear(inv_dim, 64), nn.SiLU(), nn.LayerNorm(64),
            nn.Linear(64, num_classes),
        )

        self._scalar_sig = scalar_sig
        self._vector_sig = vector_sig

    # ------------------------------------------------------------------
    def _encode_single(
        self,
        pos:             torch.Tensor,
        precomputed_geom = None,
    ) -> torch.Tensor:
        N = pos.shape[0]
        sc, vc = self._scalar_sig, self._vector_sig

        f0 = pos.norm(dim=-1, keepdim=True).unsqueeze(-1)
        pos_norm = pos.norm(dim=-1, keepdim=True)
        pos_safe = pos / pos_norm.where(pos_norm > 0, torch.ones_like(pos_norm))
        f1 = pos_safe.unsqueeze(1)
        feats: FeatureDict = {sc: f0, vc: f1}

        if precomputed_geom is not None:
            rbf, gt_edge, nbr_idx = precomputed_geom
        else:
            rbf, gt_edge, nbr_idx = knn_geometry(pos, self.rbf, self.gt_basis, self.k_neighbors)

        for layer in self.mp_layers:
            feats = layer(feats, rbf, gt_edge, nbr_idx, sparse=True)
        if self.channel_mixer is not None:
            feats = self.channel_mixer(feats)

        parts = []
        for sig in self.gt_basis.signatures:
            if sig not in feats:
                parts.append(torch.zeros(N, feats[sc].shape[1], device=pos.device))
                continue
            f = feats[sig]
            parts.append(f.squeeze(-1) if sig == sc else f.norm(dim=-1))

        node_inv = torch.cat(parts, dim=-1)
        return node_inv.sum(dim=0)

    # ------------------------------------------------------------------
    def forward(
        self,
        batch:      List[torch.Tensor],
        node_attrs: Optional[List[torch.Tensor]] = None,
        precomputed_geom: Optional[List] = None,
    ) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.num_classes, device=next(self.parameters()).device)
        descs = []
        for i, pc in enumerate(batch):
            g = precomputed_geom[i] if precomputed_geom is not None else None
            descs.append(self._encode_single(pc, precomputed_geom=g))
        return self.rho(torch.stack(descs))

