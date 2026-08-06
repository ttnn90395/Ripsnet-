"""
Robustness Utilities for Shape Classification
==============================================
Improvements to make TFN models more robust to outlier noise:

1. Noise augmentation — inject random outlier points during training
2. DTM filtering — score points by local density, downweight outliers
3. Multi-scale kNN — fuse geometry at multiple neighborhood sizes
4. Persistence diagram fusion — add PD branch to any model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# 1. Noise Augmentation
# ──────────────────────────────────────────────────────────────────────

def augment_with_outliers(point_clouds: List[torch.Tensor],
                          corruption_rate: float = 0.2,
                          scale: float = 2.0,
                          seed: Optional[int] = None) -> List[torch.Tensor]:
    """
    Inject random outlier points into point clouds for training augmentation.

    For each point cloud, replaces `corruption_rate` fraction of points
    with uniformly sampled points in a bounding box around the cloud.

    Args:
        point_clouds: list of (N, d) tensors
        corruption_rate: fraction of points to replace (0.0 = no augmentation)
        scale: bounding box multiplier around the cloud centroid
        seed: random seed for reproducibility

    Returns:
        list of (N, d) tensors with outlier points injected
    """
    if corruption_rate <= 0.0:
        return point_clouds

    rng = np.random.RandomState(seed)
    augmented = []
    for pc in point_clouds:
        pc_np = pc.detach().cpu().numpy().copy()
        N = pc_np.shape[0]
        n_outliers = int(N * corruption_rate)

        # Compute bounding box from centroid
        centroid = pc_np.mean(axis=0)
        extent = np.abs(pc_np - centroid).max(axis=0) * scale

        # Generate outlier points
        outlier_idx = rng.choice(N, size=n_outliers, replace=False)
        outliers = centroid + rng.uniform(-extent, extent, size=(n_outliers, pc_np.shape[1]))

        pc_np[outlier_idx] = outliers
        augmented.append(torch.tensor(pc_np, dtype=pc.dtype, device=pc.device))

    return augmented


# ──────────────────────────────────────────────────────────────────────
# 2. DTM-based Point Scoring
# ──────────────────────────────────────────────────────────────────────

def dtm_scores(point_cloud: torch.Tensor, m: int = 10) -> torch.Tensor:
    """
    Compute Distance-to-Measure scores for each point.
    Low score = high density (core point), high score = outlier.

    Args:
        point_cloud: (N, d) tensor
        m: number of nearest neighbors for DTM

    Returns:
        (N,) tensor of DTM scores (higher = more likely outlier)
    """
    N = point_cloud.shape[0]
    m = min(m, N - 1)
    pc = point_cloud.detach()

    # Pairwise distances
    diff = pc.unsqueeze(0) - pc.unsqueeze(1)  # (N, N, d)
    dist = diff.norm(dim=-1)                   # (N, N)

    # Mask self-distances
    mask = torch.eye(N, dtype=torch.bool, device=pc.device)
    dist = dist.masked_fill(mask, float('inf'))

    # k-nearest neighbor distances
    knn_dist, _ = dist.topk(m, dim=-1, largest=False)  # (N, m)

    # DTM = mean of kNN distances
    dtm = knn_dist.mean(dim=1)  # (N,)
    return dtm


def filter_by_dtm(point_clouds: List[torch.Tensor],
                  keep_ratio: float = 0.8,
                  m: int = 10) -> List[torch.Tensor]:
    """
    Remove low-density outlier points from each point cloud.

    Args:
        point_clouds: list of (N, d) tensors
        keep_ratio: fraction of points to keep (0.8 = remove 20% outliers)
        m: DTM neighborhood size

    Returns:
        list of point clouds with outliers removed (variable length)
    """
    filtered = []
    for pc in point_clouds:
        scores = dtm_scores(pc, m=m)
        n_keep = max(int(pc.shape[0] * keep_ratio), 1)
        _, top_idx = scores.topk(n_keep, largest=False)  # lowest DTM = densest
        top_idx, _ = top_idx.sort()  # restore original order
        filtered.append(pc[top_idx])
    return filtered


def dtm_weights(point_cloud: torch.Tensor, m: int = 10,
                temperature: float = 1.0) -> torch.Tensor:
    """
    Compute per-point attention weights based on DTM scores.
    High-density points get higher weight.

    Args:
        point_cloud: (N, d) tensor
        m: DTM neighborhood size
        temperature: softmax temperature (lower = sharper focus on dense points)

    Returns:
        (N,) tensor of weights summing to 1
    """
    scores = dtm_scores(point_cloud, m=m)
    # Invert: low DTM score → high weight
    weights = -scores / (scores.mean() + 1e-8) / temperature
    weights = torch.softmax(weights, dim=0)
    return weights


# ──────────────────────────────────────────────────────────────────────
# 3. Multi-Scale kNN Geometry
# ──────────────────────────────────────────────────────────────────────

def multi_scale_knn(pos: torch.Tensor, rbf_encoder, gt_basis, k_values: List[int]):
    """
    Compute kNN geometry at multiple scales and concatenate RBF features.

    Args:
        pos: (N, n) point coordinates
        rbf_encoder: shared RBFExpansion module
        gt_basis: shared GTBasis module
        k_values: list of k values to use

    Returns:
        (rbf, gt_edge, nbr_idx) from the largest k (for message passing),
        plus multi_scale_rbf: list of RBF tensors at each scale
    """
    from gt_tfn_layer import knn_geometry

    all_rbf = []
    all_gt = []
    all_nbr = []
    for k in k_values:
        r, g, n = knn_geometry(pos, rbf_encoder, gt_basis, k)
        all_rbf.append(r)
        all_gt.append(g)
        all_nbr.append(n)

    # Use the largest k for message passing
    largest_idx = np.argmax(k_values)
    return all_rbf[largest_idx], all_gt[largest_idx], all_nbr[largest_idx], all_rbf


# ──────────────────────────────────────────────────────────────────────
# 4. Attention Pooling
# ──────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learned attention-weighted pooling over points.
    Replaces max-pool/sum-pool with a soft assignment that learns
    to focus on informative points and ignore outliers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (N, D) or (B, N, D) point features
            mask: (N,) or (B, N) boolean mask (True = valid point)

        Returns:
            (D,) or (B, D) pooled features
        """
        if x.ndim == 2:
            # Single sample: (N, D)
            w = self.attn(x).squeeze(-1)  # (N,)
            if mask is not None:
                w = w.masked_fill(~mask, float('-inf'))
            w = torch.softmax(w, dim=0)
            return (w.unsqueeze(-1) * x).sum(dim=0)  # (D,)
        else:
            # Batched: (B, N, D)
            w = self.attn(x).squeeze(-1)  # (B, N)
            if mask is not None:
                w = w.masked_fill(~mask, float('-inf'))
            w = torch.softmax(w, dim=1)
            return (w.unsqueeze(-1) * x).sum(dim=1)  # (B, D)


# ──────────────────────────────────────────────────────────────────────
# 5. Persistence Diagram Fusion Module
# ──────────────────────────────────────────────────────────────────────

class PersistenceFusion(nn.Module):
    """
    Persistence diagram branch that can be added to any model.
    Processes H1 persistence diagram (birth, death) pairs → features.
    """

    def __init__(self, output_dim: int = 128,
                 pd_dims: tuple = (32, 64, 128),
                 rho_dims: tuple = (128, 64),
                 max_pd_pairs: int = 128):
        super().__init__()
        self.max_pd_pairs = max_pd_pairs

        # MLP on (birth, death) pairs
        layers = []
        in_dim = 2
        for h in pd_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)])
            in_dim = h
        self.pd_encoder = nn.Sequential(*layers)

        # Readout
        rho = []
        for h in rho_dims:
            rho.extend([nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h)])
            in_dim = h
        rho.append(nn.Linear(in_dim, output_dim))
        self.pd_rho = nn.Sequential(*rho)

    def forward(self, pd_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pd_list: list of (K_i, 2) tensors of (birth, death) pairs

        Returns:
            (B, output_dim) tensor
        """
        device = next(self.parameters()).device
        batch = []
        masks = []
        for pd in pd_list:
            padded = torch.zeros(self.max_pd_pairs, 2, device=device)
            m = torch.zeros(self.max_pd_pairs, dtype=torch.bool, device=device)
            if pd is not None and pd.shape[0] > 0:
                k = min(pd.shape[0], self.max_pd_pairs)
                padded[:k] = pd[:k].to(device)
                m[:k] = True
            batch.append(padded)
            masks.append(m)

        pd_padded = torch.stack(batch)   # (B, K, 2)
        pd_mask = torch.stack(masks)     # (B, K)

        B, K, _ = pd_padded.shape
        feat = self.pd_encoder(pd_padded.reshape(B * K, 2)).reshape(B, K, -1)
        feat = feat.masked_fill(~pd_mask.unsqueeze(-1), float('-inf'))
        feat, _ = feat.max(dim=1)  # (B, pd_dims[-1])
        return self.pd_rho(feat)   # (B, output_dim)


# ──────────────────────────────────────────────────────────────────────
# 6. Input Denoising (statistical + local regression)
# ──────────────────────────────────────────────────────────────────────

class StatisticalOutlierRemover(nn.Module):
    """Remove outlier points based on local mean distance.
    
    For each point, compute mean distance to k nearest neighbors.
    If this distance exceeds (global_mean + alpha * global_std), remove the point.
    Differentiable approximation: soft-weight points by outlier score.
    """
    def __init__(self, k=10, alpha=2.0):
        super().__init__()
        self.k = k
        self.alpha = alpha
    
    def forward(self, point_cloud):
        """point_cloud: (N, d) -> (N, d) with outlier scores for weighting"""
        N = point_cloud.shape[0]
        if N <= self.k:
            return point_cloud, torch.ones(N, device=point_cloud.device)
        
        # Compute pairwise distances
        diff = point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)  # (N, N, d)
        dist = diff.norm(dim=-1)  # (N, N)
        mask = torch.eye(N, dtype=torch.bool, device=point_cloud.device)
        dist = dist.masked_fill(mask, float('inf'))
        
        # k nearest neighbor mean distances
        knn_dist, _ = dist.topk(self.k, dim=-1, largest=False)  # (N, k)
        local_mean_dist = knn_dist.mean(dim=1)  # (N,)
        
        # Outlier score: how far from global distribution
        global_mean = local_mean_dist.mean()
        global_std = local_mean_dist.std() + 1e-8
        outlier_scores = (local_mean_dist - global_mean) / global_std  # (N,)
        
        # Soft weights: sigmoid-based (1 for inliers, ~0 for outliers)
        weights = torch.sigmoid(self.alpha - outlier_scores)  # (N,)
        
        return point_cloud, weights


class LocalMeanShiftDenoiser(nn.Module):
    """Denoise point cloud by shifting each point toward local centroid.
    
    For each point, compute weighted mean of k nearest neighbors (weight by 
    inverse distance), then shift point toward that centroid by factor lambda.
    Iterated for stability.
    """
    def __init__(self, k=10, n_iter=2, blend=0.5):
        super().__init__()
        self.k = k
        self.n_iter = n_iter
        self.blend = blend  # how much to shift (0=no shift, 1=full shift to centroid)
    
    def forward(self, point_cloud):
        """point_cloud: (N, d) -> denoised (N, d)"""
        N = point_cloud.shape[0]
        if N <= self.k:
            return point_cloud
        
        pc = point_cloud.detach()
        
        for _ in range(self.n_iter):
            diff = pc.unsqueeze(0) - pc.unsqueeze(1)  # (N, N, d)
            dist = diff.norm(dim=-1)  # (N, N)
            mask = torch.eye(N, dtype=torch.bool, device=pc.device)
            dist = dist.masked_fill(mask, float('inf'))
            
            knn_dist, knn_idx = dist.topk(self.k, dim=-1, largest=False)  # (N, k)
            
            # Inverse distance weights
            weights = torch.softmax(-knn_dist / (knn_dist.mean() + 1e-8), dim=-1)  # (N, k)
            
            # Gather neighbor positions
            neighbor_pos = pc[knn_idx]  # (N, k, d)
            
            # Weighted centroid
            centroid = (weights.unsqueeze(-1) * neighbor_pos).sum(dim=1)  # (N, d)
            
            # Shift
            pc = pc + self.blend * (centroid - pc)
        
        # Return with gradient connection
        result = point_cloud + (pc - pc.detach())  # straight-through estimator
        return result


def denoise_batch(point_clouds, method='statistical', **kwargs):
    """Apply denoising to a batch of point clouds.
    
    Args:
        point_clouds: list of (N, d) tensors
        method: 'statistical' for StatisticalOutlierRemover, 
                'meanshift' for LocalMeanShiftDenoiser
        **kwargs: passed to the denoiser
    
    Returns:
        list of denoised (N, d) tensors (potentially with variable length)
    """
    if method == 'statistical':
        remover = StatisticalOutlierRemover(
            k=kwargs.get('k', 10), alpha=kwargs.get('alpha', 2.0))
        result = []
        for pc in point_clouds:
            denoised, weights = remover(pc)
            # Keep points with weight > threshold
            keep = weights > kwargs.get('threshold', 0.3)
            if keep.sum() > 0:
                result.append(denoised[keep])
            else:
                result.append(denoised[:1])
        return result
    elif method == 'meanshift':
        denoiser = LocalMeanShiftDenoiser(
            k=kwargs.get('k', 10), n_iter=kwargs.get('n_iter', 2),
            blend=kwargs.get('blend', 0.5))
        return [denoiser(pc) for pc in point_clouds]
    else:
        raise ValueError(f"Unknown denoising method: {method}")


# ──────────────────────────────────────────────────────────────────────
# 7. Geometric Regularization Loss
# ──────────────────────────────────────────────────────────────────────

class GeometricRegularization(nn.Module):
    """Regularize model to be invariant to small input perturbations.
    
    During training, for each input point cloud, generate a perturbed version
    (small Gaussian noise + random permutation) and encourage the model's
    output distribution to be similar (low KL divergence).
    
    This is a LOSS function, not a preprocessing step.
    """
    def __init__(self, noise_scale=0.01, kl_weight=1.0):
        super().__init__()
        self.noise_scale = noise_scale
        self.kl_weight = kl_weight
    
    def forward(self, model, forward_fn, batch_data, batch_geom=None, 
                batch_targets=None, criterion=None, **forward_kwargs):
        """
        Compute CE loss + geometric regularization loss.
        
        Args:
            model: the neural network
            forward_fn: function(model, data, **kwargs) -> logits
            batch_data: list of input tensors
            batch_geom: precomputed geometry (optional)
            batch_targets: target labels
            criterion: loss function (CrossEntropyLoss)
            forward_kwargs: extra args for forward_fn
        
        Returns:
            total_loss, ce_loss, reg_loss
        """
        # Original forward
        logits_clean = forward_fn(model, batch_data, **forward_kwargs)
        ce_loss = criterion(logits_clean, batch_targets)
        
        # Perturbed forward
        perturbed_data = []
        for pc in batch_data:
            noise = torch.randn_like(pc) * self.noise_scale
            perm = torch.randperm(pc.shape[0])
            perturbed_data.append(pc[perm] + noise)
        
        logits_perturbed = forward_fn(model, perturbed_data, **forward_kwargs)
        
        # KL divergence between clean and perturbed outputs
        p = F.log_softmax(logits_clean, dim=1)
        q = F.softmax(logits_perturbed, dim=1)
        kl_loss = F.kl_div(p, q, reduction='batchmean')
        
        # Also add permutation invariance: output shouldn't change on permutation
        perm_loss = F.mse_loss(logits_clean, logits_perturbed)
        
        reg_loss = self.kl_weight * (kl_loss + perm_loss)
        total_loss = ce_loss + reg_loss
        
        return total_loss, ce_loss, reg_loss


# ──────────────────────────────────────────────────────────────────────
# 8. Model Ensemble
# ──────────────────────────────────────────────────────────────────────

class ModelEnsemble(nn.Module):
    """Soft-voting ensemble of multiple models.
    
    Wraps multiple models and averages their softmax outputs.
    Each model can have a different architecture.
    """
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        n = len(models)
        if weights is None:
            weights = [1.0 / n] * n
        self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, *args, **kwargs):
        """Average softmax predictions from all models."""
        logits_list = []
        for model in self.models:
            logits_list.append(model(*args, **kwargs))
        
        # Soft voting
        probs = F.softmax(torch.stack(logits_list), dim=-1)  # (M, B, C)
        weighted_probs = probs * self.weights.unsqueeze(0).unsqueeze(-1)
        avg_probs = weighted_probs.sum(dim=0)  # (B, C)
        
        # Return log-probs (so CrossEntropyLoss works)
        return torch.log(avg_probs + 1e-8)


def build_ensemble(model_names, num_classes, dim, hp, device, **extra_kwargs):
    """Build an ensemble from a list of model names."""
    from shape.train_shape import build_model
    models = []
    for name in model_names:
        m = build_model(name).to(device)
        models.append(m)
    return ModelEnsemble(models)


# ──────────────────────────────────────────────────────────────────────
# 9. Feature-Level PD Fusion (cross-attention, not late concatenation)
# ──────────────────────────────────────────────────────────────────────

class FeatureLevelPDFusion(nn.Module):
    """Fuse persistence diagram features into intermediate geometric features.
    
    Uses cross-attention: geometric features attend to PD features,
    allowing the model to selectively incorporate topological information
    at each layer rather than just at the final classification head.
    
    This is architecturally different from PersistenceFusion (Module 5)
    which only concatenates at the output.
    """
    def __init__(self, geo_dim: int, input_dim: int = None, pd_dim: int = 128, 
                 hidden_dim: int = 64, n_heads: int = 4,
                 max_pd_pairs: int = 128):
        super().__init__()
        self.max_pd_pairs = max_pd_pairs
        self.pd_dim = pd_dim
        self.geo_dim = geo_dim
        
        # Project input features to geo_dim if needed
        if input_dim is not None and input_dim != geo_dim:
            self.in_proj = nn.Linear(input_dim, geo_dim)
        else:
            self.in_proj = nn.Identity()
        
        # PD encoder: (birth, death) -> features
        self.pd_encoder = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(),
            nn.Linear(32, pd_dim), nn.LayerNorm(pd_dim),
        )
        
        # Cross-attention: geo queries, PD keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=geo_dim, num_heads=n_heads, 
            kdim=pd_dim, vdim=pd_dim, batch_first=True)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(geo_dim, geo_dim), nn.GELU(), nn.LayerNorm(geo_dim))
    
    def forward(self, geo_features: torch.Tensor, 
                pd_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            geo_features: (B, input_dim) geometric features from model
            pd_list: list of (K_i, 2) persistence diagrams
        
        Returns:
            (B, geo_dim) features augmented with PD information
        """
        device = geo_features.device
        B = geo_features.shape[0]
        
        # Project input to geo_dim
        geo_features = self.in_proj(geo_features)
        
        # Encode all PDs
        pd_encoded = []
        for pd in pd_list:
            padded = torch.zeros(self.max_pd_pairs, 2, device=device)
            if pd is not None and pd.shape[0] > 0:
                k = min(pd.shape[0], self.max_pd_pairs)
                padded[:k] = pd[:k].to(device)
            enc = self.pd_encoder(padded)  # (K, pd_dim)
            pd_encoded.append(enc)
        
        # Pad to max length for batching
        max_k = max(p.shape[0] for p in pd_encoded)
        pd_batch = torch.zeros(B, max_k, self.pd_dim, device=device)
        pd_mask = torch.ones(B, max_k, dtype=torch.bool, device=device)
        for i, p in enumerate(pd_encoded):
            k = p.shape[0]
            pd_batch[i, :k] = p
            pd_mask[i, :k] = False  # False = attend to this position
        
        # Cross-attention: geo queries attend to PD keys/values
        # geo_features: (B, 1, geo_dim) - single query
        geo_q = geo_features.unsqueeze(1)  # (B, 1, geo_dim)
        attended, _ = self.cross_attn(geo_q, pd_batch, pd_batch, 
                                       key_padding_mask=pd_mask)
        attended = attended.squeeze(1)  # (B, geo_dim)
        
        # Residual + projection
        out = self.out_proj(geo_features + attended)
        return out
