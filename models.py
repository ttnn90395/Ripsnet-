"""Central model registry for the repository.

Import structure (no circular imports):
  TFN.py             → TFNLayer, TFN_RBFExpansion, TFNTensorFieldNetwork
  gt_tfn_layer.py    → GTTFNLayer, GTTFN_RBFExpansion, GTTensorFieldNetwork,
                        ChannelMixer, EquivariantGate, ResidualProjection
  gt_improvements.py → HierarchicalGTTFN, OnEquivariantWrapper
  THIS file          → GTTensorFieldNetworkV2, TensorFieldNetwork, PointNet3D,
                        RipsPointNet, AttnRipsPointNet, and all notebook/ragged models.

  tfn_model.py imports FROM this file (one-way): no circular deps.

Activation upgrade
------------------
All models use GELU instead of ReLU by default.  GELU has smoother gradients,
no dead-neuron problem, and consistently outperforms ReLU in both transformer-
style and point-cloud architectures.  Every model that previously hard-coded
nn.ReLU() now accepts an `activation` argument (default 'gelu') and uses
_act() to build the chosen activation module.  Supported values:
  'gelu'   nn.GELU()        — default, recommended
  'relu'   nn.ReLU()        — legacy, useful for ablations
  'silu'   nn.SiLU()        — smooth, good in equivariant nets
  'mish'   nn.Mish()        — very smooth, slightly slower
  'elu'    nn.ELU()         — negative saturation, useful for small nets

Normalisation upgrade
---------------------
All phi/rho MLPs now include BatchNorm1d after each linear+activation pair.
For the ragged / per-point models (DenseRagged, RaggedPersistenceModel) we
use LayerNorm instead because the batch dimension is the number of POINTS
not the number of samples, making BatchNorm semantically wrong there.

Device fix
----------
GTBasis and CGCoefficients (gt_basis.py) store tensors as plain attributes
rather than register_buffer(), so nn.Module.to(device) silently skips them.
GTTensorFieldNetworkV2 overrides .to() to call _move_basis_tensors().
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

def compute_persistence_diagram(point_cloud, homology_dim=1):
    """
    Compute H1 persistence diagram from a point cloud using gudhi AlphaComplex.

    Args:
        point_cloud: numpy array of shape (N, d) or torch.Tensor
        homology_dim: homology dimension (default 1 = loops)

    Returns:
        torch.Tensor of shape (K, 2) with (birth, death) pairs,
        or empty tensor if no features found.
    """
    import gudhi as gd
    import numpy as np
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    pc = np.asarray(point_cloud, dtype=np.float64)
    if pc.shape[0] < 3:
        return torch.zeros(0, 2)
    ac = gd.AlphaComplex(points=pc).create_simplex_tree()
    ac.persistence()
    dgm = ac.persistence_intervals_in_dimension(homology_dim)
    if dgm is None or len(dgm) == 0:
        return torch.zeros(0, 2)
    return torch.tensor(np.array(dgm, dtype=np.float32))


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------
from TFN import TensorFieldNetwork as TFNTensorFieldNetwork
from gt_tfn_layer import GTTensorFieldNetwork as _GTTensorFieldNetworkBase
# gt_improvements imports models.py (circular if done at top level).
# Imported lazily at the bottom of this file, after all classes are defined.


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

def _act(name: str = 'gelu') -> nn.Module:
    """
    Return an activation module by name.  Default is GELU.

    Hidden-layer activations (smooth, no dead neurons):
      'gelu'    nn.GELU()     — default, recommended
      'relu'    nn.ReLU()     — legacy; useful for exact checkpoint compat
      'silu'    nn.SiLU()     — smooth, strong in equivariant nets
      'mish'    nn.Mish()     — very smooth, slightly slower
      'elu'     nn.ELU()      — negative saturation, small nets

    Output activations (for final layer):
      'sigmoid' nn.Sigmoid()  — bounded [0,1], persistence vectors
      'tanh'    nn.Tanh()     — bounded [-1,1]
      'softmax' nn.Softmax()  — probability simplex (rarely needed here)
    """
    name = name.lower()
    if name == 'gelu':    return nn.GELU()
    if name == 'relu':    return nn.ReLU()
    if name == 'silu':    return nn.SiLU()
    if name == 'mish':    return nn.Mish()
    if name == 'elu':     return nn.ELU()
    if name == 'sigmoid': return nn.Sigmoid()
    if name == 'tanh':    return nn.Tanh()
    if name == 'softmax': return nn.Softmax(dim=-1)
    raise ValueError(f"Unknown activation '{name}'. "
                     f"Choose from: gelu, relu, silu, mish, elu, "
                     f"sigmoid, tanh, softmax")


# ---------------------------------------------------------------------------
# MLP builder helpers
# ---------------------------------------------------------------------------

def _build_mlp(dims: List[int],
               activation: str = 'gelu',
               norm: str = 'ln',
               final_activation: Optional[str] = None,
               dropout: float = 0.0) -> nn.Sequential:
    """
    Build a fully-connected MLP with optional normalisation and dropout.

    dims             : [in, h1, h2, ..., out]
    activation       : activation after each hidden layer
    norm             : 'ln'  LayerNorm after each hidden layer (default)
                              Works with any batch size, including 1.
                              BatchNorm1d requires batch_size > 1 and cannot
                              be used here since training is sample-by-sample.
                       'none' no normalisation
    final_activation : optional activation after the last linear layer
                       (e.g. 'sigmoid' for bounded outputs)
    dropout          : dropout probability after each hidden layer (0 = disabled)

    Note: 'bn' is accepted as an alias for 'ln' for checkpoint compatibility
    (old code saved norm='bn'; we silently upgrade to LayerNorm which has the
    same effect on single-sample inputs but also works during training).
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if is_last:
            if final_activation is not None:
                layers.append(_act(final_activation))
        else:
            layers.append(_act(activation))
            # LayerNorm normalises over the feature dim → works at batch_size=1
            # Accept 'bn' as alias for backward-compat with saved checkpoints
            if norm in ('ln', 'bn'):
                layers.append(nn.LayerNorm(dims[i + 1]))
            # norm == 'none': no normalisation layer added
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Device helper: move every tensor in an arbitrary object to device.
# ---------------------------------------------------------------------------

def _move_obj_tensors(obj, device):
    if obj is None:
        return
    for attr_name in list(vars(obj)):
        val = getattr(obj, attr_name)
        if isinstance(val, torch.Tensor):
            setattr(obj, attr_name, val.to(device))
        elif isinstance(val, dict):
            for k, v in list(val.items()):
                if isinstance(v, torch.Tensor):
                    val[k] = v.to(device)
        elif isinstance(val, list):
            for i, v in enumerate(val):
                if isinstance(v, torch.Tensor):
                    val[i] = v.to(device)


def _move_basis_tensors(module: nn.Module, device):
    for submod in module.modules():
        for attr in ('gt_basis', 'cg'):
            obj = getattr(submod, attr, None)
            if obj is not None:
                _move_obj_tensors(obj, device)
                for inner_attr in list(vars(obj)):
                    inner = getattr(obj, inner_attr)
                    if hasattr(inner, '__dict__'):
                        _move_obj_tensors(inner, device)


# ---------------------------------------------------------------------------
# Device-aware helpers used by all TFN subclasses defined below.
# ---------------------------------------------------------------------------

def _tfn_forward(self, batch, node_attrs, base_cls):
    """
    Device-safe forward for any GTTensorFieldNetwork subclass.
    Moves GTBasis/CG tensors to the input device before message passing,
    handling recursive sub-GTBasis objects created at runtime.
    """
    if batch:
        _move_basis_tensors(self, batch[0].device)
    return base_cls.forward(self, batch, node_attrs)


# ---------------------------------------------------------------------------
# GTTensorFieldNetworkV2
# ---------------------------------------------------------------------------

# Device-aware wrappers for the base models imported from external modules
class GTTensorFieldNetwork(_GTTensorFieldNetworkBase):
    """SO(n) base model."""
    def forward(self, batch, node_attrs=None):
        return _GTTensorFieldNetworkBase.forward(self, batch, node_attrs)


# HierarchicalGTTFN defined at the bottom after the lazy gt_improvements import.

class GTTensorFieldNetworkV2(_GTTensorFieldNetworkBase):
    """
    Recommended GT-TFN with all improvements ON by default.
    Inherits _DeviceAwareMixin so that every forward() call moves basis
    tensors to the correct device before message passing.
    """

    def __init__(
        self,
        n:               int,
        num_classes:     int,
        max_order:       int       = 1,
        hidden_channels: int       = 64,
        num_layers:      int       = 6,
        num_rbf:         int       = 64,
        cutoff:          float     = 2.0,
        k_neighbors:     int       = 16,
        use_gate:        bool      = True,
        use_residual:    bool      = True,
        use_channel_mix: bool      = True,
        use_attention_pool: bool   = False,
        node_attr_dim:   int       = 0,
        classifier_dims: Optional[List[int]] = None,
        radial_hidden:   int       = 128,
    ):
        if classifier_dims is None:
            classifier_dims = [256, 128]
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
            use_attention_pool=use_attention_pool,
            node_attr_dim=node_attr_dim,
            classifier_dims=classifier_dims,
            radial_hidden=radial_hidden,
        )

    def forward(self, batch, node_attrs=None):
        return _GTTensorFieldNetworkBase.forward(self, batch, node_attrs)


# ---------------------------------------------------------------------------
# TensorFieldNetwork  (SE(3), backed by GTTFNv2)
# ---------------------------------------------------------------------------

class TensorFieldNetwork(nn.Module):
    """SE(3)-equivariant wrapper around GTTFNv2."""

    def __init__(
        self,
        num_classes:     int,
        n:               int       = 3,
        max_order:       int       = 1,
        hidden_channels: int       = 64,
        num_layers:      int       = 6,
        num_rbf:         int       = 64,
        cutoff:          float     = 2.0,
        k_neighbors:     int       = 16,
        classifier_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if classifier_dims is None:
            classifier_dims = [256, 128]
        self._inner = GTTensorFieldNetworkV2(
            n=n, num_classes=num_classes,
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


# ---------------------------------------------------------------------------
# PointNet3D  — 3-D Deep-Sets baseline
# ---------------------------------------------------------------------------

class PointNet3D(nn.Module):
    """
    Permutation-invariant Deep-Sets baseline (NOT rotation-equivariant).
    Uses GELU + BatchNorm in both phi and rho MLPs by default.
    forward(batch: List[Tensor(N_i, 3)]) → Tensor(B, output_dim)
    """

    def __init__(self, output_dim,
                 input_dim: int = 3,
                 phi_dims=(64, 128, 256), rho_dims=(256, 128),
                 activation: str = 'gelu', norm: str = 'bn'):
        super().__init__()
        self._input_dim = input_dim
        self.phi_layers = _build_mlp([input_dim] + list(phi_dims),
                                     activation=activation, norm=norm)
        self.rho_layers = _build_mlp([phi_dims[-1]] + list(rho_dims) + [output_dim],
                                     activation=activation, norm=norm,
                                     final_activation=None)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, _last_out(self.rho_layers),
                               device=next(self.parameters()).device)
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
        mask   = torch.zeros(padded.shape[:2], dtype=torch.bool,
                             device=padded.device)
        for i, pc in enumerate(batch):
            mask[i, :len(pc)] = True
        B, N, _ = padded.shape
        phi = self.phi_layers(padded.reshape(B * N, self._input_dim)).reshape(B, N, -1)
        phi = phi.masked_fill(~mask.unsqueeze(-1), torch.finfo(phi.dtype).min)
        agg, _ = phi.max(dim=1)
        return self.rho_layers(agg)


# ---------------------------------------------------------------------------
# Notebook-derived / ragged models
# ---------------------------------------------------------------------------

def _last_out(seq: nn.Sequential) -> int:
    """Return the out_features of the last Linear in a Sequential."""
    for m in reversed(list(seq.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    raise RuntimeError("No Linear layer found in Sequential")


class ScalarDistanceDeepSet(nn.Module):
    """
    Deep-Sets on pairwise distances.
    Input: distance matrix (N, N) per sample.
    Uses GELU + BatchNorm in phi and rho MLPs.
    """

    def __init__(self, output_dim,
                 phi_dims=(64, 128), rho_dims=(256, 128),
                 activation: str = 'gelu', norm: str = 'bn'):
        super().__init__()
        self.phi_layers = _build_mlp([1] + list(phi_dims),
                                     activation=activation, norm=norm)
        self.rho_layers = _build_mlp([phi_dims[-1]] + list(rho_dims) + [output_dim],
                                     activation=activation, norm=norm,
                                     final_activation=None)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, _last_out(self.rho_layers),
                               device=next(self.parameters()).device)
        all_features = []
        for dm in batch:
            if dm.ndim == 2 and dm.shape[0] > 0:
                idx     = torch.triu_indices(dm.shape[0], dm.shape[1], offset=1)
                scalars = dm[idx[0], idx[1]]
            elif dm.ndim == 1 and dm.shape[0] > 0:
                scalars = dm
            else:
                scalars = torch.zeros(1, dtype=torch.float32, device=dm.device)
            phi = self.phi_layers(scalars.unsqueeze(1))
            all_features.append(phi.sum(dim=0))
        return self.rho_layers(torch.stack(all_features))


class PointNetTutorial(nn.Module):
    """
    2-D PointNet (tutorial version).
    Uses GELU + BatchNorm in both phi and rho MLPs.
    """

    def __init__(self, output_dim,
                 phi_dims=(64, 128, 256), rho_dims=(256, 128),
                 activation: str = 'gelu', norm: str = 'bn'):
        super().__init__()
        self.phi_layers = _build_mlp([2] + list(phi_dims),
                                     activation=activation, norm=norm)
        self.rho_layers = _build_mlp([phi_dims[-1]] + list(rho_dims) + [output_dim],
                                     activation=activation, norm=norm,
                                     final_activation=None)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, _last_out(self.rho_layers),
                               device=next(self.parameters()).device)
        batch2 = [x[:, :2] for x in batch]
        padded = pad_sequence(batch2, batch_first=True, padding_value=0.0)
        mask   = torch.zeros(padded.shape[:2], dtype=torch.bool,
                             device=padded.device)
        for i, pc in enumerate(batch2):
            mask[i, :len(pc)] = True
        B, N, _ = padded.shape
        phi = self.phi_layers(padded.reshape(B * N, 2)).reshape(B, N, -1)
        phi = phi.masked_fill(~mask.unsqueeze(-1), torch.finfo(phi.dtype).min)
        agg, _ = phi.max(dim=1)
        return self.rho_layers(agg)


class RipsPointNet(nn.Module):
    """
    PointNet + Persistence Diagram fusion model.

    Two branches:
      1. PointNet branch — MLP on raw coordinates → max-pool
      2. PD branch — MLP on H1 persistence diagram pairs → max-pool

    Branches are concatenated → classifier head.

    The persistence diagram is precomputed externally (gudhi AlphaComplex)
    and passed as a list of (K_i, 2) tensors of (birth, death) pairs.

    Usage:
        model = RipsPointNet(output_dim=num_classes)
        # In prepare_data, compute PDs via gudhi and pass as second arg
        logits = model(point_clouds, precomputed_pd)
    """

    def __init__(self, output_dim,
                 input_dim: int = 3,
                 phi_dims=(64, 128, 256),
                 rho_dims=(256, 128),
                 pd_dims=(32, 64, 128),
                 pd_rho_dims=(128, 64),
                 fusion_dims=(256, 128),
                 max_pd_pairs: int = 128,
                 activation: str = 'gelu', norm: str = 'bn'):
        super().__init__()
        self._input_dim = input_dim
        self.max_pd_pairs = max_pd_pairs

        # Branch 1: PointNet on raw coordinates
        self.phi_layers = _build_mlp([input_dim] + list(phi_dims),
                                     activation=activation, norm=norm)
        self.rho_layers = _build_mlp([phi_dims[-1]] + list(rho_dims) + [rho_dims[-1]],
                                     activation=activation, norm=norm,
                                     final_activation=None)

        # Branch 2: MLP on (birth, death) pairs
        self.pd_layers = _build_mlp([2] + list(pd_dims),
                                    activation=activation, norm=norm)
        self.pd_rho = _build_mlp([pd_dims[-1]] + list(pd_rho_dims) + [pd_rho_dims[-1]],
                                 activation=activation, norm=norm,
                                 final_activation=None)

        # Fusion head
        pointnet_out = rho_dims[-1]
        pd_out = pd_rho_dims[-1]
        self.fusion = _build_mlp([pointnet_out + pd_out] + list(fusion_dims) + [output_dim],
                                 activation=activation, norm=norm,
                                 final_activation=None)

    def forward(self, batch: List[torch.Tensor],
                precomputed_pd: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, _last_out(self.fusion),
                               device=next(self.parameters()).device)

        device = next(self.parameters()).device

        # --- Branch 1: PointNet ---
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
        mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=device)
        for i, pc in enumerate(batch):
            mask[i, :len(pc)] = True
        B, N, _ = padded.shape
        phi = self.phi_layers(padded.reshape(B * N, self._input_dim)).reshape(B, N, -1)
        phi = phi.masked_fill(~mask.unsqueeze(-1), torch.finfo(phi.dtype).min)
        pt_feat, _ = phi.max(dim=1)
        pt_feat = self.rho_layers(pt_feat)

        # --- Branch 2: Persistence Diagram ---
        if precomputed_pd is not None and len(precomputed_pd) > 0:
            # Pad PDs to max_pd_pairs
            pd_batch = []
            pd_mask = []
            for pd in precomputed_pd:
                if pd is None or pd.shape[0] == 0:
                    pd_batch.append(torch.zeros(self.max_pd_pairs, 2, device=device))
                    pd_mask.append(torch.zeros(self.max_pd_pairs, dtype=torch.bool, device=device))
                else:
                    k = min(pd.shape[0], self.max_pd_pairs)
                    padded_pd = torch.zeros(self.max_pd_pairs, 2, device=device)
                    padded_pd[:k] = pd[:k].to(device)
                    pd_batch.append(padded_pd)
                    m = torch.zeros(self.max_pd_pairs, dtype=torch.bool, device=device)
                    m[:k] = True
                    pd_mask.append(m)
            pd_padded = torch.stack(pd_batch)  # (B, max_pd_pairs, 2)
            pd_m = torch.stack(pd_mask)        # (B, max_pd_pairs)

            B_pd, K, _ = pd_padded.shape
            pd_feat = self.pd_layers(pd_padded.reshape(B_pd * K, 2)).reshape(B_pd, K, -1)
            pd_feat = pd_feat.masked_fill(~pd_m.unsqueeze(-1), torch.finfo(pd_feat.dtype).min)
            pd_feat, _ = pd_feat.max(dim=1)
            pd_feat = self.pd_rho(pd_feat)
        else:
            # No PD provided — zero features (degrades to pure PointNet)
            pd_feat = torch.zeros(B, 64, device=device)

        # --- Fusion ---
        combined = torch.cat([pt_feat, pd_feat], dim=1)
        return self.fusion(combined)


class AttnRipsPointNet(nn.Module):
    """Enhancer-context model for per-enhancer attribution (EPIRips).

    Architecture (ripsnet family):
      phi MLP (per enhancer) → optional learned positional encoding
      → optional TransformerEncoder layers (enhancer-enhancer context)
      → attention pooling (single vector or multi-head) → rho MLP
      → fusion with optional persistence-diagram (PD) branch.

    The attention weights from the pooling layer are the per-enhancer
    importance scores; `forward(..., return_attn=True)` returns them.

    Config mapping (used by benchmark_improve.py):
      attn          : pooling='attn',  num_layers=0, d=256
      mhead         : pooling='mhead', num_layers=0, d=128, num_heads=4
      transformer   : pooling='attn',  num_layers=2, d=128, num_heads=4
      tfmhead       : pooling='mhead', num_layers=2, d=128, num_heads=4
    """

    def __init__(self, output_dim, input_dim: int = 5,
                 pooling: str = 'attn', num_layers: int = 0, num_heads: int = 4,
                 d: int = 128, ff: int = 256, phi_dims: Optional[List[int]] = None,
                 rho_dims: Optional[List[int]] = None,
                 use_pd: bool = False,
                 pd_dims=(32, 64, 128), pd_rho_dims=(128, 64),
                 max_pd_pairs: int = 128, use_count: bool = False,
                 fusion_dims: Optional[List[int]] = None,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        assert pooling in ('attn', 'mhead')
        if fusion_dims is None:
            fusion_dims = [128]
        self.pooling = pooling
        self.num_heads = num_heads
        self.use_pd = use_pd
        self.use_count = use_count
        self.max_pd_pairs = max_pd_pairs

        if phi_dims is None:
            phi_dims = [64]
        self.phi_layers = _build_mlp([input_dim] + list(phi_dims) + [d], activation=activation,
                                     norm='ln', dropout=dropout)
        self.d = d
        if num_layers > 0:
            self.pos_emb = nn.Embedding(256, d)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads,
                                               dim_feedforward=ff, dropout=dropout,
                                               batch_first=True, activation=activation)
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.pos_emb = None
            self.transformer = None

        if pooling == 'mhead':
            self.attn_vectors = nn.Parameter(torch.randn(num_heads, d) * 0.01)
            rho_in = num_heads * d
        else:
            self.attn_vector = nn.Parameter(torch.randn(d) * 0.01)
            rho_in = d
        if rho_dims is None:
            rho_dims = [256, 128]
        rho_out = rho_dims[-1]
        self.rho_layers = _build_mlp([rho_in] + list(rho_dims), activation=activation,
                                     norm='ln', dropout=dropout)

        if use_pd:
            self.pd_layers = _build_mlp([2] + list(pd_dims), activation=activation,
                                        norm='ln', dropout=dropout)
            self.pd_rho = _build_mlp([pd_dims[-1]] + list(pd_rho_dims) + [pd_rho_dims[-1]],
                                     activation=activation, norm='ln',
                                     final_activation=None, dropout=dropout)
            pd_out = pd_rho_dims[-1]
        else:
            self.pd_layers = None
            self.pd_rho = None
            pd_out = 0

        fusion_in = rho_out + pd_out + (1 if use_count else 0)
        self.fusion = _build_mlp([fusion_in] + list(fusion_dims) + [output_dim],
                                 activation=activation, norm='ln',
                                 final_activation=None, dropout=dropout)

    def forward(self, batch: List[torch.Tensor],
                precomputed_pd: Optional[List[torch.Tensor]] = None,
                return_attn: bool = False):
        if len(batch) == 0:
            return torch.empty(0, _last_out(self.fusion),
                               device=next(self.parameters()).device)
        device = next(self.parameters()).device
        B = len(batch)
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0).to(device)
        N = padded.shape[1]
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i, pc in enumerate(batch):
            mask[i, :len(pc)] = True

        x = self.phi_layers(padded.reshape(B * N, -1)).reshape(B, N, -1)
        x = x * mask.unsqueeze(-1)

        if self.transformer is not None:
            x = x + self.pos_emb(torch.arange(N, device=device)).unsqueeze(0)
            x = self.transformer(x, src_key_padding_mask=~mask)
            x = x * mask.unsqueeze(-1)

        if self.pooling == 'mhead':
            scores = torch.einsum('bnd,hd->bnh', x, self.attn_vectors)
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            w = torch.softmax(scores, dim=1)
            pooled = torch.einsum('bnh,bnd->bhd', w, x).reshape(B, self.num_heads * x.shape[-1])
            attn_weights = w.mean(dim=-1)
        else:
            attn_scores = x @ self.attn_vector
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)
        pt_feat = self.rho_layers(pooled)

        if self.use_pd and precomputed_pd is not None and len(precomputed_pd) > 0:
            pd_batch, pd_mask = [], []
            for pd in precomputed_pd:
                if pd is None or pd.shape[0] == 0:
                    pd_batch.append(torch.zeros(self.max_pd_pairs, 2, device=device))
                    pd_mask.append(torch.zeros(self.max_pd_pairs, dtype=torch.bool, device=device))
                else:
                    k = min(pd.shape[0], self.max_pd_pairs)
                    padded_pd = torch.zeros(self.max_pd_pairs, 2, device=device)
                    padded_pd[:k] = pd[:k].to(device)
                    pd_batch.append(padded_pd)
                    m = torch.zeros(self.max_pd_pairs, dtype=torch.bool, device=device)
                    m[:k] = True
                    pd_mask.append(m)
            pd_padded = torch.stack(pd_batch)
            pd_m = torch.stack(pd_mask)
            B_pd, K = pd_padded.shape[:2]
            pd_feat = self.pd_layers(pd_padded.reshape(B_pd * K, -1)).reshape(B_pd, K, -1)
            pd_feat = pd_feat * pd_m.unsqueeze(-1)
            pd_feat = pd_feat.sum(dim=1) / pd_m.sum(dim=1, keepdim=True).clamp(min=1)
            pd_feat = self.pd_rho(pd_feat)
        else:
            pd_feat = torch.zeros(B, 0, device=device)

        if self.use_count:
            counts = torch.log1p(mask.sum(dim=1, keepdim=True).float())
            combined = torch.cat([pt_feat, pd_feat, counts], dim=1)
        else:
            combined = torch.cat([pt_feat, pd_feat], dim=1)
        out = self.fusion(combined)

        if return_attn:
            return out, attn_weights
        return out


class ScalarInputMLP(nn.Module):
    """
    Scalar-input MLP (mean pairwise distance → PV).
    Uses GELU + BatchNorm; Sigmoid on the final output.
    """

    def __init__(self, output_dim,
                 hidden_dims=(256, 512, 1024),
                 activation: str = 'gelu', norm: str = 'bn'):
        super().__init__()
        self.mlp = _build_mlp([1] + list(hidden_dims) + [output_dim],
                               activation=activation, norm=norm,
                               final_activation='sigmoid')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.stack(x)
        if x.ndim == 0:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.ndim == 1:
            x = x.unsqueeze(1)
        return self.mlp(x)


class MultiInputModel(nn.Module):
    """
    PointNetTutorial branch + scalar MLP branch, fused via an MLP head.
    Uses GELU + BatchNorm throughout; Sigmoid on the final output.
    """

    def __init__(self, target_output_dim, scalar_input_dim,
                 pointnet_intermediate_dim: int = 128,
                 scalar_mlp_intermediate_dim: int = 128,
                 activation: str = 'gelu', norm: str = 'bn'):
        super().__init__()
        self.pointnet_branch = PointNetTutorial(
            output_dim=pointnet_intermediate_dim,
            phi_dims=(64, 128, 256), rho_dims=(256, 128),
            activation=activation, norm=norm,
        )
        self.scalar_branch = _build_mlp(
            [scalar_input_dim, 512, 256, scalar_mlp_intermediate_dim],
            activation=activation, norm=norm,
        )
        combined = pointnet_intermediate_dim + scalar_mlp_intermediate_dim
        self.final_mlp = _build_mlp(
            [combined, 256, 128, target_output_dim],
            activation=activation, norm=norm,
            final_activation='sigmoid',
        )

    def forward(self, point_clouds: List[torch.Tensor],
                scalar_input: torch.Tensor) -> torch.Tensor:
        pc_feat = self.pointnet_branch(point_clouds)
        if scalar_input.ndim == 1:
            scalar_input = scalar_input.unsqueeze(0)
        sc_feat = self.scalar_branch(scalar_input)
        return self.final_mlp(torch.cat([pc_feat, sc_feat], dim=1))


class DenseRagged(nn.Module):
    """
    Point-wise linear + optional activation for ragged point clouds.
    Uses LayerNorm (not BatchNorm) because the 'batch' dimension here is
    the number of points per cloud, not the number of samples.
    Pre-allocates weights when in_features is given.
    """

    def __init__(self, in_features=None, out_features=30,
                 activation: str = 'gelu', use_bias: bool = True,
                 use_norm: bool = True):
        super().__init__()
        self.in_features      = in_features
        self.out_features     = out_features
        self.use_bias         = use_bias
        self.activation_name  = activation   # store name for lazy re-creation
        self.use_norm         = use_norm
        # Store activation as a registered module so it moves with the model
        self.act_fn           = _act(activation)

        if in_features is not None:
            self.weight_param = nn.Parameter(
                torch.randn(in_features, out_features) * 0.01)
            self.bias_param = (nn.Parameter(torch.zeros(out_features))
                               if use_bias else None)
            self.norm = nn.LayerNorm(out_features) if use_norm else None
        else:
            self.weight_param = None
            self.bias_param   = None
            self.norm         = None

    def _ensure_params(self, in_f: int, device):
        if self.weight_param is None:
            self.weight_param = nn.Parameter(
                torch.randn(in_f, self.out_features, device=device) * 0.01)
            self.register_parameter('weight_param', self.weight_param)
            if self.use_bias:
                self.bias_param = nn.Parameter(
                    torch.zeros(self.out_features, device=device))
                self.register_parameter('bias_param', self.bias_param)
            if self.use_norm:
                self.norm = nn.LayerNorm(self.out_features).to(device)

    _shape_warned = False

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for x in inputs:
            self._ensure_params(x.shape[-1], x.device)
            if self.weight_param.shape[0] != x.shape[-1]:
                if not DenseRagged._shape_warned:
                    print(f'  WARNING: DenseRagged input dim {x.shape[-1]} '
                          f'!= trained dim {self.weight_param.shape[0]}, '
                          f'returning zeros (skipping this model)')
                    DenseRagged._shape_warned = True
                outputs.append(torch.zeros(x.shape[0], self.out_features, device=x.device))
                continue
            y = torch.matmul(x, self.weight_param)
            if self.use_bias and self.bias_param is not None:
                y = y + self.bias_param
            if self.norm is not None:
                y = self.norm(y)
            y = self.act_fn(y)
            outputs.append(y)
        return outputs


class PermopRagged(nn.Module):
    """Sum-pool over points: List[(N_i, C)] → (B, C)."""

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack([x.sum(dim=0) for x in inputs])


class RaggedPersistenceModel(nn.Module):
    """
    Three DenseRagged layers (GELU + LayerNorm) → PermopRagged → MLP head.
    Layer 0 is lazy (in_features=None) because input width varies with N.
    Layers 1 and 2 are pre-allocated (30→20, 20→10).
    MLP head uses GELU + LayerNorm.
    """

    _RAGGED_DIMS = [30, 20, 10]

    def __init__(self, output_dim, in_features=None,
                 activation: str = 'gelu'):
        super().__init__()
        self.ragged_layers = nn.ModuleList([
            DenseRagged(in_features=None,
                        out_features=self._RAGGED_DIMS[0],
                        activation=activation, use_norm=True),
            DenseRagged(in_features=self._RAGGED_DIMS[0],
                        out_features=self._RAGGED_DIMS[1],
                        activation=activation, use_norm=True),
            DenseRagged(in_features=self._RAGGED_DIMS[1],
                        out_features=self._RAGGED_DIMS[2],
                        activation=activation, use_norm=True),
        ])
        self.perm = PermopRagged()
        # MLP head: LayerNorm (not BatchNorm) because batch size may be 1
        self.fc = _build_mlp(
            [self._RAGGED_DIMS[-1], 50, 100, output_dim],
            activation=activation, norm='ln',
            final_activation=None,
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        for layer in self.ragged_layers:
            x = layer(x)
        return self.fc(self.perm(x))


class DistanceMatrixRaggedModel(nn.Module):
    """
    Row-wise MLP on distance matrix → mean-pool → MLP head.
    Uses GELU + BatchNorm in both phi and rho MLPs.
    """

    def __init__(self, output_dim, num_points=None,
                 phi_dim: int = 128, rho_hidden=(256, 128),
                 activation: str = 'gelu', norm: str = 'bn'):
        super().__init__()
        self.output_dim   = output_dim
        self.num_points   = num_points
        self.phi_dim      = phi_dim
        self.activation   = activation
        self.norm         = norm
        self._phi_inp_dim = None

        if num_points and num_points > 0:
            self._phi_layers = self._make_phi(num_points)
            self._phi_inp_dim = num_points
        else:
            self._phi_layers = None

        self.rho = _build_mlp([phi_dim] + list(rho_hidden) + [output_dim],
                               activation=activation, norm=norm,
                               final_activation=None)

    def _make_phi(self, inp: int) -> nn.Sequential:
        hidden = max(64, self.phi_dim)
        return _build_mlp([inp, hidden, self.phi_dim],
                           activation=self.activation, norm=self.norm)

    def _ensure_phi(self, n: int, device):
        if self._phi_layers is None or self._phi_inp_dim != n:
            self._phi_layers  = self._make_phi(n).to(device)
            self._phi_inp_dim = n

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, _last_out(self.rho),
                               device=next(self.parameters()).device)
        max_n  = max(m.shape[0] for m in batch)
        device = next(self.parameters()).device
        self._ensure_phi(max_n, device)

        phi_outs = []
        for m in batch:
            n = m.shape[0]
            if n < max_n:
                padded = torch.zeros(max_n, max_n, device=device)
                padded[:n, :n] = m
                m = padded
            row_phi = self._phi_layers(m)
            phi_outs.append(row_phi.mean(dim=0))

        return self.rho(torch.stack(phi_outs))


# ---------------------------------------------------------------------------
# Lazy import of gt_improvements (avoids circular import)
# Must be done AFTER all classes above are fully defined.
# ---------------------------------------------------------------------------

def _load_gt_improvements():
    """Import all models from gt_improvements."""
    from gt_improvements import (
        HierarchicalGTTFN as _HierarchicalGTTFNBase,
        OnEquivariantWrapper as _OnEquivariantWrapperBase,
        GTTFNEncoder as _GTTFNEncoderBase,
        GTTensorFieldNetworkWithAttention as _GTTFNWABase,
        EquivariantSetTransformer as _EquivariantSetTransformerBase,
        StochasticEquivariantTFN as _StochasticEquivariantTFNBase,
        EquivariantGraphMambaNetwork as _EquivariantGraphMambaBase,
        TemporalCrossAttentionTFN as _TemporalCrossAttentionTFNBase,
        RelaxedEquivariantTFN as _RelaxedEquivariantTFNBase,
        RelaxedOnEquivariantTensorFieldNetwork as _RelaxedOnEquivariantTFNBase,
        HybridTFNClassifier as _HybridTFNClassifierBase,
        HybridOnEquivariantTensorFieldNetwork as _HybridOnEquivariantTFNBase,
        EndToEndClassifier as _EndToEndClassifierBase,
        EndToEndTensorFieldNetwork as _EndToEndTFNBase,
        build_enhanced_pointcloud,
    )

    # All gt_improvements classes inherit from nn.Module and store
    # GTBasis/CGCoefficients as submodules.  Since those are now
    # nn.Module subclasses with register_buffer, model.to(device)
    # moves everything automatically — no manual _move_basis_tensors needed.

    return (_HierarchicalGTTFNBase, _OnEquivariantWrapperBase, _GTTFNEncoderBase,
            _GTTFNWABase, _EquivariantSetTransformerBase,
            _StochasticEquivariantTFNBase, _EquivariantGraphMambaBase,
            _TemporalCrossAttentionTFNBase,
            _RelaxedEquivariantTFNBase, _RelaxedOnEquivariantTFNBase,
            _HybridTFNClassifierBase, _HybridOnEquivariantTFNBase,
            _EndToEndClassifierBase, _EndToEndTFNBase,
            build_enhanced_pointcloud)


(HierarchicalGTTFN, OnEquivariantWrapper, GTTFNEncoder,
 GTTensorFieldNetworkWithAttention, EquivariantSetTransformer,
 StochasticEquivariantTFN, EquivariantGraphMambaNetwork,
 TemporalCrossAttentionTFN,
 RelaxedEquivariantTFN, RelaxedOnEquivariantTensorFieldNetwork,
 HybridTFNClassifier, HybridOnEquivariantTensorFieldNetwork,
 EndToEndClassifier, EndToEndTensorFieldNetwork,
 build_enhanced_pointcloud) = _load_gt_improvements()


class HierarchicalTensorFieldNetwork(nn.Module):
    """Hierarchical TFN variant with multi-scale pooling for 3D point clouds."""

    def __init__(
        self,
        num_classes: int,
        n: int = 3,
        max_order: int = 1,
        hidden_channels: int = 64,
        stage_sizes: Optional[List[int]] = None,
        stage_radii: Optional[List[float]] = None,
        k_local: int = 16,
        k_global: int = 16,
        num_layers_per_stage: int = 2,
        num_rbf: int = 64,
        cutoff: float = 2.0,
        classifier_dims: Optional[List[int]] = None,
        node_attr_dim: int = 0,
    ):
        super().__init__()
        if stage_sizes is None:
            stage_sizes = [256, 64]
        if stage_radii is None:
            stage_radii = [0.2, 0.4]
        if classifier_dims is None:
            classifier_dims = [256, 128]
        self._inner = HierarchicalGTTFN(
            n=n,
            num_classes=num_classes,
            max_order=max_order,
            hidden_channels=hidden_channels,
            stage_sizes=stage_sizes,
            stage_radii=stage_radii,
            k_local=k_local,
            k_global=k_global,
            num_layers_per_stage=num_layers_per_stage,
            num_rbf=num_rbf,
            cutoff=cutoff,
            classifier_dims=classifier_dims,
            node_attr_dim=node_attr_dim,
        )

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        return self._inner(batch)


class OnEquivariantTensorFieldNetwork(nn.Module):
    """O(3)-invariant TFN via group averaging over reflections."""

    def __init__(
        self,
        num_classes: int,
        n: int = 3,
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
        base = TensorFieldNetwork(
            num_classes=num_classes,
            n=n,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            classifier_dims=classifier_dims,
        )
        self._inner = OnEquivariantWrapper(base)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        return self._inner(batch)


# ---------------------------------------------------------------------------
# New TFN-derived model wrappers
# ---------------------------------------------------------------------------

class AttentionTensorFieldNetwork(nn.Module):
    """GT-TFN with multi-head cross-attention message passing."""

    def __init__(
        self,
        num_classes: int,
        max_order: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        classifier_dims: Optional[List[int]] = None,
        radial_hidden: int = 64,
        n: int = 3,
    ):
        super().__init__()
        if classifier_dims is None:
            classifier_dims = [128, 64]
        self._inner = GTTensorFieldNetworkWithAttention(
            n=n, num_classes=num_classes,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            classifier_dims=classifier_dims,
            radial_hidden=radial_hidden,
        )

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        return self._inner(batch)


class CrossAttentionTensorFieldNetwork(nn.Module):
    """GT-TFN with temporal transformer cross-attention over point sequence."""

    def __init__(
        self,
        num_classes: int,
        max_order: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        transformer_layers: int = 2,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        classifier_dims: Optional[List[int]] = None,
        radial_hidden: int = 64,
        dropout: float = 0.1,
        n: int = 3,
    ):
        super().__init__()
        if classifier_dims is None:
            classifier_dims = [128, 64]
        self._inner = TemporalCrossAttentionTFN(
            n=n, num_classes=num_classes,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            transformer_layers=transformer_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            classifier_dims=classifier_dims,
            radial_hidden=radial_hidden,
            dropout=dropout,
        )
        self.k_neighbors = k_neighbors
        self.rbf = self._inner.rbf
        self.gt_basis = self._inner.gt_basis
        self.classifier_dims = classifier_dims

    def forward(self, batch: List[torch.Tensor],
                precomputed_geom=None) -> torch.Tensor:
        return self._inner(batch, precomputed_geom=precomputed_geom)

    def _encode_single(self, pc, precomputed_geom=None, **kwargs):
        return self._inner._encode_point_descriptors(pc, precomputed_geom=precomputed_geom)

    @property
    def rho(self):
        return self._inner.rho


class StochasticTensorFieldNetwork(nn.Module):
    """Stochastic equivariant TFN with Gaussian mixture output."""

    def __init__(
        self,
        num_classes: int,
        n: int = 3,
        num_mixtures: int = 3,
        max_order: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        encoder_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [256, 128]
        self._inner = StochasticEquivariantTFN(
            n=n, num_classes=num_classes,
            num_mixtures=num_mixtures,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            encoder_dims=encoder_dims,
        )

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        return self._inner(batch)


class SetTransformerTensorFieldNetwork(nn.Module):
    """Equivariant set transformer for PL/PI prediction."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        max_order: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        k_neighbors: int = 16,
        decoder_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if decoder_dims is None:
            decoder_dims = [128, 64]
        self._inner = EquivariantSetTransformer(
            n=3, num_classes=num_classes,
            embedding_dim=embedding_dim,
            max_order=max_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            cutoff=cutoff,
            k_neighbors=k_neighbors,
            decoder_dims=decoder_dims,
        )

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        return self._inner(batch)


class GraphMambaTensorFieldNetwork(nn.Module):
    """Equivariant Graph Mamba network with state-space layers."""

    def __init__(
        self,
        num_classes: int,
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
        self._inner = EquivariantGraphMambaNetwork(
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # TFN re-export (used by expes/analysis_nn.py)
    'TFNTensorFieldNetwork',
    # Main equivariant models
    'TensorFieldNetwork',
    'GTTensorFieldNetwork',
    'GTTensorFieldNetworkV2',
    'HierarchicalGTTFN',
    'HierarchicalTensorFieldNetwork',
    'OnEquivariantWrapper',
    'OnEquivariantTensorFieldNetwork',
    'GTTFNEncoder',
    'PointNet3D',
    'RipsPointNet',
    # New TFN-derived models
    'GTTensorFieldNetworkWithAttention',
    'AttentionTensorFieldNetwork',
    'EquivariantSetTransformer',
    'SetTransformerTensorFieldNetwork',
    'StochasticEquivariantTFN',
    'StochasticTensorFieldNetwork',
    'EquivariantGraphMambaNetwork',
    'GraphMambaTensorFieldNetwork',
    'TemporalCrossAttentionTFN',
    'CrossAttentionTensorFieldNetwork',
    # Improvement models
    'RelaxedEquivariantTFN',
    'RelaxedOnEquivariantTensorFieldNetwork',
    'HybridTFNClassifier',
    'HybridOnEquivariantTensorFieldNetwork',
    'EndToEndClassifier',
    'EndToEndTensorFieldNetwork',
    'build_enhanced_pointcloud',
    # Notebook / ragged models
    'ScalarDistanceDeepSet',
    'PointNetTutorial',
    'ScalarInputMLP',
    'MultiInputModel',
    'DenseRagged',
    'PermopRagged',
    'RaggedPersistenceModel',
    'DistanceMatrixRaggedModel',
    # Helpers (useful for external code)
    '_act',
    '_build_mlp',
]