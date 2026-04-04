"""Central model registry for the repository.

Import structure (no circular imports):
  TFN.py            → TFNLayer, TFN_RBFExpansion, TFNTensorFieldNetwork
  gt_tfn_layer.py   → GTTFNLayer, GTTFN_RBFExpansion, GTTensorFieldNetwork,
                       ChannelMixer, EquivariantGate, ResidualProjection,
                       HierarchicalGTTFN, OnEquivariantWrapper
  THIS file         → re-exports all of the above + notebook models
                       + GTTensorFieldNetworkV2, PointNet3D

  tfn_model.py imports FROM this file (one-way), so it must NOT be imported here.
  GTTensorFieldNetworkV2 and PointNet3D are defined directly below to avoid the
  circular dependency while still being available to the rest of the codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List

# ---------------------------------------------------------------------------
# Low-level building blocks (no circular deps)
# ---------------------------------------------------------------------------
from TFN import (
    RBFExpansion as TFN_RBFExpansion,
    TFNLayer,
    TensorFieldNetwork as TFNTensorFieldNetwork,   # original SO(3) TFN
)
from gt_tfn_layer import (
    RBFExpansion as GTTFN_RBFExpansion,
    GTTFNLayer,
    ChannelMixer,
    EquivariantGate,
    ResidualProjection,
    GTTensorFieldNetwork,      # n-dim SO(n) base model
)

from gt_improvements import (
    HierarchicalGTTFN, OnEquivariantWrapper,
)

# ---------------------------------------------------------------------------
# GTTensorFieldNetworkV2  (defined here, not in tfn_model.py, to avoid the
#                          circular import that plagued the old layout)
# ---------------------------------------------------------------------------

class GTTensorFieldNetworkV2(GTTensorFieldNetwork):
    """
    Recommended GT-TFN with all improvements ON by default.

    Inherits GTTensorFieldNetwork (gt_tfn_layer.py) and exposes all
    hyper-parameters at the top level for easy configuration.

    Example
    -------
    model = GTTensorFieldNetworkV2(n=3, num_classes=40)
    logits = model(batch)                          # List[Tensor(N_i, 3)]
    logits = model(batch, node_attrs=attrs)        # with node features
    """

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


# ---------------------------------------------------------------------------
# TensorFieldNetwork  — SE(3) model backed by GTTFNv2
#   NOTE: this shadows TFNTensorFieldNetwork intentionally.
#   Use TFNTensorFieldNetwork explicitly if you need the original TFN.
# ---------------------------------------------------------------------------

class TensorFieldNetwork(nn.Module):
    """
    SE(3)-equivariant point cloud model.
    Same external interface as the original TFN; internally uses GTTFNv2.
    forward(batch: List[Tensor(N_i, 3)]) → Tensor(B, num_classes)
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


# ---------------------------------------------------------------------------
# PointNet3D  (3-D Deep-Sets baseline; no rotational equivariance)
# ---------------------------------------------------------------------------

class PointNet3D(nn.Module):
    """
    Deep-Sets / PointNet baseline — permutation-invariant, NOT equivariant.
    forward(batch: List[Tensor(N_i, 3)]) → Tensor(B, output_dim)
    """

    def __init__(self, output_dim, phi_dims=(64, 128, 256), rho_dims=(256, 128)):
        super().__init__()
        phi_layers, in_f = [], 3
        for d in phi_dims:
            phi_layers += [nn.Linear(in_f, d), nn.ReLU()]
            in_f = d
        self.phi_layers = nn.Sequential(*phi_layers)

        rho_layers, in_f = [], phi_dims[-1]
        for d in rho_dims:
            rho_layers += [nn.Linear(in_f, d), nn.ReLU()]
            in_f = d
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
        phi = self.phi_layers(padded.reshape(-1, 3)).reshape(*padded.shape[:2], -1)
        phi = phi.masked_fill(~mask.unsqueeze(-1), torch.finfo(phi.dtype).min)
        agg, _ = phi.max(dim=1)
        return self.rho_layers(agg)


# ---------------------------------------------------------------------------
# Notebook-derived models
# ---------------------------------------------------------------------------

class ScalarDistanceDeepSet(nn.Module):
    def __init__(self, output_dim, phi_dims=(64, 128), rho_dims=(256, 128)):
        super().__init__()
        phi_layers, in_f = [], 1
        for d in phi_dims:
            phi_layers += [nn.Linear(in_f, d), nn.ReLU()]
            in_f = d
        self.phi_layers = nn.Sequential(*phi_layers)

        rho_layers, in_f = [], phi_dims[-1]
        for d in rho_dims:
            rho_layers += [nn.Linear(in_f, d), nn.ReLU()]
            in_f = d
        rho_layers.append(nn.Linear(in_f, output_dim))
        self.rho_layers = nn.Sequential(*rho_layers)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.rho_layers[-1].out_features,
                               device=next(self.parameters()).device)
        all_features = []
        for dm in batch:
            if dm.ndim == 2 and dm.shape[0] > 0:
                idx = torch.triu_indices(dm.shape[0], dm.shape[1], offset=1)
                scalars = dm[idx[0], idx[1]]
            elif dm.ndim == 1 and dm.shape[0] > 0:
                scalars = dm
            else:
                scalars = torch.zeros(1, dtype=torch.float32, device=dm.device)
            phi = self.phi_layers(scalars.unsqueeze(1))
            all_features.append(phi.sum(dim=0))
        return self.rho_layers(torch.stack(all_features))


class PointNetTutorial(nn.Module):
    """2-D PointNet; input tensors must have at least 2 columns."""

    def __init__(self, output_dim, phi_dims=(64, 128, 256), rho_dims=(256, 128)):
        super().__init__()
        phi_layers, in_f = [], 2
        for d in phi_dims:
            phi_layers += [nn.Linear(in_f, d), nn.ReLU()]
            in_f = d
        self.phi_layers = nn.Sequential(*phi_layers)

        rho_layers, in_f = [], phi_dims[-1]
        for d in rho_dims:
            rho_layers += [nn.Linear(in_f, d), nn.ReLU()]
            in_f = d
        rho_layers.append(nn.Linear(in_f, output_dim))
        self.rho_layers = nn.Sequential(*rho_layers)

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.rho_layers[-1].out_features,
                               device=next(self.parameters()).device)
        # keep only first 2 columns
        batch2 = [x[:, :2] for x in batch]
        padded = pad_sequence(batch2, batch_first=True, padding_value=0.0)
        mask   = torch.zeros(padded.shape[:2], dtype=torch.bool, device=padded.device)
        for i, pc in enumerate(batch2):
            mask[i, :len(pc)] = True
        phi = self.phi_layers(padded.reshape(-1, 2)).reshape(*padded.shape[:2], -1)
        phi = phi.masked_fill(~mask.unsqueeze(-1), torch.finfo(phi.dtype).min)
        agg, _ = phi.max(dim=1)
        return self.rho_layers(agg)


class ScalarInputMLP(nn.Module):
    """Expects a (B, 1) scalar tensor (mean pairwise distance)."""

    def __init__(self, output_dim, hidden_dims=(256, 512, 1024)):
        super().__init__()
        layers, in_f = [], 1
        for d in hidden_dims:
            layers += [nn.Linear(in_f, d), nn.ReLU()]
            in_f = d
        layers += [nn.Linear(in_f, output_dim), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.stack(x)
        if x.ndim == 0:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.ndim == 1:
            x = x.unsqueeze(1)
        # x is (B, 1) here
        return self.mlp(x)


class MultiInputModel(nn.Module):
    """Combines a PointNetTutorial branch with a scalar MLP branch."""

    def __init__(self, target_output_dim, scalar_input_dim,
                 pointnet_intermediate_dim=128, scalar_mlp_intermediate_dim=128):
        super().__init__()
        self.pointnet_branch = PointNetTutorial(
            output_dim=pointnet_intermediate_dim,
            phi_dims=(64, 128, 256), rho_dims=(256, 128),
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),             nn.ReLU(),
            nn.Linear(256, scalar_mlp_intermediate_dim), nn.ReLU(),
        )
        combined = pointnet_intermediate_dim + scalar_mlp_intermediate_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(combined, 256), nn.ReLU(),
            nn.Linear(256, 128),     nn.ReLU(),
            nn.Linear(128, target_output_dim), nn.Sigmoid(),
        )

    def forward(self, point_clouds: List[torch.Tensor],
                scalar_input: torch.Tensor) -> torch.Tensor:
        pc_feat = self.pointnet_branch(point_clouds)
        if scalar_input.ndim == 1:
            scalar_input = scalar_input.unsqueeze(0)
        sc_feat = self.scalar_branch(scalar_input)
        return self.final_mlp(torch.cat([pc_feat, sc_feat], dim=1))


class DenseRagged(nn.Module):
    """Point-wise linear layer for ragged (variable-size) point clouds."""

    def __init__(self, in_features=None, out_features=30,
                 activation='relu', use_bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.use_bias     = use_bias
        self.activation   = activation
        self.weight_param = None
        self.bias_param   = None

    def _init_params(self, in_f: int, device):
        self.weight_param = nn.Parameter(
            torch.randn(in_f, self.out_features, device=device) * 0.01)
        self.register_parameter('weight_param', self.weight_param)
        if self.use_bias:
            self.bias_param = nn.Parameter(
                torch.zeros(self.out_features, device=device))
            self.register_parameter('bias_param', self.bias_param)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for x in inputs:
            if self.weight_param is None:
                self._init_params(x.shape[-1], x.device)
            y = torch.matmul(x, self.weight_param)
            if self.use_bias:
                y = y + self.bias_param
            if self.activation == 'relu':
                y = F.relu(y)
            elif self.activation == 'sigmoid':
                y = torch.sigmoid(y)
            elif self.activation == 'tanh':
                y = torch.tanh(y)
            outputs.append(y)
        return outputs


class PermopRagged(nn.Module):
    """Permutation-invariant pooling: sum over points → (B, C)."""

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack([x.sum(dim=0) for x in inputs])


class RaggedPersistenceModel(nn.Module):
    """Three DenseRagged layers → PermopRagged → MLP head."""

    def __init__(self, output_dim):
        super().__init__()
        self.ragged_layers = nn.ModuleList([
            DenseRagged(out_features=30, activation='relu'),
            DenseRagged(out_features=20, activation='relu'),
            DenseRagged(out_features=10, activation='relu'),
        ])
        self.perm = PermopRagged()
        self.fc = nn.Sequential(
            nn.Linear(10, 50),  nn.ReLU(),
            nn.Linear(50, 100), nn.ReLU(),
            nn.Linear(100, output_dim),
        )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        for layer in self.ragged_layers:
            x = layer(x)
        return self.fc(self.perm(x))


class DistanceMatrixRaggedModel(nn.Module):
    """
    Row-wise MLP on a distance matrix, then mean-pool and MLP head.
    `num_points` is used only as an initial hint; the phi network is
    rebuilt lazily if a different size is encountered.
    """

    def __init__(self, output_dim, num_points=None,
                 phi_dim=128, rho_hidden=(256, 128)):
        super().__init__()
        self.num_points   = num_points
        self.phi_dim      = phi_dim
        self._phi_layers  = None
        self._phi_inp_dim = None
        if num_points and num_points > 0:
            self._build_phi(num_points)
        layers, prev = [], phi_dim
        for h in rho_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.rho = nn.Sequential(*layers)

    def _build_phi(self, inp: int):
        hidden = max(64, self.phi_dim)
        self._phi_layers = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, self.phi_dim), nn.ReLU(),
        )
        self._phi_inp_dim = inp

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0, self.rho[-1].out_features,
                               device=next(self.parameters()).device)
        max_n  = max(m.shape[0] for m in batch)
        device = next(self.parameters()).device

        if self._phi_layers is None or self._phi_inp_dim != max_n:
            self._build_phi(max_n)
            self._phi_layers = self._phi_layers.to(device)

        phi_outs = []
        for m in batch:
            n = m.shape[0]
            if n < max_n:
                padded = torch.zeros(max_n, max_n, device=device)
                padded[:n, :n] = m
                m = padded
            row_phi = self._phi_layers(m)          # (max_n, phi_dim)
            phi_outs.append(row_phi.mean(dim=0))   # (phi_dim,)

        return self.rho(torch.stack(phi_outs))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Low-level TFN blocks
    'TFN_RBFExpansion', 'TFNLayer', 'TFNTensorFieldNetwork',
    # Low-level GT blocks
    'GTTFN_RBFExpansion', 'GTTFNLayer',
    'ChannelMixer', 'EquivariantGate', 'ResidualProjection',
    # Main equivariant models
    'TensorFieldNetwork',          # SE(3), backed by GTTFNv2
    'GTTensorFieldNetwork',        # SO(n) base
    'GTTensorFieldNetworkV2',      # SO(n) with all improvements
    'HierarchicalGTTFN',
    'OnEquivariantWrapper',
    'PointNet3D',
    # Notebook / ragged models
    'ScalarDistanceDeepSet',
    'PointNetTutorial',
    'ScalarInputMLP',
    'MultiInputModel',
    'DenseRagged',
    'PermopRagged',
    'RaggedPersistenceModel',
    'DistanceMatrixRaggedModel',
]
