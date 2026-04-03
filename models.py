"""Central model registry for the repository.

This module re-exports the main model class definitions distributed in:
- TFN.py, tfn_model.py, gt_tfn_layer.py
- utils.py, experiment_augmentation.py, expes/utils.py, code_3D_shapes/helper_fctns/create_ripsnet.py
- code_3D_shapes/helper_fctns/create_pointnet.py
- expes/analysis_nn.py, expes/train_nn.py

Notebook model definitions (ported from tutorial_pytorch_ragged.ipynb):
- ScalarDistanceDeepSet
- PointNetTutorial (alias from notebook PointNet)
- ScalarInputMLP
- MultiInputModel
- DenseRagged
- PermopRagged
- RaggedPersistenceModel
- DistanceMatrixRaggedModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List

from TFN import RBFExpansion as TFN_RBFExpansion, TFNLayer, TensorFieldNetwork as TFNTensorFieldNetwork
from gt_tfn_layer import RBFExpansion as GTTFN_RBFExpansion, GTTFNLayer, GTTensorFieldNetwork, ChannelMixer, EquivariantGate, ResidualProjection
from gt_improvements import HierarchicalGTTFN, OnEquivariantWrapper

# Notebook-derived models: reimplemented below (from tutorial_pytorch_ragged.ipynb)

# Only TFN and notebook models are provided per request.

# ---------------------------------------------------------------------------
# notebook models (reimplemented here for import via models.py)
# ---------------------------------------------------------------------------

class ScalarDistanceDeepSet(nn.Module):
    def __init__(self, output_dim, phi_dims=[64, 128], rho_dims=[256, 128]):
        super().__init__()

        phi_layers = []
        in_features = 1
        for dim in phi_dims:
            phi_layers.append(nn.Linear(in_features, dim))
            phi_layers.append(nn.ReLU())
            in_features = dim
        self.phi_layers = nn.Sequential(*phi_layers)

        rho_layers = []
        in_features = phi_dims[-1]
        for dim in rho_dims:
            rho_layers.append(nn.Linear(in_features, dim))
            rho_layers.append(nn.ReLU())
            in_features = dim
        rho_layers.append(nn.Linear(in_features, output_dim))
        self.rho_layers = nn.Sequential(*rho_layers)

    def forward(self, batch: list[torch.Tensor]):
        if len(batch) == 0:
            return torch.empty(0, self.rho_layers[-1].out_features, device=next(self.parameters()).device)

        all_features = []
        for dm in batch:
            if dm.ndim == 2 and dm.shape[0] > 0:
                upper_triangle_indices = torch.triu_indices(dm.shape[0], dm.shape[1], offset=1)
                scalar_distances = dm[upper_triangle_indices[0], upper_triangle_indices[1]]
            elif dm.ndim == 1 and dm.shape[0] > 0:
                scalar_distances = dm
            else:
                scalar_distances = torch.zeros(1, dtype=torch.float32, device=dm.device)

            scalar_distances = scalar_distances.unsqueeze(1)
            per_distance_features = self.phi_layers(scalar_distances)
            aggregated_features = torch.sum(per_distance_features, dim=0)
            all_features.append(aggregated_features)

        if len(all_features) > 0:
            stacked_features = torch.stack(all_features)
            output = self.rho_layers(stacked_features)
        else:
            output = torch.empty(0, self.rho_layers[-1].out_features, device=next(self.parameters()).device)

        return output


class PointNetTutorial(nn.Module):
    def __init__(self, output_dim, phi_dims=[64, 128, 256], rho_dims=[256, 128]):
        super().__init__()

        phi_layers = []
        in_features = 2
        for dim in phi_dims:
            phi_layers.append(nn.Linear(in_features, dim))
            phi_layers.append(nn.ReLU())
            in_features = dim
        self.phi_layers = nn.Sequential(*phi_layers)

        rho_layers = []
        in_features = phi_dims[-1]
        for dim in rho_dims:
            rho_layers.append(nn.Linear(in_features, dim))
            rho_layers.append(nn.ReLU())
            in_features = dim
        rho_layers.append(nn.Linear(in_features, output_dim))
        self.rho_layers = nn.Sequential(*rho_layers)

    def forward(self, batch: List[torch.Tensor]):
        if len(batch) == 0:
            return torch.empty(0, self.rho_layers[-1].out_features, device=next(self.parameters()).device)

        padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)
        mask = torch.zeros(padded_batch.shape[0], padded_batch.shape[1], dtype=torch.bool, device=padded_batch.device)
        for i, pc in enumerate(batch):
            mask[i, :len(pc)] = True

        phi_out = self.phi_layers(padded_batch.view(-1, padded_batch.shape[-1]))
        phi_out = phi_out.view(padded_batch.shape[0], padded_batch.shape[1], -1)

        min_val = torch.finfo(phi_out.dtype).min
        masked_phi_out = phi_out.masked_fill(~mask.unsqueeze(-1), min_val)

        aggregated_features, _ = torch.max(masked_phi_out, dim=1)

        output = self.rho_layers(aggregated_features)
        return output


class ScalarInputMLP(nn.Module):
    def __init__(self, output_dim, hidden_dims=[256, 512, 1024]):
        super().__init__()
        layers = []
        input_dim = 1
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
        elif x.ndim == 0:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.ndim == 1:
            x = x.unsqueeze(1)

        return self.mlp(x)


class MultiInputModel(nn.Module):
    def __init__(self, target_output_dim, scalar_input_dim, pointnet_intermediate_dim=128, scalar_mlp_intermediate_dim=128):
        super().__init__()

        self.pointnet_branch = PointNetTutorial(output_dim=pointnet_intermediate_dim, phi_dims=[64, 128, 256], rho_dims=[256, 128])

        self.scalar_mlp_branch = nn.Sequential(
            nn.Linear(scalar_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, scalar_mlp_intermediate_dim),
            nn.ReLU()
        )

        combined_input_dim = pointnet_intermediate_dim + scalar_mlp_intermediate_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, target_output_dim),
            nn.Sigmoid()
        )

    def forward(self, point_clouds: List[torch.Tensor], scalar_input_features: torch.Tensor):
        pointnet_features = self.pointnet_branch(point_clouds)
        if scalar_input_features.ndim == 1:
            scalar_input_features = scalar_input_features.unsqueeze(0)

        scalar_mlp_features = self.scalar_mlp_branch(scalar_input_features)
        combined_features = torch.cat((pointnet_features, scalar_mlp_features), dim=1)
        output = self.final_mlp(combined_features)
        return output


class DenseRagged(nn.Module):
    def __init__(self, in_features=None, out_features=30, activation='relu', use_bias=True):
        super(DenseRagged, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation

        self.weight_param = None
        self.bias_param = None

    def forward(self, inputs):
        outputs = []
        for x in inputs:
            if self.weight_param is None:
                in_features = x.shape[-1]
                self.weight_param = nn.Parameter(torch.randn(in_features, self.out_features, device=x.device) * 0.01)
                self.register_parameter('weight_param', self.weight_param)
                if self.use_bias:
                    self.bias_param = nn.Parameter(torch.zeros(self.out_features, device=x.device))
                    self.register_parameter('bias_param', self.bias_param)

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
    def forward(self, inputs):
        return torch.stack([torch.sum(x, dim=0) for x in inputs])


class RaggedPersistenceModel(nn.Module):
    def __init__(self, output_dim):
        super(RaggedPersistenceModel, self).__init__()
        self.ragged_layers = nn.ModuleList([
            DenseRagged(out_features=30, activation='relu'),
            DenseRagged(out_features=20, activation='relu'),
            DenseRagged(out_features=10, activation='relu')
        ])
        self.perm = PermopRagged()

        self.fc = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        for layer in self.ragged_layers:
            x = layer(x)
        x = self.perm(x)
        x = self.fc(x)
        return x


class DistanceMatrixRaggedModel(nn.Module):
    def __init__(self, output_dim, num_points=None, phi_dim=128, rho_hidden=(256,128)):
        super().__init__()
        self.num_points = num_points
        inp = num_points if num_points is not None else 0
        self._phi_layers = None
        self.phi_dim = phi_dim
        self._build_phi(inp)
        layers = []
        prev = phi_dim
        for h in rho_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.rho = nn.Sequential(*layers)

    def _build_phi(self, inp):
        if inp <= 0:
            self._phi_layers = None
            return
        hidden = max(64, self.phi_dim)
        self._phi_layers = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.phi_dim),
            nn.ReLU()
        )

    def forward(self, batch: List[torch.Tensor]):
        if len(batch) == 0:
            return torch.empty(0, self.rho[-1].out_features, device=next(self.parameters()).device)

        sizes = [m.shape[0] for m in batch]
        max_n = max(sizes)
        device = next(self.parameters()).device

        if self._phi_layers is None or (self.num_points and self.num_points != max_n):
            self._build_phi(max_n)
            self.num_points = max_n
            self._phi_layers = self._phi_layers.to(device)

        phi_outs = []
        for m in batch:
            if m.shape[0] < max_n:
                padded = torch.zeros(max_n, max_n, device=device)
                padded[:m.shape[0], :m.shape[1]] = m
                m = padded
            phi_out = self._phi_layers(m.view(1, -1)).squeeze(0)
            phi_outs.append(phi_out)

        aggregated = torch.stack(phi_outs)
        return self.rho(aggregated)


# Canonical alias names for widespread use
TensorFieldNetwork = TFNTensorFieldNetwork

__all__ = [
    # TFN
    'TFN_RBFExpansion', 'TFNLayer', 'TensorFieldNetwork',
    # gttfn
    'GTTFN_RBFExpansion', 'GTTFNLayer', 'GTTensorFieldNetwork', 'ChannelMixer', 'EquivariantGate', 'ResidualProjection',
    # gt improvements
    'HierarchicalGTTFN', 'OnEquivariantWrapper',
    # notebook models
    'ScalarDistanceDeepSet', 'PointNetTutorial', 'ScalarInputMLP', 'MultiInputModel',
    'DenseRagged', 'PermopRagged', 'RaggedPersistenceModel', 'DistanceMatrixRaggedModel',
]
