# Functions to create the RipsNet architecture.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseRagged(nn.Module):
    def __init__(self, units, use_bias=True, activation='linear'):
        super(DenseRagged, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.kernel = None
        self.bias = None

    def build(self, input_dim):
        self.kernel = nn.Parameter(torch.randn(input_dim, self.units) * 0.1)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.units))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        """
        Inputs can be:
        - a list of tensors with shape (Ni, D)
        - or a padded tensor of shape (B, N, D)
        """
        if isinstance(inputs, list):  # emulate ragged behavior
            outputs = []
            for x in inputs:
                if self.kernel is None:
                    self.build(x.size(-1))
                out = x @ self.kernel
                if self.use_bias:
                    out = out + self.bias
                out = self.apply_activation(out)
                outputs.append(out)
            return outputs
        else:
            if self.kernel is None:
                self.build(inputs.size(-1))
            out = torch.matmul(inputs, self.kernel)
            if self.use_bias:
                out = out + self.bias
            out = self.apply_activation(out)
            return out

    def apply_activation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        else:
            return x


class PermopRagged(nn.Module):
    def __init__(self):
        super(PermopRagged, self).__init__()

    def forward(self, inputs):
        """
        Mean (or sum) pooling along the 'ragged' dimension.
        """
        if isinstance(inputs, list):
            outputs = [torch.mean(x, dim=0, keepdim=False) for x in inputs]
            return torch.stack(outputs, dim=0)
        else:
            return torch.mean(inputs, dim=1)


def create_ripsnet(
    input_dimension,
    ragged_layers=[30, 20, 10],
    dense_layers=[64, 128, 256],
    output_units=2500,
    activation_fct='relu',
    output_activation='sigmoid',
    dropout=0,
    kernel_regularization=0
):
    class RipsNet(nn.Module):
        def __init__(self):
            super(RipsNet, self).__init__()
            self.ragged_layers = nn.ModuleList()
            in_dim = input_dimension
            for units in ragged_layers:
                layer = DenseRagged(units, use_bias=True, activation=activation_fct)
                layer.build(in_dim)
                self.ragged_layers.append(layer)
                in_dim = units

            self.permop = PermopRagged()

            dense_list = []
            for n_units in dense_layers:
                dense_list.append(nn.Linear(in_dim, n_units))
                dense_list.append(nn.ReLU() if activation_fct == 'relu' else nn.Identity())
                if dropout > 0:
                    dense_list.append(nn.Dropout(dropout))
                in_dim = n_units
            self.dense_layers = nn.Sequential(*dense_list)

            self.output_layer = nn.Linear(in_dim, output_units)
            self.output_activation = output_activation

        def forward(self, inputs):
            x = inputs
            for layer in self.ragged_layers:
                x = layer(x)
            x = self.permop(x)
            x = self.dense_layers(x)
            x = self.output_layer(x)

            if self.output_activation == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.output_activation == 'relu':
                x = F.relu(x)
            elif self.output_activation == 'tanh':
                x = torch.tanh(x)
            return x

    model = RipsNet()
    return model

