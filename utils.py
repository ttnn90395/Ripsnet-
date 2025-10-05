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
        self.kernel = nn.Parameter(torch.randn(input_dim, self.units) * 0.01)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.units))
        else:
            self.bias = None

    def forward(self, inputs):
        if self.kernel is None:
            self.build(inputs.shape[-1])

        outputs = torch.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias

        if self.activation == 'relu':
            outputs = F.relu(outputs)
        elif self.activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self.activation == 'tanh':
            outputs = torch.tanh(outputs)
        elif self.activation == 'linear' or self.activation is None:
            pass
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return outputs


class PermopRagged(nn.Module):
    def __init__(self):
        super(PermopRagged, self).__init__()

    def forward(self, inputs):
        return torch.sum(inputs, dim=1)
