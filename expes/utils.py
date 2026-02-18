import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseRagged(nn.Module):
    """
    Dense layer that handles ragged tensors (lists of variable-length sequences).
    Compatible with both list and padded tensor inputs.
    """
    def __init__(self, units, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.kernel = None
        self.bias = None

    def build(self, input_dim):
        """Initialize weights after input dimension is known."""
        self.kernel = nn.Parameter(torch.randn(input_dim, self.units) * 0.1)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.units))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        """
        Forward pass supporting ragged tensors.
        Args:
            inputs: List of tensors with shape (Ni, D) or padded tensor (B, N, D)
        """
        if isinstance(inputs, list):  # Handle ragged/variable-length sequences
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
        else:  # Handle padded tensors
            if self.kernel is None:
                self.build(inputs.size(-1))
            out = torch.matmul(inputs, self.kernel)
            if self.use_bias:
                out = out + self.bias
            out = self.apply_activation(out)
            return out

    def apply_activation(self, x):
        """Apply activation function."""
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        else:  # 'linear' or default
            return x


class PermopRagged(nn.Module):
    """
    Permutation-invariant pooling (mean pooling) along the ragged dimension.
    Handles both ragged (list) and padded tensor inputs.
    """
    def __init__(self, **kwargs):
        super(PermopRagged, self).__init__()
    
    def forward(self, inputs):
        """
        Apply mean pooling along the sequence dimension.
        Args:
            inputs: List of tensors or padded tensor of shape (B, N, D)
        Returns:
            Pooled output of shape (B, D) or stacked tensor
        """
        if isinstance(inputs, list):
            # Mean pooling for each sequence in the list
            outputs = [torch.mean(x, dim=0, keepdim=False) for x in inputs]
            return torch.stack(outputs, dim=0)
        else:
            # Mean pooling along sequence dimension
            return torch.mean(inputs, dim=1)
