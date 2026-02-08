import torch
import torch.nn as nn

class DenseRagged(nn.Module):
    def __init__(self, units, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__()
        self.units = units
        self.use_bias = use_bias
        self.activation_name = activation
        
        # Activation function mapping
        activation_map = {
            'linear': nn.Identity(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
        }
        self.activation = activation_map.get(activation, nn.Identity())
    
    def build(self, input_shape):
        last_dim = input_shape[-1] if isinstance(input_shape, (list, tuple)) else input_shape
        self.kernel = nn.Parameter(torch.randn(last_dim, self.units) * 0.01)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.units))
        else:
            self.bias = None
    
    def forward(self, inputs):
        # Handle variable length sequences by padding
        if isinstance(inputs, list):
            # If inputs is a list of tensors with different lengths, pad them
            max_length = max([x.shape[0] for x in inputs])
            padded_inputs = []
            for x in inputs:
                if x.shape[0] < max_length:
                    padding = torch.zeros(max_length - x.shape[0], x.shape[1], device=x.device)
                    x = torch.cat([x, padding], dim=0)
                padded_inputs.append(x)
            inputs = torch.stack(padded_inputs, dim=0)
        
        # Apply dense layer
        outputs = torch.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias
        outputs = self.activation(outputs)
        return outputs

class PermopRagged(nn.Module):
    def __init__(self, **kwargs):
        super(PermopRagged, self).__init__()
    
    def forward(self, inputs):
        # Sum over the sequence dimension (axis 1)
        out = torch.sum(inputs, dim=1)
        return out
