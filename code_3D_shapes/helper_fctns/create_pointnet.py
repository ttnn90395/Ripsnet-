import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
### Build the pointnet model
Each convolution and fully-connected layer (with exception for end layers) consists of
Convolution / Dense -> Batch Normalization -> ReLU Activation.
"""


def conv_bn(in_channels, filters):
    return nn.Sequential(
        nn.Conv1d(in_channels, filters, 1),
        nn.BatchNorm1d(filters, momentum=0.0),
        nn.ReLU()
    )


def dense_bn(in_features, filters):
    return nn.Sequential(
        nn.Linear(in_features, filters),
        nn.BatchNorm1d(filters, momentum=0.0),
        nn.ReLU()
    )

"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""


class OrthogonalRegularizer(nn.Module):
    def __init__(self, num_features, l2reg=0.001):
        super().__init__()
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = torch.eye(num_features)

    def forward(self, x):
        x = x.view(-1, self.num_features, self.num_features)
        xxt = torch.bmm(x, x.transpose(2, 1))
        diff = xxt - self.eye.to(x.device)
        return self.l2reg * torch.sum(diff ** 2)

"""
 We can then define a general function to build T-net layers.
"""

def tnet(inputs, num_features):
    batch_size = inputs.size(0)
    bias = torch.from_numpy(np.eye(num_features).flatten()).float().to(inputs.device)
    reg = OrthogonalRegularizer(num_features)

    conv1 = conv_bn(inputs.shape[1], 32)
    conv2 = conv_bn(32, 64)
    conv3 = conv_bn(64, 512)

    x = conv1(inputs)
    x = conv2(x)
    x = conv3(x)
    x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

    fc1 = dense_bn(512, 256)
    fc2 = dense_bn(256, 128)
    fc3 = nn.Linear(128, num_features * num_features)

    # initialize bias as identity
    nn.init.zeros_(fc3.weight)
    with torch.no_grad():
        fc3.bias.copy_(bias)

    x = fc1(x)
    x = fc2(x)
    x = fc3(x)
    feat_T = x.view(batch_size, num_features, num_features)

    # apply affine transformation
    transformed = torch.bmm(inputs.transpose(1, 2), feat_T).transpose(1, 2)
    return transformed, feat_T

"""
The main network can be then implemented in the same manner where the t-net mini models
can be dropped in a layers in the graph. Here we replicate the network architecture
published in the original paper but with half the number of weights at each layer as we
are using the smaller 10 class ModelNet dataset.
"""


def create_pointnet(NUM_CLASSES=10, NUM_POINTS=1024, BATCH_SIZE=32):
    class PointNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.NUM_CLASSES = NUM_CLASSES
            self.NUM_POINTS = NUM_POINTS
            self.BATCH_SIZE = BATCH_SIZE

            self.conv1 = conv_bn(3, 32)
            self.conv2 = conv_bn(32, 32)
            self.conv3 = conv_bn(32, 32)
            self.conv4 = conv_bn(32, 64)
            self.conv5 = conv_bn(64, 512)

            self.fc1 = dense_bn(512, 256)
            self.fc2 = dense_bn(256, 128)
            self.dropout = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, NUM_CLASSES)

        def forward(self, inputs):
            # inputs: (batch, num_points, 3)
            x = inputs.transpose(2, 1)

            x, trans_input = tnet(x, 3)
            x = self.conv1(x)
            x = self.conv2(x)
            x, trans_feat = tnet(x, 32)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            outputs = F.softmax(self.fc3(x), dim=1)
            return outputs, trans_input, trans_feat

    return PointNet()