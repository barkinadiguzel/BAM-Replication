import torch
import torch.nn as nn
from layers.conv_layers import conv1x1, conv3x3_dilated, bn, relu


class SpatialAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16, dilation=4):
        super().__init__()

        reduced_channels = channels // reduction_ratio

        self.conv_reduce = conv1x1(channels, reduced_channels)
        self.conv1 = conv3x3_dilated(reduced_channels, reduced_channels, dilation=dilation)
        self.conv2 = conv3x3_dilated(reduced_channels, reduced_channels, dilation=dilation)
        self.conv_expand = conv1x1(reduced_channels, 1)

        self.bn = bn(1)
        self.relu = relu()

    def forward(self, x):
        out = self.conv_reduce(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv_expand(out)
        out = self.bn(out)

        return out  
