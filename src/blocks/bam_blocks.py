import torch
import torch.nn as nn
from attention.channel_attention import ChannelAttention
from attention.spatial_attention import SpatialAttention


class BAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16, dilation=4):
        super().__init__()

        self.channel_att = ChannelAttention(
            channels=channels,
            reduction_ratio=reduction_ratio
        )

        self.spatial_att = SpatialAttention(
            channels=channels,
            reduction_ratio=reduction_ratio,
            dilation=dilation
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        mc = self.channel_att(x)      
        ms = self.spatial_att(x)      

        attention = mc + ms
        attention = self.sigmoid(attention)

        out = x + x * attention
        return out
