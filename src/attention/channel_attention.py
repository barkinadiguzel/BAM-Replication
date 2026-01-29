import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()

        reduced_channels = channels // reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True)
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.avg_pool(x)     
        out = self.mlp(out)       
        out = self.bn(out)
        return out                
